from __future__ import annotations

import gc
import json
from typing import Any

import torch
import torch.nn.functional as F


CATEGORY = "SCAIL-2/Simple"
RESOLUTION_PRESETS = ("512p", "704p", "custom")


def _shape(value: Any) -> list[int]:
    return list(value.shape) if hasattr(value, "shape") else []


def _round_32(value: int) -> int:
    return max(32, (int(value) // 32) * 32)


def _round_nearest_32(value: float) -> int:
    return max(32, int((float(value) + 16) // 32) * 32)


def _wan_frame_count_cover(value: int) -> int:
    value = max(1, int(value))
    if value == 1:
        return 1
    return 1 + ((value - 1 + 3) // 4) * 4


def _wan_frame_count_floor(value: int) -> int:
    value = max(1, int(value))
    return 1 + ((value - 1) // 4) * 4


def _trim_frames(value: torch.Tensor | None, frame_count: int) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("Image inputs must be ComfyUI IMAGE tensors.")
    if value.ndim != 4:
        raise ValueError("Image inputs must have shape [frames, height, width, channels].")
    return value[:frame_count].detach()


def _node_result(value: Any) -> tuple:
    if hasattr(value, "result"):
        result = value.result
        if result is None:
            return ()
        return tuple(result)
    if isinstance(value, tuple):
        return value
    return (value,)


def _infer_generation_size(pose_video: torch.Tensor) -> tuple[int, int]:
    if pose_video.ndim != 4:
        raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
    height = _round_32(int(pose_video.shape[1]))
    width = _round_32(int(pose_video.shape[2]))
    return width, height


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _target_size_for_video(images: torch.Tensor, resolution: str, custom_width: int = 832, custom_height: int = 480) -> tuple[int, int]:
    if images.ndim != 4:
        raise ValueError("video must be a ComfyUI IMAGE tensor.")
    height = int(images.shape[1])
    width = int(images.shape[2])
    if resolution == "custom":
        return _round_nearest_32(max(32, int(custom_width))), _round_nearest_32(max(32, int(custom_height)))

    target_short = 704 if resolution == "704p" else 512
    source_short = max(1, min(width, height))
    scale = target_short / source_short

    target_width = _round_nearest_32(width * scale)
    target_height = _round_nearest_32(height * scale)
    return target_width, target_height


def _resize_to_target(images: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("video must be a ComfyUI IMAGE tensor.")
    source_height = int(images.shape[1])
    source_width = int(images.shape[2])
    if source_width == target_width and source_height == target_height:
        return images.detach().contiguous()

    dtype = images.dtype
    chunks = []

    for start in range(0, int(images.shape[0]), 32):
        chunk = images[start : start + 32].detach().float().movedim(-1, 1)
        chunk = F.interpolate(
            chunk,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        chunks.append(chunk.movedim(1, -1).to(dtype=dtype).contiguous())

    return torch.cat(chunks, dim=0).clamp(0, 1)


def _resize_mask_like(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == image.shape[1] and mask.shape[2] == image.shape[2]:
        return mask.detach()
    resized = F.interpolate(
        mask.detach().float().movedim(-1, 1),
        size=(int(image.shape[1]), int(image.shape[2])),
        mode="nearest",
    )
    return resized.movedim(1, -1).to(dtype=image.dtype).contiguous()


def _apply_reference_mask(reference_image: torch.Tensor, reference_image_mask: torch.Tensor) -> torch.Tensor:
    mask = _resize_mask_like(reference_image_mask[:1], reference_image[:1])
    alpha = (mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(dtype=reference_image.dtype)
    return (reference_image[:1].detach() * alpha).contiguous()


def _run_native_scail_chunk(
    *,
    model,
    positive,
    negative,
    vae,
    sampler,
    sigmas,
    reference_image: torch.Tensor,
    pose_video: torch.Tensor,
    clip_vision_output,
    pose_video_mask: torch.Tensor | None,
    reference_image_mask: torch.Tensor | None,
    previous_frames: torch.Tensor | None,
    width: int,
    height: int,
    length: int,
    video_frame_offset: int,
    previous_frame_count: int,
    replacement_mode: bool,
    seed: int,
    cfg: float,
    pose_strength: float,
) -> tuple[torch.Tensor, int, dict]:
    from comfy_extras.nodes_custom_sampler import SamplerCustom
    from comfy_extras.nodes_scail import WanSCAILToVideo
    import nodes

    scail_out = _node_result(
        WanSCAILToVideo.execute(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            1,
            pose_strength,
            0.0,
            1.0,
            video_frame_offset,
            previous_frame_count,
            replacement_mode=replacement_mode,
            reference_image=reference_image,
            clip_vision_output=clip_vision_output,
            pose_video=pose_video,
            pose_video_mask=pose_video_mask,
            reference_image_mask=reference_image_mask,
            previous_frames=previous_frames,
        )
    )
    if len(scail_out) != 4:
        raise RuntimeError("WanSCAILToVideo returned an unexpected output shape.")

    chunk_positive, chunk_negative, latent, next_offset = scail_out
    sampled = _node_result(
        SamplerCustom.execute(
            model,
            True,
            int(seed),
            float(cfg),
            chunk_positive,
            chunk_negative,
            sampler,
            sigmas,
            latent,
        )
    )
    if not sampled:
        raise RuntimeError("SamplerCustom returned no latent output.")

    # The native reference workflow decodes denoised_output when available.
    latent_to_decode = sampled[1] if len(sampled) > 1 else sampled[0]
    frames = nodes.VAEDecode().decode(vae, latent_to_decode)[0].detach().clamp(0, 1)
    summary = {
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "seed": int(seed),
        "cfg": float(cfg),
        "input_video_frame_offset": int(video_frame_offset),
        "output_video_frame_offset": int(next_offset),
        "decoded_shape": _shape(frames),
    }
    return frames, int(next_offset), summary


def _create_scail_masks(driving_track_data, reference_track_data, replacement_mode: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from comfy_extras.nodes_scail import SCAIL2ColoredMask

    masks = _node_result(
        SCAIL2ColoredMask.execute(
            driving_track_data,
            "",
            "left_to_right",
            bool(replacement_mode),
            ref_track_data=reference_track_data,
        )
    )
    if len(masks) != 2:
        raise RuntimeError("SCAIL2ColoredMask returned an unexpected output shape.")
    return masks[0], masks[1]


def _encode_clip_vision(clip_vision, image: torch.Tensor):
    import nodes

    return nodes.CLIPVisionEncode().encode(clip_vision, image, "none")[0]


class SCAIL2SimpleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "reference_image": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "clip_vision": ("CLIP_VISION",),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "mode": (["replacement", "animation"], {"default": "replacement"}),
                "advanced": ("BOOLEAN", {"default": False}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "chunk_frames": ("INT", {"default": 81, "min": 17, "max": 321, "step": 4}),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 33, "step": 1}),
            },
            "optional": {
                "driving_track_data": ("SAM3_TRACK_DATA",),
                "reference_track_data": ("SAM3_TRACK_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "summary")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        model,
        positive,
        negative,
        vae,
        sampler,
        sigmas,
        reference_image: torch.Tensor,
        pose_video: torch.Tensor,
        clip_vision,
        seed: int,
        cfg: float,
        mode: str,
        advanced: bool,
        max_frames: int,
        chunk_frames: int = 81,
        overlap_frames: int = 5,
        driving_track_data=None,
        reference_track_data=None,
    ):
        if reference_image.ndim != 4 or pose_video.ndim != 4:
            raise ValueError("reference_image and pose_video must be ComfyUI IMAGE tensors.")

        total_frames = int(pose_video.shape[0])
        if max_frames > 0:
            total_frames = min(total_frames, int(max_frames))
        if total_frames <= 0:
            raise ValueError("pose_video has no frames.")

        mask_replacement_mode = mode == "replacement"
        pose_video_mask = None
        reference_image_mask = None
        if mask_replacement_mode:
            if driving_track_data is None or reference_track_data is None:
                raise ValueError("replacement mode requires driving_track_data and reference_track_data from SAM3_VideoTrack.")
            pose_video_mask, reference_image_mask = _create_scail_masks(
                driving_track_data,
                reference_track_data,
                True,
            )
        pose_video = _trim_frames(pose_video, total_frames)
        reference_image = reference_image[:1].detach()
        pose_video_mask = _trim_frames(pose_video_mask, total_frames)
        if reference_image_mask is not None:
            reference_image_mask = reference_image_mask[:1].detach()
            reference_image = _apply_reference_mask(reference_image, reference_image_mask)
        clip_vision_output = _encode_clip_vision(clip_vision, reference_image)

        width, height = _infer_generation_size(pose_video)
        first_chunk_replacement_mode = mask_replacement_mode
        pose_strength = 1.0
        chunk_frames = _wan_frame_count_cover(max(17, int(chunk_frames)))
        first_length = chunk_frames
        requested_overlap = max(0, int(overlap_frames))
        previous_frame_count = 0 if requested_overlap == 0 else max(1, min(_wan_frame_count_floor(requested_overlap), 33, max(1, chunk_frames - 4)))
        if chunk_frames <= previous_frame_count:
            raise ValueError("chunk_frames must be larger than overlap_frames.")

        stitched: list[torch.Tensor] = []
        chunk_summaries: list[dict[str, Any]] = []
        previous_frames = None
        video_frame_offset = 0
        produced = 0
        chunk_index = 0
        max_chunks = max(1, (total_frames // max(1, chunk_frames - previous_frame_count)) + 4)

        while produced < total_frames and chunk_index < max_chunks:
            anchor_start = max(0, video_frame_offset - (previous_frame_count if previous_frames is not None else 0))
            remaining_from_anchor = max(1, total_frames - anchor_start)
            target_length = first_length if chunk_index == 0 else chunk_frames
            length = min(target_length, _wan_frame_count_floor(remaining_from_anchor))
            if previous_frames is not None and previous_frame_count > 0 and length <= previous_frame_count:
                break

            decoded, next_offset, chunk_summary = _run_native_scail_chunk(
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                sampler=sampler,
                sigmas=sigmas,
                reference_image=reference_image,
                pose_video=pose_video,
                clip_vision_output=clip_vision_output,
                pose_video_mask=pose_video_mask,
                reference_image_mask=reference_image_mask,
                previous_frames=previous_frames,
                width=width,
                height=height,
                length=length,
                video_frame_offset=video_frame_offset,
                previous_frame_count=previous_frame_count,
                replacement_mode=first_chunk_replacement_mode and chunk_index == 0,
                seed=int(seed) + chunk_index,
                cfg=float(cfg),
                pose_strength=float(pose_strength),
            )

            discard_head = 0 if chunk_index == 0 else min(previous_frame_count, int(decoded.shape[0]))
            kept = decoded[discard_head:].contiguous()
            remaining_output = total_frames - produced
            if kept.shape[0] > remaining_output:
                kept = kept[:remaining_output].contiguous()
            if kept.shape[0] <= 0:
                raise RuntimeError("A SCAIL-2 chunk produced no keepable frames.")

            stitched.append(kept.cpu())
            produced += int(kept.shape[0])
            chunk_summary.update(
                {
                    "chunk_index": chunk_index,
                    "replacement_mode": bool(first_chunk_replacement_mode and chunk_index == 0),
                    "discard_head": int(discard_head),
                    "kept_frames": int(kept.shape[0]),
                    "produced_total": int(produced),
                }
            )
            chunk_summaries.append(chunk_summary)

            previous_frames = decoded[-previous_frame_count:].detach().cpu() if previous_frame_count > 0 else None
            video_frame_offset = int(next_offset)
            chunk_index += 1
            del decoded, kept
            _empty_cache()

        frames = torch.cat(stitched, dim=0).contiguous().clamp(0, 1)
        summary = {
            "mode": mode,
            "mask_replacement_mode": bool(mask_replacement_mode),
            "first_chunk_replacement_mode": bool(first_chunk_replacement_mode),
            "extension_chunk_replacement_mode": False,
            "frames": _shape(frames),
            "width": width,
            "height": height,
            "source_frames": int(total_frames),
            "generated_frames": int(frames.shape[0]),
            "dropped_tail_frames": int(total_frames - frames.shape[0]),
            "chunks": len(chunk_summaries),
            "first_chunk_frames": int(first_length),
            "chunk_frames": int(chunk_frames),
            "overlap_frames": int(previous_frame_count),
            "cfg": float(cfg),
            "seed_start": int(seed),
            "core": "ComfyUI native WanSCAILToVideo -> SamplerCustom -> VAEDecode",
            "chunk_details": chunk_summaries,
        }
        return (frames, json.dumps(summary, indent=2))


class SCAIL2FitVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "resolution": (list(RESOLUTION_PRESETS), {"default": "512p"}),
                "custom_width": ("INT", {"default": 832, "min": 32, "max": 4096, "step": 32}),
                "custom_height": ("INT", {"default": 480, "min": 32, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "summary")
    FUNCTION = "fit"
    CATEGORY = CATEGORY

    def fit(self, video: torch.Tensor, resolution: str, custom_width: int = 832, custom_height: int = 480):
        target_width, target_height = _target_size_for_video(video, resolution, custom_width, custom_height)
        fitted = _resize_to_target(video, target_width, target_height)
        summary = {
            "resolution": resolution,
            "custom_width": int(custom_width),
            "custom_height": int(custom_height),
            "source": _shape(video),
            "target": _shape(fitted),
            "width": target_width,
            "height": target_height,
            "fit": "preserve_aspect_resize_to_nearest_32",
        }
        return (fitted, target_width, target_height, json.dumps(summary, indent=2))


NODE_CLASS_MAPPINGS = {
    "SCAIL2FitVideo": SCAIL2FitVideo,
    "SCAIL2SimpleVideo": SCAIL2SimpleVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2FitVideo": "SCAIL-2 Fit Video",
    "SCAIL2SimpleVideo": "SCAIL-2 Simple Video",
}
