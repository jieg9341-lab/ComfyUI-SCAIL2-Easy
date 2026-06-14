from __future__ import annotations

import logging
from typing import Any

import torch


_SCAIL_PATCHED = False


def _copy_cond(cond_value: Any, tensor: torch.Tensor) -> Any:
    if hasattr(cond_value, "_copy_with"):
        return cond_value._copy_with(tensor)
    return tensor


def _cond_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "cond") and isinstance(value.cond, torch.Tensor):
        return value.cond
    return None


def _take_dim(tensor: torch.Tensor, dim: int, indexes: list[int], device=None) -> torch.Tensor:
    if not indexes:
        return tensor
    if tensor.size(dim) == len(indexes):
        return tensor.to(device)
    if max(indexes) >= tensor.size(dim):
        return tensor.to(device)
    slices = [slice(None)] * tensor.ndim
    slices[dim] = indexes
    return tensor[tuple(slices)].to(device)


def _window_indexes(window) -> list[int]:
    indexes = [int(i) for i in getattr(window, "index_list", [])]
    anchor = getattr(window, "causal_anchor_index", None)
    if anchor is not None and int(anchor) >= 0:
        return [int(anchor)] + indexes
    return indexes


def _mask_time_dim(tensor: torch.Tensor, prefer_ref: bool) -> int | None:
    if tensor.ndim < 5:
        return None
    if tensor.shape[1] == 28:
        return 2
    if tensor.shape[2] == 28:
        return 1
    if prefer_ref and tensor.shape[1] > tensor.shape[2]:
        return 1
    return 2


def _slice_driving_mask(tensor: torch.Tensor, indexes: list[int], device=None) -> torch.Tensor:
    dim = _mask_time_dim(tensor, prefer_ref=False)
    if dim is None:
        return tensor.to(device)
    return _take_dim(tensor, dim, indexes, device=device)


def _slice_reference_mask(tensor: torch.Tensor, reference_frames: int, indexes: list[int], device=None) -> torch.Tensor:
    dim = _mask_time_dim(tensor, prefer_ref=True)
    if dim is None or reference_frames <= 0:
        return tensor.to(device)
    total_needed = reference_frames + len(indexes)
    if tensor.size(dim) == total_needed:
        return tensor.to(device)
    if tensor.size(dim) <= reference_frames:
        return tensor.to(device)

    ref_slice = [slice(None)] * tensor.ndim
    ref_slice[dim] = slice(0, reference_frames)
    video_slice = [slice(None)] * tensor.ndim
    video_slice[dim] = slice(reference_frames, None)
    ref_part = tensor[tuple(ref_slice)]
    video_part = tensor[tuple(video_slice)]
    video_part = _take_dim(video_part, dim, indexes, device=tensor.device)
    return torch.cat([ref_part, video_part], dim=dim).to(device)


def _reference_frame_count(model_conds: dict[str, Any]) -> int:
    for key in ("reference_latent", "reference_latents"):
        tensor = _cond_tensor(model_conds.get(key))
        if tensor is not None and tensor.ndim >= 3:
            return int(tensor.shape[2])
    return 0


def _resize_scail_model_conds(model_conds: dict[str, Any], window, device=None) -> dict[str, Any]:
    resized = model_conds.copy()
    indexes = _window_indexes(window)
    reference_frames = _reference_frame_count(model_conds)

    for key in ("driving_mask_28ch", "driving_mask_latents"):
        tensor = _cond_tensor(model_conds.get(key))
        if tensor is not None:
            resized[key] = _copy_cond(model_conds[key], _slice_driving_mask(tensor, indexes, device=device))

    for key in ("ref_mask_28ch", "ref_mask_latents"):
        tensor = _cond_tensor(model_conds.get(key))
        if tensor is not None:
            resized[key] = _copy_cond(
                model_conds[key],
                _slice_reference_mask(tensor, reference_frames, indexes, device=device),
            )

    return resized


def _patch_scail_forward():
    global _SCAIL_PATCHED
    if _SCAIL_PATCHED:
        return

    try:
        import comfy.ldm.wan.model as wan_model
    except Exception as exc:
        logging.warning("[SCAIL2-Easy] Could not import Wan model for context patch: %s", exc)
        return

    model_cls = getattr(wan_model, "SCAILWanModel", None)
    original = getattr(model_cls, "_forward", None)
    if model_cls is None or original is None:
        return
    if getattr(original, "_scail2_easy_context_patch", False):
        _SCAIL_PATCHED = True
        return

    def patched_forward(
        self,
        x,
        timestep,
        context,
        clip_fea=None,
        time_dim_concat=None,
        transformer_options=None,
        pose_latents=None,
        **kwargs,
    ):
        options = transformer_options or {}
        window = options.get("context_window")
        if window is not None and hasattr(window, "index_list"):
            indexes = _window_indexes(window)
            reference_latent = kwargs.get("reference_latent")
            reference_frames = int(reference_latent.shape[2]) if isinstance(reference_latent, torch.Tensor) and reference_latent.ndim >= 3 else 0

            for key in ("driving_mask_28ch", "driving_mask_latents"):
                value = kwargs.get(key)
                if isinstance(value, torch.Tensor):
                    kwargs[key] = _slice_driving_mask(value, indexes, device=value.device)

            for key in ("ref_mask_28ch", "ref_mask_latents"):
                value = kwargs.get(key)
                if isinstance(value, torch.Tensor):
                    kwargs[key] = _slice_reference_mask(value, reference_frames, indexes, device=value.device)

        return original(
            self,
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            time_dim_concat=time_dim_concat,
            transformer_options=options,
            pose_latents=pose_latents,
            **kwargs,
        )

    patched_forward._scail2_easy_context_patch = True
    model_cls._forward = patched_forward
    _SCAIL_PATCHED = True


def _latent_frames(pixel_frames: int) -> int:
    return max(1, ((max(1, int(pixel_frames)) - 1) // 4) + 1)


def apply_scail2_easy_context(model, context_frames: int, context_overlap_frames: int):
    try:
        import comfy.context_windows as context_windows
    except Exception as exc:
        raise RuntimeError("This ComfyUI build does not provide comfy.context_windows, so context sampling is unavailable.") from exc

    _patch_scail_forward()

    context_length = _latent_frames(context_frames)
    context_overlap = 0 if int(context_overlap_frames) <= 0 else _latent_frames(context_overlap_frames)
    if context_overlap >= context_length:
        raise ValueError("context_overlap_frames must be smaller than context_frames.")

    class SCAIL2EasyContextHandler(context_windows.IndexListContextHandler):
        def get_resized_cond(self, cond_in, x_in, window, device=None):
            resized = super().get_resized_cond(cond_in, x_in, window, device)
            if resized is None:
                return None
            for cond in resized:
                model_conds = cond.get("model_conds")
                if isinstance(model_conds, dict):
                    cond["model_conds"] = _resize_scail_model_conds(model_conds, window, device=device)
            return resized

    model = model.clone()
    handler_kwargs = {
        "context_schedule": context_windows.get_matching_context_schedule(context_windows.ContextSchedules.STATIC_STANDARD),
        "fuse_method": context_windows.get_matching_fuse_method(context_windows.ContextFuseMethods.PYRAMID),
        "context_length": context_length,
        "context_overlap": context_overlap,
        "context_stride": 1,
        "closed_loop": False,
        "dim": 2,
        "freenoise": False,
        "cond_retain_index_list": "",
        "split_conds_to_windows": False,
    }
    try:
        handler = SCAIL2EasyContextHandler(**handler_kwargs)
    except TypeError:
        handler_kwargs.pop("cond_retain_index_list", None)
        handler = SCAIL2EasyContextHandler(**handler_kwargs)
    model.model_options["context_handler"] = handler
    context_windows.create_prepare_sampling_wrapper(model)

    return model, {
        "context_frames": int(context_frames),
        "context_overlap_frames": int(context_overlap_frames),
        "context_latent_frames": int(context_length),
        "context_overlap_latent_frames": int(context_overlap),
        "context_schedule": context_windows.ContextSchedules.STATIC_STANDARD,
        "fuse_method": context_windows.ContextFuseMethods.PYRAMID,
    }
