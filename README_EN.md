# ComfyUI-SCAIL2-Easy

Chinese documentation: [README.md](README.md)

A simplified ComfyUI helper plugin for SCAIL-2. This plugin does not replace ComfyUI's native SCAIL-2 implementation. It reuses native ComfyUI support and wraps the parts that are repetitive or easy to wire incorrectly:

- Fit input videos to valid SCAIL-2 dimensions
- Hide the wiring differences between animation and replacement modes
- Generate SCAIL-2 colored masks internally
- Run CLIP vision encoding internally
- Generate long videos with automatic chunks and overlap
- Support `512p`, `704p`, and custom resolution
- Support multi-reference Reference Packs
- Output frames through VideoHelperSuite `VHS_VideoCombine`

## Nodes

This plugin mainly provides these nodes:

- `SCAIL-2 Fit Video`
- `SCAIL-2 Simple Video`
- `SCAIL-2 Reference Pack`
- `SCAIL-2 Reference SAM Builder`

`SCAIL-2 Fit Video` has one main parameter:

- `512p`: scale the short side to 512, calculate the long side from the original aspect ratio, then align both dimensions to multiples of 32
- `704p`: scale the short side to 704, calculate the long side from the original aspect ratio, then align both dimensions to multiples of 32
- `custom`: expose `custom_width` and `custom_height`

`SCAIL-2 Simple Video` is the main generation node. The main user-facing controls are:

- `seed`
- `cfg`
- `mode`
- `max_frames`

Advanced controls are hidden by default. They include chunk size, overlap frames, and long-video mode.

`SCAIL-2 Reference Pack` organizes multi-reference inputs:

- Up to 6 subjects
- One main image for each subject
- Up to 5 extra reference images
- Optional scene image for animation background guidance

For multi-subject packs, the subject main images are composed into one primary reference collage before being used by SCAIL-2. Extra reference images provide additional views, clothing details, story frames, or scene information.

`SCAIL-2 Reference SAM Builder` is recommended for multi-reference workflows. For multi-subject or replacement mode, set `SAM3_VideoTrack.max_objects` high enough to cover all target people in the driving video.

## Included Workflow

The included workflows are:

```text
workflow/1. SCAIL2_simple.json
workflow/2. SCAIL2_multi_ref.json
```

## Requirements

Use a recent ComfyUI build that includes native SCAIL-2 support:

```text
feat: Add model support for SCAIL-2
```

The default workflow also expects these common custom nodes:

- ComfyUI-Manager
- ComfyUI-VideoHelperSuite
- ComfyUI-KJNodes
- ComfyUI-SAM3 or an environment that provides `SAM3_VideoTrack`
- Nodes that provide `DiffusionModelLoaderKJ`, `WanChunkFeedForward`, and `LoraLoaderModelOnly`

If the workflow opens with red nodes, install the missing nodes through ComfyUI-Manager.

## Example Directory Layout

Recommended layout for the default workflow:

```text
ComfyUI/
├─ custom_nodes/
│  └─ ComfyUI-SCAIL2-Easy/
│     ├─ __init__.py
│     ├─ nodes.py
│     ├─ requirements.txt
│     ├─ README.md
│     ├─ README_EN.md
│     ├─ LICENSE
│     ├─ web/
│     │  └─ scail2_easy.js
│     └─ workflow/
│        ├─ 1. SCAIL2_simple.json
│        └─ 2. SCAIL2_multi_ref.json
└─ models/
   ├─ diffusion_models/
   │  └─ wan2.1_14B_SCAIL_2_fp8_scaled.safetensors
   ├─ text_encoders/
   │  └─ umt5_xxl_fp8_e4m3fn_scaled.safetensors
   ├─ clip_vision/
   │  └─ clip_vision_vit_h.safetensors
   ├─ vae/
   │  └─ Wan2_1_VAE_bf16.safetensors
   ├─ loras/
   │  └─ Lightx2v/
   │     └─ lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors
   └─ checkpoints/
      └─ sam3.1_multiplex_fp16.safetensors
```

## Usage

1. Install the plugin into:

```text
ComfyUI/custom_nodes/ComfyUI-SCAIL2-Easy/
```

2. Restart ComfyUI.

3. Import:

```text
ComfyUI-SCAIL2-Easy/workflow/1. SCAIL2_simple.json
ComfyUI-SCAIL2-Easy/workflow/2. SCAIL2_multi_ref.json
```

4. Load the reference image in `LoadImage`.

5. Load the driving video in `VHS_LoadVideo`.

6. Select the resolution in `SCAIL-2 Fit Video`.

7. Select the mode in `SCAIL-2 Simple Video`.

8. Use `VHS_VideoCombine` to output the video.

## Multi-Reference Usage

Use `2. SCAIL2_multi_ref.json`:

1. Set `subject_count` in `SCAIL-2 Reference Pack`.
2. Connect one main image for each subject.
3. Increase `reference_count` if you need extra views or details.
4. Connect `scene_image` if you want background guidance.
5. Check that `SAM3_VideoTrack.max_objects` covers the target people.
6. Select `animation` or `replacement` in `SCAIL-2 Simple Video` and run.

Start with a small number of reference images. Clear full-body subject images are usually more stable.

## Modes

### animation

Motion transfer mode. In simple terms:

```text
Make the person in the reference image perform the motion from the driving video.
```

This mode tends to preserve the body shape, outfit, and visual style of the reference image.

If you use a Reference Pack, the subject main images are used in the primary reference collage. If `scene_image` is connected, the result will lean more toward that background.

### replacement

Character replacement mode. In simple terms:

```text
Replace the person in the driving video with the person from the reference image.
```

This mode uses SAM3 track data and generates the colored masks required by SCAIL-2. It follows the position and proportion of the person in the driving video more closely.

Users only need to change `SCAIL-2 Simple Video.mode`. No rewiring is needed when switching modes.

## Long Video Logic

`SCAIL-2 Simple Video` automatically chunks the driving video:

```text
chunk_frames = 81
overlap_frames = 5
```

For a short test:

```text
VHS_LoadVideo.frame_load_cap = 81
SCAIL-2 Simple Video.max_frames = 81 or 0
```

For long videos:

```text
VHS_LoadVideo.frame_load_cap = 0
SCAIL-2 Simple Video.max_frames = 0
```

`0` means no extra frame limit.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
