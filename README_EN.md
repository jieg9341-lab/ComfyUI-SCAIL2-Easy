# ComfyUI-SCAIL2-Easy

Chinese documentation: [README.md](README.md)

A simplified ComfyUI helper plugin for SCAIL-2. This plugin does not replace ComfyUI's native SCAIL-2 implementation. It reuses native ComfyUI nodes and wraps the parts that are repetitive or easy to wire incorrectly:

- Fit input videos to valid SCAIL-2 dimensions
- Hide the wiring differences between animation and replacement modes
- Generate SCAIL-2 colored masks internally
- Run CLIP vision encoding internally
- Generate long videos with automatic 81-frame chunks and 5-frame overlap
- Output frames through VideoHelperSuite `VHS_VideoCombine`

## Nodes

This plugin provides two nodes:

- `SCAIL-2 Fit Video`
- `SCAIL-2 Simple Video`

`SCAIL-2 Fit Video` has one parameter:

- `512p`: scale the short side to 512, calculate the long side from the original aspect ratio, then align both dimensions to multiples of 32
- `704p`: scale the short side to 704, calculate the long side from the original aspect ratio, then align both dimensions to multiples of 32

`SCAIL-2 Simple Video` is the main generation node. The main user-facing controls are:

- `seed`
- `cfg`
- `mode`
- `max_frames`

## Included Workflow

The default workflow is included here:

```text
workflow/SCAIL2_simple.json
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
│     └─ workflow/
│        └─ SCAIL2_simple.json
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
ComfyUI-SCAIL2-Easy/workflow/SCAIL2_simple.json
```

4. Load the reference image in `LoadImage`.

5. Load the driving video in `VHS_LoadVideo`.

6. Select the resolution in `SCAIL-2 Fit Video`:

```text
512p
704p
```

7. Select the mode in `SCAIL-2 Simple Video`:

```text
animation: motion transfer
replacement: character replacement
```

8. Use `VHS_VideoCombine` to output the video.

## Modes

### animation

Motion transfer mode. In simple terms:

```text
Make the person in the reference image perform the motion from the driving video.
```

Internal behavior:

- Does not use SAM masks
- Keeps the reference image background
- Uses the original reference image for CLIP vision
- Runs every chunk in animation mode

Useful when:

- You want to animate a character from a still image
- The driving video is only used as a motion source
- You want to preserve the proportions, clothing, and visual context of the reference image

### replacement

Character replacement mode. In simple terms:

```text
Replace the person in the driving video with the person from the reference image.
```

Internal behavior:

- Uses SAM3 driving and reference tracks
- Calls native `SCAIL2ColoredMask` internally
- Masks the reference person onto a black background
- Uses the masked reference image for CLIP vision
- Runs the first chunk in replacement mode
- Runs extension chunks with previous frames as continuity anchors

Useful when:

- The driving video already contains a person performing the motion
- You want to replace that person with the reference character
- Multi-person scenes need identity correspondence

Users only need to change `SCAIL-2 Simple Video.mode`. No rewiring is needed when switching modes.

## Long Video Logic

`SCAIL-2 Simple Video` automatically chunks the driving video:

- 81 frames per chunk
- 5 shared history frames between adjacent chunks
- Extension chunks drop their first 5 repeated frames before stitching
- The final chunk is rounded down to a valid Wan frame count

This follows the default SCAIL-2 idea:

```text
segment_len = 81
segment_overlap = 5
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
