# ComfyUI-SCAIL2-Easy

英文文档：[README_EN.md](README_EN.md)

面向 SCAIL-2 的 ComfyUI 简化节点。这个插件不重写 SCAIL-2 推理逻辑，而是复用 ComfyUI 原生节点，把容易接错、重复、难理解的部分封装起来：

- 自动适配视频分辨率到 SCAIL-2 合法尺寸
- 自动处理动作迁移 / 角色替换两种模式的不同接线逻辑
- 自动生成 SCAIL-2 colored mask
- 自动做 CLIP vision 编码
- 自动按 81 帧 + 5 帧重叠分段生成长视频
- 输出交给 VideoHelperSuite 的 `VHS_VideoCombine`

## 节点

插件提供两个节点：

- `SCAIL-2 Fit Video`
- `SCAIL-2 Simple Video`

`SCAIL-2 Fit Video` 只有一个参数：

- `512p`：短边缩放到 512，长边按原比例动态计算，并对齐到 32 的倍数
- `704p`：短边缩放到 704，长边按原比例动态计算，并对齐到 32 的倍数

`SCAIL-2 Simple Video` 是生成主节点。用户主要关注：

- `seed`
- `cfg`
- `mode`
- `max_frames`

## 自带工作流

插件目录包含默认工作流：

```text
workflow/SCAIL2_simple.json
```

## 必需环境

需要较新的 ComfyUI，必须包含原生 SCAIL-2 支持：

```text
feat: Add model support for SCAIL-2
```

还需要以下常用自定义节点：

- ComfyUI-Manager
- ComfyUI-VideoHelperSuite
- ComfyUI-KJNodes
- ComfyUI-SAM3 或当前 ComfyUI 中可用的 `SAM3_VideoTrack`
- 支持 `DiffusionModelLoaderKJ`、`WanChunkFeedForward`、`LoraLoaderModelOnly` 的节点环境

如果默认工作流加载后有红色节点，优先用 ComfyUI-Manager 安装缺失节点。

## 示例目录结构

下面是使用默认工作流时推荐的目录结构：

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

## 使用方法

1. 安装插件到：

```text
ComfyUI/custom_nodes/ComfyUI-SCAIL2-Easy/
```

2. 重启 ComfyUI。

3. 导入：

```text
ComfyUI-SCAIL2-Easy/workflow/SCAIL2_simple.json
```

4. 在 `LoadImage` 里放参考图。

5. 在 `VHS_LoadVideo` 里放驱动视频。

6. 在 `SCAIL-2 Fit Video` 里选择分辨率：

```text
512p
704p
```

7. 在 `SCAIL-2 Simple Video` 里选择模式：

```text
animation：动作迁移
replacement：角色替换
```

8. 使用 `VHS_VideoCombine` 输出视频。

## 两种模式的区别

### animation

动作迁移模式。可以理解为：

```text
让参考图里的人，做驱动视频里的动作。
```

内部逻辑：

- 不使用 SAM mask
- 不抠参考图背景
- CLIP vision 使用原始参考图
- 所有分段都按 animation 方式生成

适合：

- 想让一张角色图动起来
- 驱动视频只是动作来源
- 希望保留参考图人物自己的身材、衣服、画面氛围

### replacement

角色替换模式。可以理解为：

```text
把驱动视频里的人，替换成参考图里的人。
```

内部逻辑：

- 使用 SAM3 的 driving track 和 reference track
- 内部调用原生 `SCAIL2ColoredMask`
- 内部把参考图人物压到黑背景
- CLIP vision 使用处理后的参考图
- 第一段按 replacement 生成
- 后续段使用 previous frames 接力续跑

适合：

- 视频里已经有人在表演
- 想把视频里的人替换成参考角色
- 多人场景需要按位置/颜色对应角色

用户切换模式时只需要改 `SCAIL-2 Simple Video.mode`，不用改线。

## 长视频逻辑

`SCAIL-2 Simple Video` 内部自动分段：

- 每段 81 帧
- 相邻段共享 5 帧历史帧
- 后续段丢掉前 5 帧再拼接，避免重复帧
- 尾段会向下取合法 Wan 帧数

这对应 SCAIL-2 官方的默认思路：

```text
segment_len = 81
segment_overlap = 5
```

如果只想短测：

```text
VHS_LoadVideo.frame_load_cap = 81
SCAIL-2 Simple Video.max_frames = 81 或 0
```

如果跑长视频：

```text
VHS_LoadVideo.frame_load_cap = 0
SCAIL-2 Simple Video.max_frames = 0
```

`0` 表示不额外限制，按读入的视频帧数生成。

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
