# ComfyUI-SCAIL2-Easy

English documentation: [README_EN.md](README_EN.md)

面向 SCAIL-2 的 ComfyUI 简化节点。这个插件不重写 SCAIL-2 推理逻辑，而是复用 ComfyUI 原生 SCAIL-2 节点，把容易接错、重复、理解成本高的部分封装起来。

主要特点：

- 简化 SCAIL-2 工作流，只保留常用控制项
- 支持 `animation` 动作迁移和 `replacement` 角色替换
- 切换模式不需要重新改连线
- 内部自动处理 SCAIL-2 colored mask 和 CLIP vision 编码
- 内置长视频分段生成逻辑
- 支持 `512p`、`704p` 和自定义分辨率
- 高级参数默认收起，需要时再展开

## 节点

插件提供两个节点：

- `SCAIL-2 Fit Video`
- `SCAIL-2 Simple Video`

### SCAIL-2 Fit Video

用来统一处理输入视频尺寸。

分辨率选项：

- `512p`：短边缩放到 512，长边按原比例动态计算，并对齐到 32 的倍数
- `704p`：短边缩放到 704，长边按原比例动态计算，并对齐到 32 的倍数
- `custom`：显示 `custom_width` 和 `custom_height`，手动指定输出尺寸

选择 `512p` 或 `704p` 时，宽高参数会自动隐藏。

### SCAIL-2 Simple Video

这是主要生成节点。

常用参数：

- `seed`
- `cfg`
- `mode`

模式：

- `animation`：动作迁移
- `replacement`：角色替换

高级参数默认隐藏。打开 `advanced` 后会显示：

- `max_frames`：最大生成帧数，`0` 表示不额外限制
- `chunk_frames`：每段生成帧数，默认 `81`
- `overlap_frames`：分段之间的重叠帧数，默认 `5`，可以设为 `0`

## 自带工作流

默认工作流在：

```text
workflow/SCAIL2_simple.json
```

## 必需环境

需要较新的 ComfyUI，必须包含原生 SCAIL-2 支持：

```text
feat: Add model support for SCAIL-2
```

默认工作流还会用到这些常见节点：

- ComfyUI-VideoHelperSuite
- ComfyUI-KJNodes
- ComfyUI-SAM3 或当前 ComfyUI 中可用的 `SAM3_VideoTrack`
- 支持 `DiffusionModelLoaderKJ`、`WanChunkFeedForward`、`LoraLoaderModelOnly` 的节点环境

如果工作流打开后有红色节点，优先用 ComfyUI-Manager 安装缺失节点。

## 示例目录结构

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
│        └─ SCAIL2_simple.json
└─ models/
   ├─ diffusion_models/
   ├─ text_encoders/
   ├─ clip_vision/
   ├─ vae/
   ├─ loras/
   └─ checkpoints/
```

## 使用方法

1. 安装插件到：

```text
ComfyUI/custom_nodes/ComfyUI-SCAIL2-Easy/
```

2. 重启 ComfyUI。

3. 导入工作流：

```text
ComfyUI-SCAIL2-Easy/workflow/SCAIL2_simple.json
```

4. 在 `LoadImage` 里放参考图。

5. 在 `VHS_LoadVideo` 里放驱动视频。

6. 在 `SCAIL-2 Fit Video` 里选择分辨率。

7. 在 `SCAIL-2 Simple Video` 里选择模式。

8. 用 `VHS_VideoCombine` 输出视频。

## 两种模式

### animation

动作迁移模式。可以理解为：

```text
让参考图里的人，做驱动视频里的动作。
```

这个模式更偏向保留参考图人物自身的身材、衣服和画面风格。

### replacement

角色替换模式。可以理解为：

```text
把驱动视频里的人，替换成参考图里的角色。
```

这个模式会使用 SAM3 track data，并在内部生成 SCAIL-2 需要的 colored mask。它更贴合驱动视频里人物的位置和比例。

用户切换模式时只需要改 `SCAIL-2 Simple Video.mode`，不用改线。

## 长视频逻辑

`SCAIL-2 Simple Video` 内部自动分段生成并拼接。

默认设置：

```text
chunk_frames = 81
overlap_frames = 5
```

如果打开高级参数，可以自行修改：

- `chunk_frames` 越大，单段越长，但显存和稳定性压力也更高
- `overlap_frames` 越大，段间上下文越多，但会增加重复计算
- `overlap_frames = 0` 表示不使用上一段历史帧作为上下文

短测试可以把 `VHS_LoadVideo.frame_load_cap` 或 `SCAIL-2 Simple Video.max_frames` 设为 `81`。

长视频一般保持：

```text
VHS_LoadVideo.frame_load_cap = 0
SCAIL-2 Simple Video.max_frames = 0
```

`0` 表示不额外限制，按读取到的视频帧数生成。

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
