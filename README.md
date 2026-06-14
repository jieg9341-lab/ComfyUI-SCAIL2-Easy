# ComfyUI-SCAIL2-Easy

English documentation: [README_EN.md](README_EN.md)

面向 SCAIL-2 的 ComfyUI 简化节点。这个插件不重写 SCAIL-2 推理逻辑，而是复用 ComfyUI 原生 SCAIL-2 支持，把容易接错、重复、理解成本高的部分封装起来。

主要特点：
- 简化 SCAIL-2 工作流，只保留常用控制项
- 支持 `animation` 动作迁移和 `replacement` 角色替换
- 切换模式不需要重新改连线
- 内部自动处理 SCAIL-2 colored mask 和 CLIP vision 编码
- 内置长视频分段生成逻辑
- 支持 `512p`、`704p` 和自定义分辨率
- 支持 Reference Pack 多参考图
- 高级参数默认收起，需要时再展开

## 节点

插件主要提供这些节点：
- `SCAIL-2 Fit Video`
- `SCAIL-2 Simple Video`
- `SCAIL-2 Reference Pack`
- `SCAIL-2 Reference SAM Builder`

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
- `long_video_mode`：长视频方式，默认使用接续分段
- `chunk_frames`：每段生成帧数，默认 `81`
- `overlap_frames`：分段之间的重叠帧数，默认 `5`，可以设为 `0`

### SCAIL-2 Reference Pack

用来组织多参考图。

- `主体数量`：要处理的主体数量，最多 6 个
- `主体N_主图`：每个主体放一张清晰、完整的主参考图
- `参考图数量`：额外参考图数量，最多 5 张
- `参考图N`：可以放不同角度、服装细节、剧情画面或多人画面
- `场景图`：可选，主要用于 `animation` 模式指定背景

多主体时，插件会先把各个主体的主图拼成一张主参考图，再交给 SCAIL-2 使用。这个拼图是为了让 SCAIL-2 在同一张主参考画面里看到多个主体；补充参考图则用于提供更多角度、服装或场景信息。

多主体建议让 `主体1`、`主体2`、`主体3` 的顺序尽量对应驱动视频里的目标人物顺序，这样更容易稳定对应。

### SCAIL-2 Reference SAM Builder

多参考图工作流建议接这个节点。它会配合 Reference Pack 生成 SCAIL-2 需要的参考信息和 colored mask。

如果使用多主体或 `replacement` 模式，驱动视频的 `SAM3_VideoTrack.max_objects` 要设置得足够大，至少覆盖视频里要处理的人物数量。

## 自带工作流

默认工作流在：

```text
workflow/1. SCAIL2_simple.json
workflow/2. SCAIL2_multi_ref.json
```

- `1. SCAIL2_simple.json`：普通单图工作流
- `2. SCAIL2_multi_ref.json`：多参考图工作流

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
│        ├─ 1. SCAIL2_simple.json
│        └─ 2. SCAIL2_multi_ref.json
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
ComfyUI-SCAIL2-Easy/workflow/1. SCAIL2_simple.json
ComfyUI-SCAIL2-Easy/workflow/2. SCAIL2_multi_ref.json
```

4. 在 `LoadImage` 里放参考图。
5. 在 `VHS_LoadVideo` 里放驱动视频。
6. 在 `SCAIL-2 Fit Video` 里选择分辨率。
7. 在 `SCAIL-2 Simple Video` 里选择模式。
8. 用 `VHS_VideoCombine` 输出视频。

## 多参考图用法

使用 `2. SCAIL2_multi_ref.json`：

1. 在 `SCAIL-2 Reference Pack` 里设置 `主体数量`。
2. 给每个主体连接一张 `主体N_主图`。
3. 需要更多角度或细节时，增加 `参考图数量` 并连接 `参考图N`。
4. 如果想指定背景，连接 `场景图`。
5. 检查 `SAM3_VideoTrack.max_objects` 是否覆盖目标人物数量。
6. 在 `SCAIL-2 Simple Video` 里选择 `animation` 或 `replacement` 后运行。

多参考图建议先从少量图片开始测试。主体主图越清晰、人物越完整，通常越稳定。

## 两种模式

### animation

动作迁移模式。可以理解为：

```text
让参考图里的人，做驱动视频里的动作。
```

这个模式更偏向保留参考图人物自身的身材、衣服和画面风格。

如果使用 Reference Pack，主体主图会参与主参考拼图；如果连接了 `场景图`，结果会更倾向使用指定背景。

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
