# 通用 MiniMax H3 可视化视频工作流

这是一个基于 ComfyUI 的通用 MiniMax H3 长视频工作流。它不包含任何已有故事、角色、对白、参考图、声线、工作流实例、生成视频或模型文件；这些内容只在本地项目中创建，并且已被 Git 忽略。

工作流的核心是：按配置生成可在 ComfyUI 中打开和检查的画布；H3 以分镜链生成原生画面与环境声；带对白的镜头使用独立的说话人、身份图和 Audio Spine，降低角色串台和嘴型滞后；成片节点再按实际视频帧对齐旁白、对白、字幕和环境声。

## 已包含

- ComfyUI 源码快照（上游提交 `95d755cd8107a72258d452b5d3657273d571f07d`）
- `custom_nodes/ComfyUI-H3-Multishot/`：H3 多镜头扩展，含本工作流需要的 Ref2VA 分段提示词兼容修正
- `custom_nodes/Workflow2-Visual/`：低内存链合并与按真实帧时轴成片的 ComfyUI 节点
- `story/workflow2/`：通用画布生成、声音参考、Audio Spine、原生声音质检与成片脚本
- `examples/story-project.example.json`：可复制的脱敏项目配置示例

## 安装

1. 按 ComfyUI 的常规方式创建 Python 环境并安装根目录的 `requirements.txt`。
2. 安装 H3 扩展的依赖：`pip install -r custom_nodes/ComfyUI-H3-Multishot/requirements.txt`。
3. 额外安装工作流依赖：`pip install edge-tts soundfile transformers`。系统还需要可用的 `ffmpeg` 与 `ffprobe`。
4. 按 MiniMax H3 和 H3 扩展的说明下载模型；模型位于 `models/`，不会也不应提交到 Git。
5. 将示例配置复制到 `story/workflow2/stories/<你的项目>.json`，准备参考图和可选的声音参考到 `input/story_workflow2/<你的项目>/`。这两个目录均只保留在本地。

## 从可视化画布开始

先检查配置，不启动 ComfyUI：

```bash
.venv/bin/python story/workflow2/run_workflow2.py \
  --config story/workflow2/stories/<你的项目>.json --dry-run
```

生成并打开可视化工作流：

```bash
.venv/bin/python story/workflow2/run_workflow2.py \
  --config story/workflow2/stories/<你的项目>.json --mode visual
```

打开画布后，从左到右检查模型、身份参考、声线参考、带对白镜头的同步音频脊柱、H3 分镜链和最终成片节点。每次换题材时只新建项目 JSON 与本地素材，不需要修改 Python 代码。

## 数据边界

请勿将项目 JSON、人物或声音参考、输入素材、`output/`、`models/`、`user/`、渲染运行目录或成片加入版本控制。仓库的 `.gitignore` 已覆盖这些路径；提交前仍建议执行：

```bash
git status --short
```

## 上游与许可

ComfyUI 保留其原始许可证。`custom_nodes/ComfyUI-H3-Multishot/` 保留其 MIT 许可证和版权声明。MiniMax H3 的模型及其使用条款以官方发布为准，模型文件不随本仓库分发。
