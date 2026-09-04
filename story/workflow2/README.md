# MiniMax H3 通用长叙事视频工作流

工作流2是通用的“配置驱动 + ComfyUI 可视化”工作流，不绑定任何单一故事。准备一份自己的 JSON 配置并传给同一个脚本即可，无需复制或修改 Python 代码。本仓库只保存通用工作流代码；具体故事配置、身份素材和生成媒体留在本地。

## 运行

```bash
cd /home/jx/codes/comfyui-minimax-h3
.venv/bin/python story/workflow2/run_workflow2.py \
  --config /absolute/path/to/your_story.json
```

默认会：

- 生成标准 ComfyUI 画布 JSON（含 `nodes`、`links` 和分组）；
- 放入 ComfyUI 的个人工作流库 `MiniMax-H3-Workflow2/`；
- 启动本机 ComfyUI 并打开页面；
- 在一个画布中显示模型、参考素材、每条 H3 分镜链、分链预览、A/V 合并和最终保存节点。

在页面左侧工作流库中打开 `MiniMax-H3-Workflow2/<slug>_Workflow2_可视化`，检查或调整节点后点击“运行”。最终原生视频由画布最右侧的“保存最终原生 AI 视频”节点输出。

运行前可只检查配置和输出计划：

```bash
.venv/bin/python story/workflow2/run_workflow2.py \
  --config /absolute/path/to/your_story.json --dry-run
```

保留无人值守的后台 API 模式，但它不再是默认入口：

```bash
.venv/bin/python story/workflow2/run_workflow2.py \
  --config /absolute/path/to/your_story.json --mode api
```

输出按故事隔离在 `story/workflow2/runs/<slug>/`，成片位于其中的 `final/`。默认成片是竖屏 1080×1920；可在配置中调整输出尺寸、帧率、镜头数量和文件名。每个 H3 镜头最多 15 秒，3–4 分钟成片应使用 12–16 个镜头（24fps 下每镜头通常为 360 帧）。

## 配置结构

脚本要求显式传入 `--config`，支持以下通用字段（本地故事示例配置不随仓库发布）：

- `title`、可选 `slug`、`subtitle`、`lesson`、`minimum_duration`、`output_filename`
- `video`（或兼容的顶层 `width`、`height`、`fps`、`frames_per_shot`、`steps`、`sampler`、`scheduler`）
- `identity_policy`：新故事必填的独立身份命名空间（`scope=story`、`namespace=story:<slug>`、`require_unique_story_cast=true`）。工作流会写入 `story_identity.json`，并在 `identity_registry.json` 中登记角色身份和参考图哈希，拒绝跨故事复用。
- `assets.directory` 和可选的 `assets.identity_reference` / `assets.identity_references`：身份/画面参考图。参考图只能放在当前故事自己的资产目录；不再使用公共 `identity_reference.png`。
- `subjects`：稳定角色、物件或场景参考，每项有 `id`、`description` 和（新故事）带故事 slug 的 `identity_key`。没有参考图时，H3 依据本故事角色特性生成独立人物。
- `voice_references`：H3 音频参考，每项可配置 `file`、`voice`、`sample_text`、语速、音调和音量
- `speaker_map`、`default_voice_reference`、`narration_voice`
- `bgm`（独立背景音乐轨：`enabled`、`source_type`、`source_url`/`file`、`sha256`、`volume`、`fade_in`、`fade_out`、`duck_ratio`；`source_type=download` 会下载到故事自己的缓存并校验哈希，`source_type=file` 使用已准备好的曲目；`source_type=generated` 只有显式 `allow_synthetic_fallback=true` 才允许）
- `prompting`（可选通用电影化导演指令：`screenplay` 控制镜头、节拍、空间和风格，`dialogue` 控制对白的说话人、反应节拍和口型约束；不绑定任何具体故事）
- `generation`（通用生成策略：默认 `scene_mode=single_shot`，每个分镜独立生成后再按剧本顺序回拼；`continuity=cut` 不依赖额外插件；安装并确认 Motion Context 后才可改为 `context_pin`）
- `chains`：按顺序把镜头分成若干条短链，每条最多引用三个声音参考；在 `scene_mode=single_shot` 下，链只是成片顺序，不会把多个叙事事件塞进一次 H3 采样
- `scenes`：每个镜头至少有 `id`、`narration`、`narration_offset`、`action_timing`、`sound_design`；建议补齐 `visual_action`、`subject_ids`、`content_contract.must_show/must_not_show`、`dialogue` 和 `shot_contract`
- `shot_contract`（高风险动作的通用安全契约：可设 `isolated`、`primary_action`、`action_owner`、`rider`、`falling_subject`、分段 `beats`、`hand_plan` 与 `forbidden`；工作流会自动拆出独立 H3 上下文，锁定动作归属、演员/动物数量、物理接触和手脚数量）

对白建议明确填写 `speaker`、`start`、`text`，必要时填写 `speaker_id`、`subject_id` 和 `language`。提示词会保留原文对白，并将其放在 H3 官方 Ref2VA 的 `subject_definitions → summary → retention_analysis → detailed_description → overall_soundscape → non_diegetic_music` 结构中。

## 音视频策略

可视化画布保留每条链独立生成、参考图锁定、最多三路人物声音参考、原生 H3 画面与声音、分链视频/音频预览，并通过低内存路径合并节点生成最终视频。配置 `shot_contract.isolated=true` 的脱缰、落马、接触和多人动作会在画布和 API 中自动拆成独立 H3 采样上下文，避免前一镜头的骑手、道具或肢体状态串入后一镜头。配置 `bgm` 后，画布会出现独立的“BGM · 最终混音闪避”音频节点，接入精准时轴成片节点；混音会在旁白和明确对白出现时自动压低 BGM，同时保留 H3 的环境声和拟音。`generate_bgm.py` 支持 `ye_gong`、`nan_yuan`、`cup_snake`、`paoding`、`zhuang_zhou` 五种旋律/速度/配器 profile，便于不同题材使用不同的古风器乐底色。后台 API 模式使用相同的 BGM、闪避、旁白、对白回退、音频质检、字幕和片尾合成逻辑，适合确认画布参数后无人值守批量生产。

优先使用明确来源、与故事气质匹配的现成古风/古装剧器乐曲，并设置：

```json
"bgm": {
  "enabled": true,
  "source_type": "download",
  "source_url": "https://example.org/track.mp3",
  "sha256": "<可选的完整 SHA-256>",
  "credit": "曲名 / 作者 / 来源页"
}
```

工作流会把曲目缓存到故事专属目录、验证音频流并在最终混音中循环、淡入淡出；不会把 H3 原生音频或一段合成噪声误当成 BGM。没有现成曲目时，离线五声音阶生成器只能作为明确标注的草稿：

```bash
.venv/bin/python story/workflow2/generate_bgm.py \
  --output input/story_workflow2/<故事>/bgm.wav --duration 210 --seed 1234 --profile ye_gong
```

新项目应在自己的 JSON 中填写 `source_url`；工作流不会自动生成或替换成单调的合成配乐。只有明确把 `source_type` 设为 `generated` 并开启 `allow_synthetic_fallback`，才会使用离线草稿配乐。

## 一致性与内容验收

### 故事级角色隔离

每个新故事都应生成独立的角色卡和身份命名空间，例如 `story:<slug>`；角色身份键形如 `<slug>:<subject_id>`。每个 H3 提示词都包含 `STORY IDENTITY NAMESPACE`、`STORY CAST UNIQUENESS LOCK` 和 `CAST CARD`，明确禁止借用其他故事或公共默认肖像。若配置了参考图，运行前会检查它是否位于本故事资产目录，并与已登记故事的参考图做 SHA-256 去重；发现复用会直接停止，避免再次生成同一角色。

旧配置仍可读取以保证兼容，但位于故事资产目录之外的旧身份参考图（包括公共 `input/story_workflow2/identity_reference.png`）只会被忽略，改用该故事自己的文字角色描述；已完成或已经在运行的成片不会被回溯修改。

每个分镜现在默认是一次独立的 H3 采样（约 15 秒），对白镜头和高风险动作镜头继续使用单角色/单事件上下文；最终只在后端按 `chains` 顺序回拼。提示词会把 `visual_action` 作为“SCENE TRUTH”，并将 `content_contract.must_show/must_not_show` 和 `shot_contract` 作为硬性验收条件，禁止用静态肖像、无关反应或自创事件替代剧本动作。这样旁白只能解释已经明确拍到的动作，不能反过来让 H3 自由改写分镜。

如果确实需要跨镜头连续运动，必须在 `generation.continuity` 中明确选择 `first_frame` 或已验证的 `context_pin`，并在环境中安装对应的 Motion Context；工作流不会再把身份参考图误接成“上一镜头首帧”。

如果没有 H3 声音参考，可在 `voice_references` 中填写 `sample_text` 和 Edge TTS 声音，脚本会自动生成短参考；若只需要纯旁白，仍可在配置中关闭对白并保留一个旁白声音配置。
