#!/usr/bin/env python3
"""Build a standard ComfyUI canvas workflow for the generic H3 runner."""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
H3_TEMPLATE = ROOT / "custom_nodes" / "ComfyUI-H3-Multishot" / "workflows" / "H3_Seamless_Chain_v2.json"


def _template_node(workflow: dict, node_type: str, *, node_id: int | None = None) -> dict:
    for node in workflow["nodes"]:
        if node.get("type") == node_type and (node_id is None or node.get("id") == node_id):
            return copy.deepcopy(node)
    raise RuntimeError(f"Visual workflow template does not contain node type: {node_type}")


def _reset_node(node: dict, node_id: int, position: tuple[float, float], title: str | None = None) -> dict:
    node["id"] = node_id
    node["pos"] = [float(position[0]), float(position[1])]
    node["order"] = node_id
    node["mode"] = 0
    node["flags"] = {}
    if title is not None:
        node["title"] = title
    node.setdefault("properties", {}).pop("h3_widget_values", None)
    for item in node.get("inputs", []):
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = []
    return node


def _ensure_input(node: dict, name: str, data_type: str) -> None:
    if not any(item.get("name") == name for item in node.get("inputs", [])):
        node.setdefault("inputs", []).append({"name": name, "type": data_type, "link": None})


class Canvas:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.links: list[list[Any]] = []
        self.node_map: dict[int, dict] = {}
        self.next_link_id = 1

    def add(self, node: dict) -> dict:
        node_id = int(node["id"])
        if node_id in self.node_map:
            raise ValueError(f"duplicate node id: {node_id}")
        self.nodes.append(node)
        self.node_map[node_id] = node
        return node

    def connect(self, source_id: int, source_slot: int, target_id: int, input_name: str, data_type: str) -> int:
        source = self.node_map[source_id]
        target = self.node_map[target_id]
        target_input = next((item for item in target.get("inputs", []) if item.get("name") == input_name), None)
        if target_input is None:
            target_input = {"name": input_name, "type": data_type, "link": None}
            target.setdefault("inputs", []).append(target_input)
        link_id = self.next_link_id
        self.next_link_id += 1
        target_slot = target["inputs"].index(target_input)
        target_input["link"] = link_id
        outputs = source.get("outputs", [])
        if source_slot >= len(outputs):
            raise ValueError(f"node {source_id} has no output slot {source_slot}")
        outputs[source_slot].setdefault("links", []).append(link_id)
        self.links.append([link_id, source_id, source_slot, target_id, target_slot, data_type])
        return link_id


def _sampler_widgets(node: dict, spec: dict) -> None:
    values = list(node.get("widgets_values", []))
    if len(values) < 13:
        raise RuntimeError("Installed H3 visual template is incompatible: sampler widgets are incomplete")
    values[0] = spec["script"]
    values[1] = 0
    values[2] = int(spec["width"])
    values[3] = int(spec["height"])
    values[4] = int(spec["frames_per_shot"])
    values[5] = int(spec["seed"])
    values[6] = "fixed"
    values[7] = int(spec["steps"])
    values[8] = True
    values[9] = 0
    # The identity batch is supplied through reference_images.  Feeding the
    # same batch into start_image as well appends its first frame a second time
    # and shifts a two-person mapping ("1,1") to [1, 2, 2].
    values[10] = 0
    values[11] = str(spec["sampler"])
    values[12] = str(spec["scheduler"])
    if len(values) > 15:
        values[14] = str(spec.get("chain_gain_control", "off"))
        values[15] = str(spec.get("continuity", "cut"))
        values[16] = int(spec.get("bank_clip_frames", 22))
        values[17] = str(spec.get("color_level", "off"))
        values[18] = float(spec.get("join_anchor_noise", 0.0))
        values[19] = bool(spec.get("join_blend", False))
        values[20] = float(spec.get("handoff_release", 0.30))
        values[21] = float(spec.get("bank_ref_noise", 0.0))
        values[22] = bool(spec.get("end_anchor", False))
        values[23] = "off"
        values[24] = bool(spec.get("audio_lock", False))
        values[25] = int(spec.get("handoff_taper", 0))
        values[26] = str(spec.get("handoff_depth", "block"))
    if len(values) > 19:
        values[19] = bool(spec.get("join_blend", False))
    if len(values) > 38:
        values[27] = bool(spec.get("self_anchor_voice", False))
        values[28] = "match"
        values[29] = True
        values[30] = 1.0
        values[31] = bool(spec.get("low_ram_master", True))
        values[32] = "(none)"
        values[33] = str(spec.get("master_normalize", "luma+contrast"))
        values[34] = str(spec.get("pin_frames", "22"))
        values[35] = float(spec.get("pin_noise", 0.0))
        values[36] = bool(spec.get("pin_renorm", False))
        values[37] = str(spec.get("reference_subjects", ""))
        values[38] = True
    node["widgets_values"] = values


def _note(template: dict, node_id: int, position: tuple[float, float], title: str, text: str) -> dict:
    node = _reset_node(_template_node(template, "Note", node_id=1), node_id, position, title)
    node["widgets_values"] = [text]
    node["size"] = [520, 260]
    return node


def _path_video_node(node_id: int, position: tuple[float, float], title: str, filename_prefix: str) -> dict:
    return {
        "id": node_id,
        "type": "Workflow2VideoFromPaths",
        "pos": [float(position[0]), float(position[1])],
        "size": [390, 300],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "title": title,
        "inputs": [
            {"name": "path_a", "type": "STRING", "link": None},
            {"name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None},
            *({"name": f"path_{letter}", "type": "STRING", "link": None} for letter in "bcdefgh"),
        ],
        "outputs": [
            {"name": "video", "type": "VIDEO", "links": []},
            {"name": "saved_path", "type": "STRING", "links": []},
        ],
        "properties": {"Node name for S&R": "Workflow2VideoFromPaths"},
        "widgets_values": [filename_prefix],
    }


def _story_assembler_node(node_id: int, position: tuple[float, float], title: str, config_path: Path) -> dict:
    return {
        "id": node_id,
        "type": "Workflow2StoryAssembler",
        "pos": [float(position[0]), float(position[1])],
        "size": [430, 340],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "title": title,
        "inputs": [
            {"name": "config_path", "type": "STRING", "widget": {"name": "config_path"}, "link": None},
            {"name": "path_a", "type": "STRING", "link": None},
            *({"name": f"path_{letter}", "type": "STRING", "link": None} for letter in "bcdefgh"),
            {"name": "bgm", "type": "AUDIO", "link": None},
        ],
        "outputs": [
            {"name": "video", "type": "VIDEO", "links": []},
            {"name": "saved_path", "type": "STRING", "links": []},
        ],
        "properties": {"Node name for S&R": "Workflow2StoryAssembler"},
        "widgets_values": [str(config_path)],
    }


def build_visual_workflow(
    *,
    title: str,
    slug: str,
    config_path: Path,
    identity_input: str,
    voice_inputs: list[tuple[str, str]],
    bgm_input: str | None,
    chain_specs: list[dict],
    chain_part_indices: list[list[int]],
    model_names: dict[str, str],
    output_path: Path,
    user_workflow_path: Path,
) -> tuple[Path, Path]:
    """Create one loadable canvas with all H3 chains and a final A/V join."""
    base = json.loads(H3_TEMPLATE.read_text())
    canvas = Canvas()

    canvas.add(
        _note(
            base,
            1,
            (-2350, -760),
            f"工作流2 · {title}",
            "这是配置驱动的 MiniMax H3 可视化长视频工作流。\n\n"
            f"故事级身份命名空间：story:{slug}\n"
            "每个故事拥有独立角色卡；禁止复用其他故事的脸、发型、体型、服饰配色、道具签名或公共身份参考图。缺少故事专属参考图时，只按本故事角色卡中的特性生成。\n\n"
            "1. 先检查左侧模型、身份参考图、声音参考和青色的同步音频脊柱。\n"
            "2. 每句对白会拆成独立的‘说话人锁定’镜头：只接入该角色的身份图、声线与同步音频脊柱，避免角色换脸、换声或代说台词。\n"
            "3. 标记 shot_contract.isolated 的脱缰、落马、接触和多人动作会自动拆成独立 H3 上下文，严格锁定动作归属、演员/动物数量和肢体连续性。\n"
            "4. 默认每个分镜都是一次独立 H3 采样；chains 只负责按剧本顺序回拼，避免把多个叙事事件混在一次生成里。\n"
            "5. 锁定对白和高风险动作镜头会在画布内自动回拼为原始叙事链。\n"
            "6. 右上角低内存合并节点保存最终原生视频，便于检查 H3 的分镜输出。\n"
            "7. 右下角精准时轴节点读取回拼后的 H3 链，按真实帧数生成旁白、带角色名的对白字幕，并将明确来源的独立 BGM 在人声处自动闪避后保存发布成片。\n"
            "8. 修改故事时只更换 JSON 配置，再重新生成此画布。\n\n"
            f"配置：{config_path}",
        )
    )

    model = _reset_node(_template_node(base, "H3ModelLoaderAny", node_id=15), 10, (-2350, -390), "H3 模型")
    model["widgets_values"] = [model_names["unet"], 6.0]
    canvas.add(model)
    clip = _reset_node(_template_node(base, "H3ClipLoaderAny", node_id=16), 11, (-2350, -160), "H3 文本编码器")
    clip["widgets_values"] = [model_names["clip"], "minimax", "(auto)"]
    canvas.add(clip)
    video_vae = _reset_node(_template_node(base, "VAELoader", node_id=18), 12, (-2350, 80), "视频 VAE")
    video_vae["widgets_values"] = [model_names["video_vae"]]
    canvas.add(video_vae)
    audio_vae = _reset_node(_template_node(base, "VAELoader", node_id=19), 13, (-2350, 230), "音频 VAE")
    audio_vae["widgets_values"] = [model_names["audio_vae"]]
    canvas.add(audio_vae)

    identity_node_ids: dict[str, int] = {}
    identity_inputs = list(dict.fromkeys(
        str(spec["identity_input"] if "identity_input" in spec else identity_input)
        for spec in chain_specs
        if (spec["identity_input"] if "identity_input" in spec else identity_input)
    ))
    for index, current_identity_input in enumerate(identity_inputs):
        node_id = 30 + index
        image_node = _reset_node(
            _template_node(base, "LoadImage", node_id=20),
            node_id,
            (-1850, 560 + index * 360),
            f"身份参考图 · {Path(current_identity_input).stem}",
        )
        image_node["widgets_values"] = [current_identity_input, "image"]
        canvas.add(image_node)
        identity_node_ids[current_identity_input] = node_id

    bgm_node_id: int | None = None
    if bgm_input:
        bgm_node_id = 25
        bgm = _reset_node(
            _template_node(base, "LoadAudio", node_id=22),
            bgm_node_id,
            (-2350, 1450),
            "独立 BGM · 最终混音闪避",
        )
        bgm["widgets_values"] = [bgm_input, None, None]
        canvas.add(bgm)

    voice_node_ids: dict[str, int] = {}
    for index, (voice_key, voice_input) in enumerate(voice_inputs):
        node_id = 20 + index
        voice = _reset_node(_template_node(base, "LoadAudio", node_id=22), node_id, (-2350, 900 + index * 240), f"声音参考 · {voice_key}")
        voice["widgets_values"] = [voice_input, None, None]
        canvas.add(voice)
        voice_node_ids[voice_key] = node_id

    guide_node_ids: dict[str, int] = {}
    guide_chain_labels: dict[str, int] = {}
    guide_inputs = []
    for spec in chain_specs:
        if not spec.get("guide_audio"):
            continue
        guide_input = str(spec["guide_audio"])
        if guide_input not in guide_chain_labels:
            guide_inputs.append(guide_input)
            guide_chain_labels[guide_input] = int(spec.get("chain_index", len(guide_inputs)))
    for index, guide_input in enumerate(guide_inputs):
        node_id = 50 + index
        guide = _reset_node(
            _template_node(base, "LoadAudio", node_id=22),
            node_id,
            (-1850, 1700 + index * 250),
            f"同步音频脊柱 · 分镜链 {guide_chain_labels[guide_input]}",
        )
        guide["widgets_values"] = [guide_input, None, None]
        canvas.add(guide)
        guide_node_ids[guide_input] = node_id

    component_path_ids: list[int] = []
    for chain_offset, spec in enumerate(chain_specs):
        sampler_id = 100 + chain_offset
        lock_label = " · 说话人锁定" if spec.get("speaker_locked") else ""
        sampler = _reset_node(
            _template_node(base, "H3MultishotMemorySampler", node_id=30),
            sampler_id,
            (-1300, -500 + chain_offset * 980),
            f"H3 组件 {chain_offset + 1} · 原链 {spec['chain_index']} · 镜头 {', '.join(map(str, spec['scene_ids']))}{lock_label}",
        )
        _ensure_input(sampler, "voice_ref_2", "AUDIO")
        _ensure_input(sampler, "voice_ref_3", "AUDIO")
        _ensure_input(sampler, "guide_audio", "AUDIO")
        _sampler_widgets(sampler, spec)
        sampler["size"] = [760, 760]
        canvas.add(sampler)
        canvas.connect(10, 0, sampler_id, "model", "MODEL")
        canvas.connect(11, 0, sampler_id, "clip", "CLIP")
        canvas.connect(12, 0, sampler_id, "video_vae", "VAE")
        canvas.connect(13, 0, sampler_id, "audio_vae", "VAE")
        raw_identity_input = spec["identity_input"] if "identity_input" in spec else identity_input
        current_identity_input = str(raw_identity_input) if raw_identity_input else ""
        if current_identity_input:
            canvas.connect(identity_node_ids[current_identity_input], 0, sampler_id, "reference_images", "IMAGE")
        for voice_index, voice_key in enumerate(spec.get("voice_keys", [])[:3]):
            if voice_key in voice_node_ids:
                input_name = "voice_ref" if voice_index == 0 else f"voice_ref_{voice_index + 1}"
                canvas.connect(voice_node_ids[voice_key], 0, sampler_id, input_name, "AUDIO")
        guide_input = spec.get("guide_audio")
        if guide_input:
            canvas.connect(guide_node_ids[str(guide_input)], 0, sampler_id, "guide_audio", "AUDIO")

        preview_video_id = 200 + chain_offset * 3
        preview_save_id = preview_video_id + 1
        preview_audio_id = preview_video_id + 2
        preview = _path_video_node(
            preview_video_id,
            (-360, -400 + chain_offset * 980),
            f"加载/预览 H3 组件 {chain_offset + 1}",
            f"video/workflow2/{slug}/component_{chain_offset + 1:02d}_visual",
        )
        canvas.add(preview)
        component_path_ids.append(preview_video_id)
        save = _reset_node(_template_node(base, "SaveVideo", node_id=27), preview_save_id, (40, -430 + chain_offset * 980), f"保存 H3 组件 {chain_offset + 1}")
        save["widgets_values"] = [f"video/workflow2/{slug}/component_{chain_offset + 1:02d}", "auto", "auto"]
        canvas.add(save)
        save_audio = _reset_node(_template_node(base, "SaveAudio", node_id=28), preview_audio_id, (40, -170 + chain_offset * 980), f"保存 H3 组件 {chain_offset + 1} 音频")
        save_audio["widgets_values"] = [f"audio/workflow2/{slug}/component_{chain_offset + 1:02d}"]
        canvas.add(save_audio)
        canvas.connect(sampler_id, 6, preview_video_id, "path_a", "STRING")
        canvas.connect(preview_video_id, 0, preview_save_id, "video", "VIDEO")
        canvas.connect(sampler_id, 1, preview_audio_id, "audio", "AUDIO")

    if len(chain_part_indices) > 8:
        raise ValueError("one visual workflow supports at most eight narrative chains")
    chain_merge_ids: list[int] = []
    for original_chain_index, part_indices in enumerate(chain_part_indices, start=1):
        if not part_indices:
            raise ValueError(f"narrative chain {original_chain_index} has no H3 components")
        if len(part_indices) > 8:
            raise ValueError(f"narrative chain {original_chain_index} has more than eight H3 components")
        merge_id = 500 + original_chain_index
        merge = _path_video_node(
            merge_id,
            (250, -220 + (original_chain_index - 1) * 370),
            f"回拼原始分镜链 {original_chain_index}",
            f"video/workflow2/{slug}/chain_{original_chain_index:02d}_speaker_locked",
        )
        canvas.add(merge)
        chain_merge_ids.append(merge_id)
        for part_offset, component_index in enumerate(part_indices):
            input_name = "path_a" if part_offset == 0 else f"path_{chr(ord('a') + part_offset)}"
            canvas.connect(component_path_ids[component_index], 1, merge_id, input_name, "STRING")

    final_video = _path_video_node(400, (720, 250), "合并所有 H3 分镜链（低内存）", f"video/workflow2/{slug}/{slug}_joined")
    canvas.add(final_video)
    for index, merge_id in enumerate(chain_merge_ids):
        input_name = "path_a" if index == 0 else f"path_{chr(ord('a') + index)}"
        canvas.connect(merge_id, 1, 400, input_name, "STRING")
    final_save = _reset_node(_template_node(base, "SaveVideo", node_id=27), 401, (1190, 100), "保存并预览最终原生 AI 视频")
    final_save["widgets_values"] = [f"video/workflow2/{slug}/{slug}_native_master", "auto", "auto"]
    canvas.add(final_save)
    components = _reset_node(_template_node(base, "GetVideoComponents", node_id=62), 402, (1190, 410), "拆分最终视频/音频")
    canvas.add(components)
    final_audio = _reset_node(_template_node(base, "SaveAudio", node_id=28), 403, (1590, 440), "保存最终原生音频")
    final_audio["widgets_values"] = [f"audio/workflow2/{slug}/{slug}_native_master"]
    canvas.add(final_audio)
    canvas.connect(400, 0, 401, "video", "VIDEO")
    canvas.connect(400, 0, 402, "video", "VIDEO")
    canvas.connect(402, 1, 403, "audio", "AUDIO")

    assembler = _story_assembler_node(404, (720, 840), "精准时轴：旁白、对白、BGM 与环境声", config_path)
    canvas.add(assembler)
    for index, merge_id in enumerate(chain_merge_ids):
        input_name = "path_a" if index == 0 else f"path_{chr(ord('a') + index)}"
        canvas.connect(merge_id, 1, 404, input_name, "STRING")
    if bgm_node_id is not None:
        canvas.connect(bgm_node_id, 0, 404, "bgm", "AUDIO")
    delivery_save = _reset_node(_template_node(base, "SaveVideo", node_id=27), 405, (1210, 820), "保存并预览最终发布成片")
    delivery_save["widgets_values"] = [f"video/workflow2/{slug}/{slug}_delivery_preview", "auto", "auto"]
    canvas.add(delivery_save)
    delivery_components = _reset_node(_template_node(base, "GetVideoComponents", node_id=62), 406, (1210, 1120), "拆分最终发布成片")
    canvas.add(delivery_components)
    delivery_audio = _reset_node(_template_node(base, "SaveAudio", node_id=28), 407, (1600, 1130), "保存最终混音")
    delivery_audio["widgets_values"] = [f"audio/workflow2/{slug}/{slug}_delivery"]
    canvas.add(delivery_audio)
    canvas.connect(404, 0, 405, "video", "VIDEO")
    canvas.connect(404, 0, 406, "video", "VIDEO")
    canvas.connect(406, 1, 407, "audio", "AUDIO")

    groups = [
        {"id": 1, "title": "1 · 通用配置、模型与音画同步", "bounding": [-2390, -810, 570, 3100], "color": "#3f789e", "flags": {}},
        {"id": 2, "title": "2 · H3 组件（对白自动说话人锁定）", "bounding": [-1350, -610, 1430, max(1060, len(chain_specs) * 980)], "color": "#8154a1", "flags": {}},
        {"id": 3, "title": "3 · 回拼与最终输出", "bounding": [190, -350, 1850, max(1600, len(chain_specs) * 520)], "color": "#3f8e5f", "flags": {}},
    ]
    workflow = {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": max(canvas.node_map),
        "last_link_id": canvas.next_link_id - 1,
        "nodes": canvas.nodes,
        "links": canvas.links,
        "floatingLinks": [],
        "groups": groups,
        "config": {},
        "extra": {"ds": {"scale": 0.55, "offset": [2550, 950]}, "frontendVersion": base.get("extra", {}).get("frontendVersion", "1.48.7")},
        "version": 0.4,
    }
    validate_visual_workflow(workflow)
    payload = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload)
    user_workflow_path.parent.mkdir(parents=True, exist_ok=True)
    user_workflow_path.write_text(payload)
    return output_path, user_workflow_path


def validate_visual_workflow(workflow: dict) -> None:
    if not workflow.get("nodes") or not workflow.get("links"):
        raise ValueError("visual workflow must contain nodes and links")
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    if len(nodes) != len(workflow["nodes"]):
        raise ValueError("visual workflow contains duplicate node ids")
    links = {int(link[0]): link for link in workflow["links"]}
    if len(links) != len(workflow["links"]):
        raise ValueError("visual workflow contains duplicate link ids")
    for link_id, source_id, source_slot, target_id, target_slot, _ in workflow["links"]:
        if source_id not in nodes or target_id not in nodes:
            raise ValueError(f"link {link_id} references a missing node")
        if source_slot >= len(nodes[source_id].get("outputs", [])):
            raise ValueError(f"link {link_id} references an invalid source slot")
        if target_slot >= len(nodes[target_id].get("inputs", [])):
            raise ValueError(f"link {link_id} references an invalid target slot")
    referenced_inputs = {
        int(item["link"])
        for node in workflow["nodes"]
        for item in node.get("inputs", [])
        if item.get("link") is not None
    }
    if referenced_inputs != set(links):
        raise ValueError("visual workflow input links do not match its link table")
