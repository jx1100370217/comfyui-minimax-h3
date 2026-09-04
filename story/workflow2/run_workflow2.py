#!/usr/bin/env python3
"""Generic, configuration-driven MiniMax H3 Ref2VA story/video workflow.

The runner contains no story-specific characters, scenes, title, or dialogue.
Those belong in a JSON story configuration passed with ``--config``.  The
same runner can therefore be used for idiom stories, historical shorts,
product narratives, or any other shot-based video with optional dialogue.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from urllib.parse import unquote, urlparse
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import edge_tts
import requests


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent
STREAM_DIR = ROOT / "output" / "video" / "H3CHAIN_STREAM"

UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
WORKFLOW_SIGNATURE_VERSION = 4
SHARED_IDENTITY_REFERENCE = (ROOT / "input" / "story_workflow2" / "identity_reference.png").resolve()
DEFAULT_IDENTITY_REGISTRY = WORK_DIR / "identity_registry.json"


@dataclass
class RuntimePaths:
    config_path: Path
    slug: str
    run_dir: Path
    chains_dir: Path
    narration_dir: Path
    dialogue_fallback_dir: Path
    audio_spines_dir: Path
    final_dir: Path
    workflows_dir: Path
    assets_dir: Path
    identity_source: Path | None
    identity_name: str
    voice_ref_dir: Path
    audio_qc_path: Path
    comfy_identity_ref: str | None
    comfy_voice_refs: dict[str, str] = field(default_factory=dict)


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, **kwargs)


def media_duration(path: Path) -> float:
    result = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def video_frame_info(path: Path) -> tuple[int, float]:
    """Return the exact decoded video-frame count and frame rate for a master."""
    result = run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,nb_read_frames,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream in {path}")
    stream = streams[0]
    frame_text = stream.get("nb_frames") or stream.get("nb_read_frames")
    if not frame_text or str(frame_text) == "N/A":
        raise RuntimeError(f"Could not read video frame count from {path}")
    numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
    fps = float(numerator) / float(denominator)
    if fps <= 0:
        raise RuntimeError(f"Invalid video frame rate in {path}: {stream['r_frame_rate']}")
    return int(frame_text), fps


def build_scene_timeline(config: dict, chain_paths: list[Path]) -> list[dict[str, float | int]]:
    """Map every scene to exact video-frame boundaries in the H3 chain masters.

    H3's multishot master normally keeps one shared frame at every internal
    chain cut; some streamed masters also drop the cut-adjacent terminal
    frame during muxing.
    Audio used to be placed with ``master_duration / scene_count``; that makes
    narration and dialogue drift whenever a chain contains more than one shot.
    This function derives the timeline from the actual video frames instead.
    """
    if len(chain_paths) != len(config["chains"]):
        raise ValueError("chain path count must match config.chains")
    target_fps = float(video_value(config, "fps", 24))
    frames_per_shot = int(video_value(config, "frames_per_shot", 362))
    global_frame = 0
    timeline: list[dict[str, float | int]] = []

    for chain_index, (scene_ids, chain_path) in enumerate(zip(config["chains"], chain_paths), start=1):
        actual_frames, actual_fps = video_frame_info(chain_path)
        if not math.isclose(actual_fps, target_fps, rel_tol=0.0, abs_tol=0.02):
            raise RuntimeError(
                f"Chain {chain_index} is {actual_fps:g} fps, expected {target_fps:g} fps: {chain_path}"
            )
        shot_count = len(scene_ids)
        expected_frames = shot_count * frames_per_shot
        overlap_total = expected_frames - actual_frames
        # A single streamed component may lose one terminal frame during muxing;
        # a multishot chain may additionally lose one shared and one terminal
        # frame at each internal cut.
        max_cut_frames = max(1, 2 * max(0, shot_count - 1))
        if overlap_total < 0 or overlap_total > max_cut_frames:
            raise RuntimeError(
                f"Chain {chain_index} has {actual_frames} frames; expected {expected_frames} with at most two cut-adjacent frames removed per cut"
            )
        boundary_count = max(0, shot_count - 1)
        boundary_overlaps = [
            ((boundary + 1) * overlap_total) // boundary_count - (boundary * overlap_total) // boundary_count
            for boundary in range(boundary_count)
        ]
        local_frame = 0
        for within_chain, scene_id in enumerate(scene_ids):
            next_local_frame = (
                actual_frames
                if within_chain == shot_count - 1
                else local_frame + frames_per_shot - boundary_overlaps[within_chain]
            )
            start_frame = global_frame + local_frame
            end_frame = global_frame + next_local_frame
            timeline.append(
                {
                    "scene_id": int(scene_id),
                    "start": start_frame / target_fps,
                    "end": end_frame / target_fps,
                    "duration": (end_frame - start_frame) / target_fps,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "chain_index": chain_index,
                }
            )
            local_frame = next_local_frame
        global_frame += actual_frames
    return timeline


def normalize_chain_paths(config: dict, paths: RuntimePaths, chain_paths: list[Path]) -> list[Path]:
    """Trim tiny mux-length overshoots before timeline and final concatenation.

    H3 streamed masters can contain one extra terminal frame per component
    (for example 722 instead of 720 frames for a two-shot chain).  Keeping
    those frames shifts every later narration cue, so normalize only this
    bounded transport artifact and leave intentional short-chain overlap
    handling to ``build_scene_timeline``.
    """
    fps = float(video_value(config, "fps", 24))
    frames_per_shot = int(video_value(config, "frames_per_shot", 362))
    output_dir = paths.final_dir / "normalized_chains"
    normalized: list[Path] = []
    for chain_index, (scene_ids, source) in enumerate(zip(config["chains"], chain_paths), start=1):
        actual_frames, actual_fps = video_frame_info(source)
        expected_frames = len(scene_ids) * frames_per_shot
        excess = actual_frames - expected_frames
        max_overshoot = max(1, 2 * max(0, len(scene_ids) - 1))
        if excess <= 0 or excess > max_overshoot:
            normalized.append(source)
            continue
        destination = output_dir / f"chain_{chain_index:02d}.mp4"
        needs_render = not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns
        if not needs_render:
            try:
                needs_render = video_frame_info(destination)[0] != expected_frames
            except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError):
                needs_render = True
        if needs_render:
            output_dir.mkdir(parents=True, exist_ok=True)
            duration = expected_frames / fps
            filter_complex = (
                f"[0:v]trim=start_frame=0:end_frame={expected_frames},setpts=PTS-STARTPTS[v];"
                f"[0:a]atrim=start=0:end={duration:.6f},asetpts=PTS-STARTPTS[a]"
            )
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
                "-r", f"{fps:.6f}", "-frames:v", str(expected_frames),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart", str(destination),
            ])
        print(f"NORMALIZE CHAIN {chain_index:02d}: {actual_frames}->{expected_frames} frames", flush=True)
        normalized.append(destination)
    return normalized


def valid_video(path: Path, minimum: float) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        return media_duration(path) >= minimum
    except (subprocess.CalledProcessError, ValueError):
        return False


def valid_audio_reference(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100_000:
        return False
    try:
        duration = media_duration(path)
    except (subprocess.CalledProcessError, ValueError):
        return False
    return 2.0 <= duration <= 8.0


def slugify(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value.strip(), flags=re.UNICODE).strip("-")
    return value or "story"


def resolve_project_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (ROOT / path, config_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def video_value(config: dict, key: str, default: Any) -> Any:
    video = config.get("video", {})
    return video.get(key, config.get(key, default))


def style_value(config: dict, key: str, default: str) -> str:
    style = config.get("style", {})
    if isinstance(style, str):
        return style if key == "visual" else default
    return str(style.get(key, default))


def prompting_value(config: dict, key: str, default: str) -> str:
    """Read optional story-direction text shared by every H3 shot.

    Keeping screenplay and dialogue direction in the generic runner means a
    new story only supplies its narrative material; the same film-language,
    blocking, and speaker discipline is automatically applied to every shot.
    """
    prompting = config.get("prompting", {})
    if isinstance(prompting, str):
        return prompting if key == "screenplay" else default
    if not isinstance(prompting, dict):
        return default
    return str(prompting.get(key, default))


def render_signature(config: dict, scene_ids: list[int]) -> str:
    """Fingerprint the prompt and sampler inputs so stale H3 clips cannot be reused."""
    scenes_by_id = {int(scene["id"]): scene for scene in config.get("scenes", [])}
    payload = {
        "version": WORKFLOW_SIGNATURE_VERSION,
        "model": [UNET, CLIP, VIDEO_VAE, AUDIO_VAE],
        "seed": config.get("seed"),
        "video": config.get("video", {}),
        "width": config.get("width"),
        "height": config.get("height"),
        "fps": config.get("fps"),
        "frames_per_shot": config.get("frames_per_shot"),
        "steps": config.get("steps"),
        "sampler": config.get("sampler"),
        "scheduler": config.get("scheduler"),
        "style": config.get("style", {}),
        "prompting": config.get("prompting", {}),
        "generation": config.get("generation", {}),
        "identity_policy": config.get("identity_policy", {}),
        "identity_namespace": identity_namespace(config),
        "identity_references": config.get("assets", {}).get("identity_references", {}),
        "subjects": normalize_subjects(config),
        "voice_references": config.get("voice_references", {}),
        "scenes": [scenes_by_id[int(scene_id)] for scene_id in scene_ids],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def signature_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".workflow2.signature")


def signature_matches(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    marker = signature_path(path)
    try:
        return marker.read_text().strip() == signature
    except OSError:
        return False


def write_signature(path: Path, signature: str) -> None:
    signature_path(path).write_text(signature + "\n")


def config_slug(config: dict) -> str:
    return slugify(str(config.get("slug") or config.get("title") or "story"))


def identity_policy(config: dict) -> dict:
    policy = config.get("identity_policy", {})
    return policy if isinstance(policy, dict) else {}


def identity_namespace(config: dict) -> str:
    """Return the stable visual-cast namespace for one story project."""
    policy = identity_policy(config)
    return str(policy.get("namespace") or f"story:{config_slug(config)}")


def resolve_asset_path(value: str | Path, config: dict, config_path: Path) -> Path:
    assets = config.get("assets", {}) if isinstance(config.get("assets", {}), dict) else {}
    resolved = resolve_project_path(value, config_path)
    if not resolved.exists():
        assets_dir = resolve_project_path(assets.get("directory", "input/story_workflow2"), config_path)
        candidate = assets_dir / str(value)
        if candidate.exists():
            resolved = candidate
    return resolved


def configured_identity_sources(config: dict, config_path: Path) -> list[tuple[str, Path]]:
    assets = config.get("assets", {}) if isinstance(config.get("assets", {}), dict) else {}
    sources: list[tuple[str, Path]] = []
    combined = assets.get("identity_reference")
    if combined:
        sources.append(("__combined__", resolve_asset_path(combined, config, config_path)))
    references = assets.get("identity_references", {})
    if isinstance(references, dict):
        for subject_id, value in references.items():
            if value:
                sources.append((str(subject_id), resolve_asset_path(value, config, config_path)))
    return sources


def is_legacy_shared_identity_value(config: dict, value: str | Path) -> bool:
    if "identity_policy" in config:
        return False
    candidate = Path(value)
    candidates = [candidate] if candidate.is_absolute() else [ROOT / candidate, ROOT / "input" / "story_workflow2" / candidate]
    return any(path.resolve() == SHARED_IDENTITY_REFERENCE for path in candidates)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _known_identity_sources(config_path: Path) -> list[tuple[str, str, Path]]:
    """Read existing story configs to catch copied portraits, including legacy configs."""
    records: list[tuple[str, str, Path]] = []
    story_dir = WORK_DIR / "stories"
    if not story_dir.exists():
        return records
    for candidate in sorted(story_dir.glob("*.json")):
        if candidate.resolve() == config_path.resolve():
            continue
        try:
            other = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        namespace = identity_namespace(other)
        for subject_id, source in configured_identity_sources(other, candidate):
            if source.exists() and source.is_file():
                records.append((namespace, subject_id, source))
    return records


def validate_story_identity_policy(config: dict, config_path: Path | None = None) -> None:
    """Require story-local, non-reused cast assets for newly authored projects."""
    policy = config.get("identity_policy")
    if policy is None:
        return  # Legacy projects remain readable; their generic fallback is ignored at staging time.
    if not isinstance(policy, dict):
        raise ValueError("identity_policy must be an object")
    if str(policy.get("scope", "story")) != "story":
        raise ValueError("identity_policy.scope must be 'story'")
    if not bool(policy.get("require_unique_story_cast", True)):
        raise ValueError("identity_policy.require_unique_story_cast must remain true")
    expected_namespace = f"story:{config_slug(config)}"
    if identity_namespace(config) != expected_namespace:
        raise ValueError(f"identity_policy.namespace must be {expected_namespace!r}")
    subjects = normalize_subjects(config)
    if not subjects:
        raise ValueError("identity_policy requires at least one subject")
    keys = [str(subject.get("identity_key", "")).strip() for subject in subjects]
    if any(not key for key in keys):
        raise ValueError("every subject in a unique story cast requires identity_key")
    if len(keys) != len(set(keys)):
        raise ValueError("subject identity_key values must be unique within a story")
    if any(not key.startswith(f"{config_slug(config)}:") for key in keys):
        raise ValueError("subject identity_key must be namespaced by the story slug")
    if config_path is None:
        return
    assets = config.get("assets", {}) if isinstance(config.get("assets", {}), dict) else {}
    assets_dir = resolve_project_path(assets.get("directory", "input/story_workflow2"), config_path).resolve()
    configured_references = assets.get("identity_references", {})
    if not assets.get("identity_reference") and isinstance(configured_references, dict):
        dangling = [
            subject["id"] for subject in subjects
            if "reference_picture" in subject and subject["id"] not in configured_references
        ]
        if dangling:
            raise ValueError(
                "subjects with reference_picture require a matching story-local assets.identity_references entry: "
                + ", ".join(dangling)
            )
    local_records: list[tuple[str, Path]] = []
    for subject_id, source in configured_identity_sources(config, config_path):
        if not source.exists() or not source.is_file():
            raise ValueError(f"identity reference for {subject_id} does not exist: {source}")
        if not _is_relative_to(source, assets_dir):
            raise ValueError(
                f"identity reference for {subject_id} must live inside this story's asset directory {assets_dir}: {source}"
            )
        local_records.append((subject_id, source.resolve()))
    known = _known_identity_sources(config_path)
    for subject_id, source in local_records:
        digest = file_sha256(source)
        for other_namespace, other_subject, other_source in known:
            if file_sha256(other_source) == digest and other_namespace != identity_namespace(config):
                raise ValueError(
                    f"identity reference {source} reuses the {other_namespace}/{other_subject} portrait; "
                    "each story must have its own cast"
                )
    registry_value = policy.get("registry", str(DEFAULT_IDENTITY_REGISTRY.relative_to(ROOT)))
    registry_path = resolve_project_path(registry_value, config_path)
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"identity registry is not valid JSON: {registry_path}") from exc
        for item in registry.get("references", []):
            if not isinstance(item, dict) or item.get("namespace") == identity_namespace(config):
                continue
            if any(str(item.get("sha256")) == file_sha256(source) for _, source in local_records):
                raise ValueError(
                    f"identity reference {source} is already registered to another story: {item.get('namespace')}"
                )


def write_story_identity_manifest(config: dict, paths: RuntimePaths) -> Path:
    """Persist a human-readable, story-scoped cast card used by future renders/QC."""
    subjects = normalize_subjects(config)
    assets = config.get("assets", {}) if isinstance(config.get("assets", {}), dict) else {}
    references = assets.get("identity_references", {})
    manifest = {
        "version": 1,
        "namespace": identity_namespace(config),
        "title": str(config.get("title", "")),
        "policy": identity_policy(config) or {"legacy": True, "shared_reference_ignored": True},
        "subjects": [
            {
                "id": subject["id"],
                "identity_key": subject.get("identity_key") or f"{config_slug(config)}:{subject['id']}",
                "description": subject["description"],
                "reference_configured": subject["id"] in references if isinstance(references, dict) else False,
            }
            for subject in subjects
        ],
        "rules": [
            "Never reuse a face, body, hair, costume palette, prop signature, or voice identity from another story.",
            "Create missing characters from this story's written traits; use references only from this story's asset directory.",
            "A reference image is an identity anchor, not permission to import its background people or props.",
        ],
    }
    destination = paths.run_dir / "story_identity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return destination


def register_story_identity(config: dict, paths: RuntimePaths) -> None:
    """Record the cast namespace and reference hashes for future collision checks."""
    policy = identity_policy(config)
    if not policy or not bool(policy.get("require_unique_story_cast", True)):
        return
    registry_value = policy.get("registry", str(DEFAULT_IDENTITY_REGISTRY.relative_to(ROOT)))
    registry_path = resolve_project_path(registry_value, paths.config_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"identity registry is not valid JSON: {registry_path}") from exc
    registry.setdefault("version", 1)
    registry.setdefault("stories", [])
    registry.setdefault("references", [])
    namespace = identity_namespace(config)
    registry["stories"] = [item for item in registry["stories"] if item.get("namespace") != namespace]
    registry["stories"].append({
        "namespace": namespace,
        "title": str(config.get("title", "")),
        "subject_keys": [subject.get("identity_key") for subject in normalize_subjects(config)],
    })
    registry["references"] = [item for item in registry["references"] if item.get("namespace") != namespace]
    for subject_id, source in configured_identity_sources(config, paths.config_path):
        if source.exists() and source.is_file():
            registry["references"].append({
                "namespace": namespace,
                "subject_id": subject_id,
                "path": str(source.resolve()),
                "sha256": file_sha256(source),
            })
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")


def build_paths(config: dict, config_path: Path) -> RuntimePaths:
    slug = config_slug(config)
    run_dir = resolve_project_path(config.get("run_dir", f"story/workflow2/runs/{slug}"), config_path)
    assets = config.get("assets", {})
    assets_dir = resolve_project_path(assets.get("directory", "input/story_workflow2"), config_path)
    identity_value = assets.get("identity_reference")
    identity_source = resolve_project_path(identity_value, config_path) if identity_value else None
    if identity_value and not identity_source.exists() and (assets_dir / str(identity_value)).exists():
        identity_source = assets_dir / str(identity_value)
    # The former generator pointed every story at this one marketplace portrait.
    # Keep legacy JSON readable, but never pass that shared fallback into a new render.
    if identity_source and "identity_policy" not in config and not _is_relative_to(identity_source, assets_dir):
        identity_source = None
    identity_name = Path(identity_value).name if identity_source and identity_value else ""
    workflows_dir = resolve_project_path(
        config.get("workflows_dir", f"story/workflows/workflow2/{slug}"), config_path
    )
    identity_suffix = Path(identity_name).suffix.lower() or ".png"
    identity_stem = slugify(Path(identity_name).stem) if identity_name else ""
    return RuntimePaths(
        config_path=config_path,
        slug=slug,
        run_dir=run_dir,
        chains_dir=run_dir / "chains",
        narration_dir=run_dir / "narration",
        dialogue_fallback_dir=run_dir / "dialogue_fallback",
        audio_spines_dir=run_dir / "audio_spines",
        final_dir=run_dir / "final",
        workflows_dir=workflows_dir,
        assets_dir=assets_dir,
        identity_source=identity_source,
        identity_name=identity_name,
        voice_ref_dir=run_dir / "voice_refs",
        audio_qc_path=run_dir / "audio_qc.json",
        # Built-in ComfyUI LoadImage/LoadAudio nodes only enumerate files at
        # the top level of input/.  Story-prefixed names avoid collisions while
        # keeping the assets selectable and valid on the visual canvas.
        comfy_identity_ref=(f"workflow2_{slug}_{identity_stem}{identity_suffix}" if identity_name else None),
    )


def resolve_bgm_source(config: dict, paths: RuntimePaths, override: Path | None = None) -> Path | None:
    """Resolve the optional, separately mixed background-music source.

    A BGM file is deliberately an external source of truth.  The old project
    generator created a synthetic WAV automatically, which made it too easy
    to mistake a placeholder bed for the requested period-drama score.  New
    projects may use ``source_type=download`` and provide a URL; generated
    beds are accepted only when they explicitly opt in with
    ``allow_synthetic_fallback``.
    """
    setting = config.get("bgm", {})
    if override is None and (setting is False or not isinstance(setting, dict) or not setting.get("enabled", False)):
        return None
    source_type = str(setting.get("source_type", "file")).lower()
    if source_type in {"none", "off", "disabled"}:
        return None
    if source_type == "generated" and not bool(setting.get("allow_synthetic_fallback", False)):
        raise RuntimeError(
            "Synthetic BGM is disabled by default; provide bgm.source_type='file' or 'download', "
            "or explicitly set bgm.allow_synthetic_fallback=true for a temporary placeholder"
        )
    if override is not None:
        source = Path(override).expanduser().resolve()
    elif source_type == "download":
        source = ensure_downloaded_bgm(setting, paths)
    else:
        configured = setting.get("file")
        if not configured:
            raise ValueError("bgm.enabled is true but bgm.file is missing (or use bgm.source_type='download')")
        source = resolve_project_path(str(configured), paths.config_path)
        if not source.exists() and (paths.assets_dir / str(configured)).exists():
            source = (paths.assets_dir / str(configured)).resolve()
    if not source.is_file():
        raise RuntimeError(f"Configured BGM does not exist: {source}")
    return source


def ensure_downloaded_bgm(setting: dict, paths: RuntimePaths) -> Path:
    """Download and cache one declared score without silently replacing it.

    The cache is keyed by the story slug and URL; an optional SHA-256 makes a
    published workflow reproducible even when a host later replaces a file.
    Downloads are streamed to a ``.part`` file and promoted atomically only
    after the checksum and an ffprobe audio-stream check pass.
    """
    url = str(setting.get("source_url") or setting.get("url") or "").strip()
    if not url:
        raise ValueError("bgm.source_type='download' requires bgm.source_url")
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path)).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}:
        suffix = ".audio"
    cache_dir = paths.assets_dir / "bgm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    destination = cache_dir / f"download_{cache_key}{suffix}"
    expected_hash = str(setting.get("sha256") or "").strip().lower()
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            digest = hashlib.sha256()
            with requests.get(url, stream=True, timeout=(15, 120), headers={"User-Agent": "Workflow2-BGM/1.0"}) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
            actual_hash = digest.hexdigest()
            if expected_hash and actual_hash != expected_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded BGM sha256 mismatch: expected {expected_hash}, got {actual_hash}")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if expected_hash:
        actual_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Cached BGM sha256 mismatch: expected {expected_hash}, got {actual_hash}")
    try:
        probe = run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(destination)],
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(f"Downloaded BGM is not a readable audio file: {destination}") from exc
    if not probe.stdout.strip():
        raise RuntimeError(f"Downloaded BGM has no audio stream: {destination}")
    return destination


def stage_bgm(config: dict, paths: RuntimePaths, force: bool = False) -> str | None:
    """Copy the configured BGM into ComfyUI's top-level input directory."""
    source = resolve_bgm_source(config, paths)
    if source is None:
        return None
    destination = ROOT / "input" / f"workflow2_{paths.slug}_bgm{source.suffix.lower() or '.wav'}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.exists() or destination.stat().st_size != source.stat().st_size or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    return destination.name


def audio_spine_enabled(config: dict, dialogue: dict | None = None) -> bool:
    """Whether this dialogue is generated against the final clean audio.

    Existing story configurations keep their legacy post-generation lip-window
    behavior.  A story opts in once with ``audio_spine.enabled: true``; an
    individual dialogue can then explicitly opt out with
    ``dialogue.audio_spine: false`` when it is deliberately off-screen.
    """
    setting = config.get("audio_spine", {})
    if isinstance(setting, bool):
        enabled = setting
    elif isinstance(setting, dict):
        enabled = bool(setting.get("enabled", False))
    else:
        enabled = False
    if dialogue is not None and "audio_spine" in dialogue:
        return bool(dialogue["audio_spine"])
    return enabled


def audio_spine_input_name(paths: RuntimePaths, key: str) -> str:
    return f"workflow2_{paths.slug}_{slugify(key)}_audio_spine.wav"


def stage_audio_spine(
    config: dict,
    paths: RuntimePaths,
    *,
    key: str,
    scene_ids: list[int],
    force: bool,
) -> str | None:
    """Render the exact clean dialogue timing that drives an H3 chain.

    The resulting track is intentionally silent outside authored dialogue.
    It is an H3 generation control, not the release mix: assembly still uses
    the same clean dialogue sources plus separately ducked native ambience.
    """
    scenes_by_id = {int(scene["id"]): scene for scene in config["scenes"]}
    spine_scenes = [scenes_by_id[int(scene_id)] for scene_id in scene_ids]
    if not any(scene.get("dialogue") and audio_spine_enabled(config, scene["dialogue"]) for scene in spine_scenes):
        return None

    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    total = len(spine_scenes) * shot_seconds
    events: list[dict[str, Any]] = []
    for local_index, scene in enumerate(spine_scenes):
        dialogue = scene.get("dialogue")
        if not dialogue or not audio_spine_enabled(config, dialogue):
            continue
        source = paths.dialogue_fallback_dir / f"scene_{int(scene['id']):02d}.mp3"
        duration = media_duration(source)
        start = local_index * shot_seconds + float(dialogue["start"])
        end = start + duration
        if end > (local_index + 1) * shot_seconds - 0.12:
            raise RuntimeError(
                f"Scene {scene['id']} dialogue does not fit its H3 audio spine; shorten the line or use a faster dialogue voice"
            )
        events.append({"source": source, "start": start, "end": end, "speed": 1.0, "lufs": -16})

    paths.audio_spines_dir.mkdir(parents=True, exist_ok=True)
    destination = paths.audio_spines_dir / f"{slugify(key)}.wav"
    if force or not destination.exists() or abs(media_duration(destination) - total) > 0.05:
        render_speech_track(events, total, destination)
    destination_input = ROOT / "input" / audio_spine_input_name(paths, key)
    destination_input.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination, destination_input)
    return destination_input.name


def normalize_subjects(config: dict) -> list[dict]:
    subjects = config.get("subjects", [])
    if not subjects:
        return [{"id": "subject_1", "description": "the principal subject described by the shot"}]
    result = []
    for index, subject in enumerate(subjects, start=1):
        if isinstance(subject, str):
            result.append({"id": f"subject_{index}", "description": subject})
        else:
            normalized = {
                "id": str(subject.get("id", f"subject_{index}")),
                "description": str(subject["description"]),
            }
            if "reference_picture" in subject:
                normalized["reference_picture"] = int(subject["reference_picture"])
            for key in ("identity_key", "identity_traits", "role"):
                if key in subject and subject[key] is not None:
                    normalized[key] = str(subject[key])
            result.append(normalized)
    return result


def voice_profiles(config: dict) -> dict[str, dict]:
    profiles = config.get("voice_references", {})
    if isinstance(profiles, list):
        profiles = {str(item["id"]): item for item in profiles}
    return {str(key): dict(value) for key, value in profiles.items()}


def dialogue_speaker_key(config: dict, dialogue: dict) -> str:
    explicit = dialogue.get("speaker_id") or dialogue.get("voice_reference")
    if explicit:
        return str(explicit)
    speaker_map = config.get("speaker_map", {})
    speaker = str(dialogue.get("speaker", ""))
    return str(speaker_map.get(speaker, config.get("default_voice_reference", "default")))


def dialogue_speaker_subject(config: dict, dialogue: dict) -> dict:
    """Return the subject that is allowed to deliver one authored line."""
    key = dialogue_speaker_key(config, dialogue)
    subjects = normalize_subjects(config)
    subject = next((item for item in subjects if item["id"] == key), None)
    if subject is not None:
        return subject
    if len(subjects) == 1:
        return subjects[0]
    raise ValueError(
        f"Dialogue speaker {dialogue.get('speaker', key)!r} must map to a configured subject; "
        f"got speaker_id={key!r}"
    )


def scene_voice_keys(config: dict, scenes: list[dict]) -> list[str]:
    profiles = voice_profiles(config)
    requested = {
        dialogue_speaker_key(config, scene["dialogue"])
        for scene in scenes
        if scene.get("dialogue")
    }
    if not requested:
        return []
    subject_order = [item["id"] for item in normalize_subjects(config)]
    keys = [key for key in subject_order if key in profiles and key in requested]
    keys.extend(key for key in profiles if key in requested and key not in keys)
    return keys


def subject_numbers(config: dict, scene: dict) -> list[int]:
    subjects = normalize_subjects(config)
    requested = set(str(item) for item in scene.get("subject_ids", []))
    return [index + 1 for index, item in enumerate(subjects) if not requested or item["id"] in requested]


def scene_subject_entries(config: dict, scene: dict) -> list[tuple[int, dict]]:
    """Return only the declared cast, retaining the configured subject numbers."""
    subjects = normalize_subjects(config)
    requested = {str(item) for item in scene.get("subject_ids", [])}
    return [
        (index + 1, subject)
        for index, subject in enumerate(subjects)
        if not requested or subject["id"] in requested
    ]


def shot_contract_prompt(config: dict, scene: dict, subject_entries: list[tuple[int, dict]]) -> str:
    """Build reusable physical-continuity and anatomy constraints for H3."""
    cast = ", ".join(f"<Subject {number}> ({subject['id']})" for number, subject in subject_entries)
    lines = [
        "SHOT INTEGRITY: make this one self-contained, physically continuous take. Use only the listed cast; do not invent, duplicate, merge, or swap people, animals, riders, or props.",
        "ANATOMY AND OBJECT CONTINUITY: every visible person has exactly two arms and two hands, every visible horse has one coherent body and four anatomically consistent legs, and ropes, reins, crutches, staffs, and clothing remain attached only where physically held or tied. No extra limbs, duplicate hands, fused bodies, floating props, or impossible attachments.",
        f"CAST LOCK: only {cast or 'the explicitly described subject'} may be visible, including the background and reflections. No unlisted extras, passersby, silhouettes, mannequins, portraits that look alive, or duplicate/partial bodies may appear. Keep each subject's identity, role, costume, scale, and position stable from the first frame to the last.",
        "CAMERA AND ACTION: use motivated camera motion and real contact, weight, momentum, and cause-and-effect. Do not use an unmotivated montage, time jump, or off-screen action to hide a continuity change.",
    ]
    if len(subject_entries) > 1:
        lines.append(
            "MULTI-SUBJECT SPATIAL LOCK: assign every listed subject a stable screen lane and depth layer. "
            "Keep a visible body-width gap unless the shot contract explicitly requires real contact; "
            "never let one subject pass through another, cross a torso, swap front/back depth ordering, "
            "or teleport across the frame. If paths would intersect, slow or stop the moving subject and "
            "stage the action side-by-side with clear occlusion and a complete visible cause-and-effect path."
        )
    contract = scene.get("shot_contract")
    if isinstance(contract, dict):
        render_strategy = str(contract.get("render_strategy", "")).strip()
        if render_strategy == "single_subject_safe":
            lines.append(
                "SINGLE-SUBJECT SAFETY: render only the first listed subject as a visible human. "
                "All other story subjects remain off-camera and must not appear as silhouettes, reflections, "
                "duplicates, partial bodies, or background extras. Keep the visible subject's action continuous "
                "and use narration to carry any off-camera story information."
            )
        primary = str(contract.get("primary_action", "")).strip()
        if primary:
            lines.append(f"PRIMARY ACTION: perform exactly this physical event and make its owner unambiguous: {primary}")
        owner = str(contract.get("action_owner", "")).strip()
        if owner:
            lines.append(f"ACTION OWNER: {owner} performs the primary action; no other subject performs or receives it.")
        rider = str(contract.get("rider", "")).strip()
        if rider:
            lines.append(f"RIDER LOCK: {rider}; if this says 'none', no person is mounted, seated on, or falling from a horse.")
        falling_subject = str(contract.get("falling_subject", "")).strip()
        if falling_subject:
            lines.append(f"FALLING SUBJECT LOCK: {falling_subject} is the only subject who falls; all other subjects remain upright and never appear to fall from the horse.")
        hand_plan = str(contract.get("hand_plan", "")).strip()
        if hand_plan:
            lines.append(f"HAND PLAN: {hand_plan}")
        spatial_plan = str(contract.get("spatial_plan", "")).strip()
        if spatial_plan:
            lines.append(f"SPATIAL PLAN: {spatial_plan}")
        beats = contract.get("beats")
        if isinstance(beats, list) and beats:
            beat_lines = []
            for beat in beats:
                if isinstance(beat, dict) and beat.get("time") and beat.get("action"):
                    beat_lines.append(f"{beat['time']}: {beat['action']}")
            if beat_lines:
                lines.append("TIMED BEATS (do not reorder or blend): " + " | ".join(beat_lines))
        forbidden = contract.get("forbidden")
        if isinstance(forbidden, list) and forbidden:
            lines.append("FORBIDDEN IN THIS TAKE: " + "; ".join(str(item) for item in forbidden))
    content = scene.get("content_contract") if isinstance(scene.get("content_contract"), dict) else {}
    must_show = [str(item).strip() for item in content.get("must_show", []) if str(item).strip()]
    if not must_show:
        visual = scene.get("visual_action") or scene.get("visual") or scene.get("visual_description")
        if visual:
            must_show = [str(visual).strip()]
    must_not_show = [str(item).strip() for item in content.get("must_not_show", []) if str(item).strip()]
    if not must_not_show:
        must_not_show = [
            "a decorative still portrait replacing the declared action",
            "unlisted people, animals, props, speech, or a different story event",
        ]
    if must_show:
        lines.append("NARRATIVE FACTS — MUST SHOW VERBATIM: " + "; ".join(must_show))
    lines.append("NARRATIVE FACTS — MUST NOT SHOW: " + "; ".join(must_not_show))
    lines.append(
        "SEMANTIC ACCEPTANCE: the final frame must still be a recognizable continuation of this exact shot's "
        "declared action. Do not replace the event with a decorative portrait, establishing still, unrelated "
        "reaction, flashback, dream, or montage. If an action cannot fit, simplify the camera move while "
        "preserving the named actor, object, direction, and result."
    )
    return "\n".join(lines)


def story_identity_prompt(config: dict, subject_entries: list[tuple[int, dict]]) -> str:
    """Build the hard story-level identity boundary shared by every H3 take."""
    namespace = identity_namespace(config)
    cast = []
    for number, subject in subject_entries:
        key = subject.get("identity_key") or f"{config_slug(config)}:{subject['id']}"
        traits = subject.get("identity_traits") or subject["description"]
        cast.append(f"<Subject {number}> identity_key={key}; role/traits={traits}")
    return "\n".join([
        f"STORY IDENTITY NAMESPACE: {namespace}. This namespace is isolated from every other story and prior run.",
        "STORY CAST UNIQUENESS LOCK: build every character only from the cast below and this story's own reference assets. Never reuse or import a face, hair, body proportions, costume palette, prop signature, background person, or voice identity from another story, including any shared/default identity portrait.",
        "IDENTITY CONTINUITY: the same identity_key must keep the same face, age, body, hair, costume, accessories, and role in every shot of this story. Different identity_keys must remain visibly different people or entities; never merge, swap, or clone them.",
        "CAST CARD:\n" + "\n".join(cast),
    ])


def dialogue_end(scene: dict, shot_seconds: float) -> float:
    dialogue = scene["dialogue"]
    spoken_chars = len(re.sub(r"[，。！？；：、,.!?;:]", "", str(dialogue["text"])))
    return min(shot_seconds - 0.4, float(dialogue["start"]) + max(1.8, min(3.0, spoken_chars / 5.0)))


def dialogue_lip_window(scene: dict, shot_seconds: float) -> tuple[float, float, str]:
    """Return the visual speaking window within a shot.

    ``dialogue.start`` is the requested H3 cue.  A generated take can still
    begin or finish its visible articulation at a different moment.  When a
    post-generation visual QC supplies ``dialogue.lip_sync``, that observed
    window is authoritative for the external clean dialogue track.
    """
    dialogue = scene["dialogue"]
    observed = dialogue.get("lip_sync")
    if observed is None:
        return float(dialogue["start"]), dialogue_end(scene, shot_seconds), "prompt"
    if not isinstance(observed, dict) or "start" not in observed or "end" not in observed:
        raise ValueError(f"Scene {scene['id']} dialogue.lip_sync requires start and end")
    start, end = float(observed["start"]), float(observed["end"])
    if not 0 <= start < end <= shot_seconds:
        raise ValueError(f"Scene {scene['id']} dialogue.lip_sync is outside the shot")
    return start, end, "observed_lips"


def build_prompt(config: dict, scene: dict, shot_seconds: float, audio_label_map: dict[str, int]) -> str:
    scene_id = scene["id"]
    visual = scene.get("visual_action") or scene.get("visual") or scene.get("visual_description")
    visual = visual or f"Show the story beat clearly and continuously: {scene['narration']}"
    action_timing = scene.get("action_timing", "Maintain continuous motivated motion from the first frame through the final frame.")
    sound_design = scene.get("sound_design", "Use physically synchronized location ambience and Foley for every visible action.")
    screenplay_direction = prompting_value(
        config,
        "screenplay",
        "SCREENPLAY DIRECTION: Treat this as a finished period-film scene, not a slideshow. Build a clear objective, visible cause-and-effect, reaction beats, and a motivated camera path; preserve spatial geography and let each shot change the dramatic situation.",
    )

    dialogue_direction = prompting_value(
        config,
        "dialogue",
        "DIALOGUE DIRECTION: Use concise, subtext-rich Chinese dialogue. Every authored line belongs only to its assigned speaker; show a reaction beat before or after it, keep listeners' mouths closed, and never invent extra speech, singing, or lip movement.",
    )
    visual_style = style_value(
        config,
        "visual",
        "Photorealistic cinematic live-action, natural skin and material detail, physically accurate lighting and motion blur, restrained camera movement, coherent continuity, and a phone-safe vertical composition.",
    )
    all_subjects = normalize_subjects(config)
    subject_entries = scene_subject_entries(config, scene)
    subjects = [subject for _, subject in subject_entries]
    subject_ids = [number for number, _ in subject_entries]
    if not subject_entries:
        subject_entries = list(enumerate(all_subjects, start=1))
        subjects = all_subjects
        subject_ids = [number for number, _ in subject_entries]
    subject_lines = [
        (
            f"<Subject {subject_id}> is {subject['description']}, defined by <Picture {int(subject['reference_picture'])}>."
            if "reference_picture" in subject
            else f"<Subject {subject_id}> is {subject['description']}; create this non-reference subject only as described."
        )
        for subject_id, subject in subject_entries
    ]
    audio_lines = []
    for key, number in sorted(audio_label_map.items(), key=lambda item: item[1]):
        matching_subject = next((index for index, item in subject_entries if item["id"] == key), None)
        if matching_subject:
            audio_lines.append(
                f"<Audio {number}> is the voice-timbre reference for <Subject {matching_subject}> (S{number})."
            )
        else:
            audio_lines.append(f"<Audio {number}> is a clean voice-timbre reference for speaker S{number}.")
    retention_lines = [
        f"<Subject {subject_id}> (appears in [Shot 1]): fully_preserved - preserve identity, face, hair, body proportions, costume, and distinguishing material details."
        for subject_id in subject_ids
    ]
    retention_lines.extend(
        f"<Audio {number}>: reference - follow its voice timbre and natural delivery without copying the source signal."
        for number in sorted(audio_label_map.values())
    )
    audio_refs_text = " and ".join(f"<Audio {number}>" for number in sorted(audio_label_map.values()))
    picture_refs = sorted({int(subject["reference_picture"]) for _, subject in subject_entries if "reference_picture" in subject})
    picture_refs_text = " and ".join(f"<Picture {number}>" for number in picture_refs) or "the written subject descriptions"
    dialogue = scene.get("dialogue")
    if dialogue:
        key = dialogue_speaker_key(config, dialogue)
        audio_ref = f"<Audio {audio_label_map.get(key, 1)}>"
        speaker_subject = dialogue_speaker_subject(config, dialogue)
        subject_id = next(
            index + 1 for index, item in enumerate(subjects) if item["id"] == speaker_subject["id"]
        )
        speaker_id = audio_label_map.get(key, 1)
        speaker_name = str(dialogue.get("speaker") or speaker_subject["id"])
        language = dialogue.get("language_tag") or dialogue.get("language") or "Chinese"
        finish = dialogue_end(scene, shot_seconds)
        if audio_spine_enabled(config, dialogue):
            timing_rule = (
                "A synchronized Audio Spine supplies the exact final dialogue waveform for this shot. "
                "Treat its first audible phoneme as the cue for the first visible mouth motion, match every syllable with one-to-one lip articulation, and let the final audible phoneme coincide with the final mouth closure. "
            )
            ending_rule = "The delivery ends only when the Audio Spine ends; lips then meet and remain closed through the rest of the shot."
        else:
            timing_rule = ""
            ending_rule = f"The delivery finishes by {finish:.1f} seconds; lips meet and remain closed through the rest of the shot."
        performance = (
            f"DIALOGUE ASSIGNMENT: {speaker_name} is <Subject {subject_id}> and the sole speaker (S{speaker_id}). "
            f"Never transfer this line, its voice, or its lip motion to any other subject. "
            f"Keep <Subject {subject_id}>'s lips fully closed before {float(dialogue['start']):.1f} seconds. At exactly {float(dialogue['start']):.1f} seconds, hold a stable mouth-visible medium close-up and have <Subject {subject_id}> (S{speaker_id}) speak one clearly articulated line in natural performance using the timbre in {audio_ref}: <d>[{language}] {dialogue['text']}</d>. {timing_rule}Do not turn away, cut, or make unrelated mouth movements during this line. {ending_rule} This authored line is the only intentional human speech in the shot."
        )
        if dialogue.get("speaker_lock", False):
            performance += (
                f" Speaker-lock take: only <Subject {subject_id}> is visible during the authored line; "
                "any listener remains off-camera and must never mouth, repeat, or react with speech. "
                "This speaker-lock instruction overrides incidental two-person staging in the story beat."
            )
    else:
        performance = "All visible people communicate through eye movement, hand gestures, walking, and physical action. Keep their lips naturally closed; no unscripted speech or singing."
    return "\n\n".join(
        [
            "subject_definitions:\n" + story_identity_prompt(config, subject_entries) + "\n" + "\n".join(subject_lines + audio_lines),
            f"summary:\n[reference generation + audio reference] Native 9:16 portrait story beat {scene_id}, approximately {shot_seconds:.1f} seconds at {int(video_value(config, 'fps', 24))} fps, using {picture_refs_text} for identity and costume continuity and {audio_refs_text} for voice-timbre reference. Continuous physical motion fills a phone-safe vertical composition.",
            "retention_analysis:\n" + "\n".join(retention_lines),
            f"detailed_description:\n{screenplay_direction}\n{dialogue_direction}\n{visual_style}\n"
            f"SCENE TRUTH: narration is context only; render exactly the visible action below and do not invent a different event.\n"
            f"[Shot 1] {visual}\nAction timeline: {action_timing}\n{performance}\n{shot_contract_prompt(config, scene, subject_entries)}",
            f"overall_soundscape: {sound_design} {style_value(config, 'soundscape', 'Sound layers remain separated, naturally dynamic, and synchronized to visible causes.')}",
            f"non_diegetic_music: {style_value(config, 'music', 'N/A')}",
        ]
    )


def write_scene_manifest(config: dict, paths: RuntimePaths) -> Path:
    """Persist the exact narrative contract used by every H3 prompt.

    This sidecar is intentionally human-readable: it lets a producer compare
    narration, declared visible action, cast, dialogue owner, and forbidden
    content before spending GPU time, and gives later QC one authoritative
    source instead of reverse-engineering a generated MP4.
    """
    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    manifest = []
    for scene in config["scenes"]:
        keys = scene_voice_keys(config, [scene])
        audio_label_map = {key: index + 1 for index, key in enumerate(keys)}
        manifest.append({
            "scene_id": int(scene["id"]),
            "subject_ids": [str(item) for item in scene.get("subject_ids", [])],
            "narration": str(scene["narration"]),
            "visual_action": str(scene.get("visual_action") or scene.get("visual") or scene.get("visual_description") or ""),
            "content_contract": scene.get("content_contract", {}),
            "shot_contract": scene.get("shot_contract", {}),
            "dialogue": scene.get("dialogue"),
            "duration_seconds": shot_seconds,
            "prompt": build_prompt(config, scene, shot_seconds, audio_label_map),
        })
    destination = paths.run_dir / "scene_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "workflow_signature_version": WORKFLOW_SIGNATURE_VERSION,
        "title": config["title"],
        "identity_namespace": identity_namespace(config),
        "identity_policy": identity_policy(config) or {"legacy": True, "shared_reference_ignored": True},
        "generation": config.get("generation", {}),
        "scenes": manifest,
    }, ensure_ascii=False, indent=2) + "\n")
    return destination


def validate_config(config: dict) -> None:
    if not str(config.get("title", "")).strip():
        raise ValueError("config.title is required")
    scenes = config.get("scenes")
    chains = config.get("chains")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("config.scenes must be a non-empty list")
    if not isinstance(chains, list) or not chains:
        raise ValueError("config.chains must be a non-empty list")
    scene_ids = [int(scene.get("id")) for scene in scenes]
    flattened = [int(scene_id) for chain in chains for scene_id in chain]
    if flattened != scene_ids:
        raise ValueError("config.chains must list every scene id exactly once, in scene order")
    bgm = config.get("bgm", {})
    if bgm is not False and isinstance(bgm, dict) and bgm.get("enabled", False):
        source_type = str(bgm.get("source_type", "file")).lower()
        if source_type == "download" and not str(bgm.get("source_url") or bgm.get("url") or "").strip():
            raise ValueError("bgm.source_type='download' requires a non-empty bgm.source_url")
        if source_type == "file" and not str(bgm.get("file") or "").strip():
            raise ValueError("bgm.source_type='file' requires bgm.file")
        if source_type == "generated" and not bool(bgm.get("allow_synthetic_fallback", False)):
            raise ValueError("bgm.source_type='generated' requires explicit bgm.allow_synthetic_fallback=true")
        if source_type not in {"file", "download", "generated", "none", "off", "disabled"}:
            raise ValueError(f"Unsupported bgm.source_type: {source_type}")
    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    known_subject_ids = {subject["id"] for subject in normalize_subjects(config)}
    for scene in scenes:
        for key in ("id", "narration", "narration_offset", "action_timing", "sound_design"):
            if key not in scene:
                raise ValueError(f"scene {scene.get('id')} missing {key}")
        if not (scene.get("visual_action") or scene.get("visual") or scene.get("visual_description")):
            raise ValueError(f"scene {scene.get('id')} must declare visual_action (the narrative acceptance target)")
        dialogue = scene.get("dialogue")
        declared_subject_ids = [str(item) for item in scene.get("subject_ids", [])]
        if len(declared_subject_ids) != len(set(declared_subject_ids)):
            raise ValueError(f"scene {scene['id']} subject_ids contains duplicates")
        unknown_subject_ids = sorted(set(declared_subject_ids) - known_subject_ids)
        if unknown_subject_ids:
            raise ValueError(f"scene {scene['id']} references unknown subjects: {unknown_subject_ids}")
        contract = scene.get("shot_contract")
        if contract is not None:
            if not isinstance(contract, dict):
                raise ValueError(f"scene {scene['id']} shot_contract must be an object")
            for key in ("action_owner", "rider", "falling_subject"):
                value = contract.get(key)
                if value and str(value).lower() not in {"none", "无", "n/a"} and str(value) not in known_subject_ids:
                    raise ValueError(f"scene {scene['id']} shot_contract.{key} references unknown subject {value!r}")
                if value and str(value).lower() not in {"none", "无", "n/a"} and declared_subject_ids and str(value) not in declared_subject_ids:
                    raise ValueError(f"scene {scene['id']} shot_contract.{key} must be listed in scene.subject_ids")
            beats = contract.get("beats")
            if beats is not None and (not isinstance(beats, list) or any(
                not isinstance(beat, dict) or not beat.get("time") or not beat.get("action")
                for beat in beats
            )):
                raise ValueError(f"scene {scene['id']} shot_contract.beats must contain time/action objects")
        if dialogue:
            if not dialogue.get("text") or "start" not in dialogue or not dialogue.get("speaker"):
                raise ValueError(f"scene {scene['id']} dialogue requires speaker, text, and start")
            if not 0 <= float(dialogue["start"]) < shot_seconds:
                raise ValueError(f"scene {scene['id']} dialogue.start is outside the shot")
            speaker_subject = dialogue_speaker_subject(config, dialogue)
            if scene.get("subject_ids") and speaker_subject["id"] not in {str(item) for item in scene["subject_ids"]}:
                raise ValueError(
                    f"scene {scene['id']} dialogue speaker {speaker_subject['id']} is not listed in scene.subject_ids"
                )
            declared_subject_id = dialogue.get("subject_id")
            if declared_subject_id is not None:
                expected_subject_id = next(
                    index + 1
                    for index, subject in enumerate(normalize_subjects(config))
                    if subject["id"] == speaker_subject["id"]
                )
                if int(declared_subject_id) != expected_subject_id:
                    raise ValueError(
                        f"scene {scene['id']} dialogue.subject_id disagrees with speaker {speaker_subject['id']}"
                    )
            dialogue_lip_window(scene, shot_seconds)
    if any(len(scene_ids) > 0 for scene_ids in chains):
        for chain in chains:
            keys = scene_voice_keys(config, [next(scene for scene in scenes if int(scene["id"]) == int(sid)) for sid in chain])
            if len(keys) > 3:
                raise ValueError("each chain can reference at most three voice profiles")


def graph_for_chain(
    config: dict,
    paths: RuntimePaths,
    chain_index: int,
    scene_ids: list[int],
    guide_audio: str | None = None,
) -> dict:
    scenes_by_id = {int(scene["id"]): scene for scene in config["scenes"]}
    chain_scenes = [scenes_by_id[int(scene_id)] for scene_id in scene_ids]
    keys = scene_voice_keys(config, chain_scenes)
    if len(keys) > 3:
        raise ValueError(f"chain {chain_index} references more than three voice profiles: {keys}")
    audio_label_map = {key: index + 1 for index, key in enumerate(keys)}
    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    script = "\n---\n".join(build_prompt(config, scene, shot_seconds, audio_label_map) for scene in chain_scenes)
    generation = config.get("generation", {}) if isinstance(config.get("generation", {}), dict) else {}
    continuity = str(generation.get("continuity", "cut"))
    if continuity == "context_pin" and not bool(generation.get("motion_context_installed", False)):
        raise RuntimeError(
            "generation.continuity='context_pin' requires ComfyUI-H3-Motion-Context; "
            "set generation.continuity='first_frame' or 'cut' when that pack is unavailable"
        )
    graph = {
        "1": {"class_type": "H3ModelLoaderAny", "inputs": {"model_name": UNET, "activation_reserve_gb": 6.0}},
        "2": {"class_type": "H3ClipLoaderAny", "inputs": {"clip_name": CLIP, "type": "minimax", "mmproj_name": "(auto)"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {
            "class_type": "H3MultishotMemorySampler",
            "inputs": {
                "model": ["1", 0], "clip": ["2", 0], "video_vae": ["3", 0], "audio_vae": ["4", 0],
                "script": script, "shot_count": 0,
                "width": int(video_value(config, "width", 768)), "height": int(video_value(config, "height", 1344)),
                "frames_per_shot": int(video_value(config, "frames_per_shot", 362)),
                "seed": int(config.get("seed", 920260901)) + chain_index * 104729,
                "steps": int(video_value(config, "steps", 14)), "seed_per_shot": True, "memory_frames": 0,
                "anchor_frames": 1, "sampler_name": video_value(config, "sampler", "euler"),
                "scheduler": video_value(config, "scheduler", "beta57"),
                "bank_pinned": int(generation.get("bank_pinned", 1)),
                "chain_gain_control": str(generation.get("chain_gain_control", "off")),
                "continuity": continuity,
                "bank_clip_frames": int(generation.get("bank_clip_frames", 22)),
                "color_level": str(generation.get("color_level", "off")),
                "join_anchor_noise": float(generation.get("join_anchor_noise", 0.0)),
                "join_blend": bool(generation.get("join_blend", False)),
                "handoff_release": float(generation.get("handoff_release", 0.30)),
                "bank_ref_noise": float(generation.get("bank_ref_noise", 0.0)),
                "end_anchor": bool(generation.get("end_anchor", False)),
                "join_fx": "off", "audio_lock": bool(generation.get("audio_lock", False)),
                "handoff_taper": int(generation.get("handoff_taper", 0)),
                "handoff_depth": str(generation.get("handoff_depth", "block")),
                "self_anchor_voice": bool(generation.get("self_anchor_voice", False)),
                "reference_image_size": "match", "preview_first_shot": True,
                "save_every_shot": True, "output_scale": 1.0, "upscale_model_name": "(none)",
                "master_normalize": str(generation.get("master_normalize", "luma+contrast")),
                "pin_frames": str(generation.get("pin_frames", "22")),
                "pin_noise": float(generation.get("pin_noise", 0.0)),
                "pin_renorm": bool(generation.get("pin_renorm", False)),
                "reference_subjects": str(config.get("assets", {}).get("reference_subjects", "")), "low_ram_master": True, "audio_pin_frames": 0, "pin_noise_audio": False,
                "audio_tone_control": False, "x0_texture_clamp": 0.0, "refresh_renoise": False, "pin_noise_ramp": False,
                "auto_chunk_ffn": True, "x0_clamp_window": 0.30, "sampler_2": "(off)", "sampler_2_at": 0.40,
            },
        },
        "7": {"class_type": "SaveAudio", "inputs": {"audio": ["6", 1], "filename_prefix": f"workflow2/{paths.slug}/native_chain_{chain_index:02d}_audio"}},
    }
    if paths.comfy_identity_ref:
        graph["5"] = {"class_type": "LoadImage", "inputs": {"image": paths.comfy_identity_ref}}
        graph["6"]["inputs"]["start_image"] = ["5", 0]
    for node_id, key in zip((8, 9, 10), keys):
        graph[str(node_id)] = {"class_type": "LoadAudio", "inputs": {"audio": paths.comfy_voice_refs[key]}}
        input_name = "voice_ref" if node_id == 8 else f"voice_ref_{node_id - 7}"
        graph["6"]["inputs"][input_name] = [str(node_id), 0]
    if guide_audio:
        graph["11"] = {"class_type": "LoadAudio", "inputs": {"audio": guide_audio}}
        graph["6"]["inputs"]["guide_audio"] = ["11", 0]
    return graph


def wait_for_server(base_url: str, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/system_stats", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"ComfyUI did not become ready at {base_url}")


def ensure_server(base_url: str) -> None:
    try:
        wait_for_server(base_url, timeout=3)
        return
    except RuntimeError:
        pass
    port = base_url.rsplit(":", 1)[-1]
    log_path = WORK_DIR / "comfyui_workflow2.log"
    log_handle = log_path.open("ab")
    subprocess.Popen(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "main.py"), "--listen", "127.0.0.1", "--port", port, "--disable-auto-launch", "--fast-disk", "--disable-pinned-memory", "--cache-none"],
        cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
    )
    wait_for_server(base_url)


def stage_identity(paths: RuntimePaths) -> None:
    if paths.identity_source is None:
        return
    if not paths.identity_source.exists():
        raise RuntimeError(f"Configured identity reference does not exist: {paths.identity_source}")
    destination = ROOT / "input" / paths.comfy_identity_ref
    destination.parent.mkdir(parents=True, exist_ok=True)
    if paths.identity_source.resolve() != destination.resolve():
        shutil.copy2(paths.identity_source, destination)


def stage_subject_identity_references(config: dict, paths: RuntimePaths) -> dict[str, str]:
    """Stage one dedicated H3 identity input for each declared subject.

    The combined reference board remains useful for silent ensemble shots.  A
    speaking shot, however, must be able to give H3 exactly one face so the
    voice, mouth motion, and identity cannot migrate to a second person.
    """
    configured = config.get("assets", {}).get("identity_references", {})
    if not isinstance(configured, dict):
        raise RuntimeError("assets.identity_references must be an object mapping subject ids to files")
    staged: dict[str, str] = {}
    for subject_id, configured_path in configured.items():
        source = resolve_project_path(configured_path, paths.config_path)
        if not source.exists() and (paths.assets_dir / str(configured_path)).exists():
            source = paths.assets_dir / str(configured_path)
        if "identity_policy" not in config and not _is_relative_to(source, paths.assets_dir):
            print(
                f"IDENTITY POLICY: ignoring external legacy portrait for {subject_id}; using this story's written character traits",
                flush=True,
            )
            continue
        if not source.exists():
            raise RuntimeError(f"Configured identity reference for {subject_id} does not exist: {source}")
        suffix = source.suffix.lower() or ".png"
        destination = ROOT / "input" / f"workflow2_{paths.slug}_{slugify(str(subject_id))}_identity{suffix}"
        shutil.copy2(source, destination)
        staged[str(subject_id)] = destination.name
    return staged


def speaker_locked_dialogue_config(config: dict, scene: dict) -> tuple[dict, dict, str]:
    """Build a single-subject H3 context for one authored dialogue shot."""
    dialogue = scene.get("dialogue")
    if not dialogue:
        raise ValueError(f"scene {scene.get('id')} has no dialogue to speaker-lock")
    speaker_key = dialogue_speaker_key(config, dialogue)
    speaker_subject = dialogue_speaker_subject(config, dialogue)
    local_scene = copy.deepcopy(scene)
    local_scene["subject_ids"] = [speaker_subject["id"]]
    local_dialogue = local_scene["dialogue"]
    local_dialogue.pop("subject_id", None)
    local_dialogue["speaker_id"] = speaker_key
    local_dialogue["speaker_lock"] = True

    local_config = copy.deepcopy(config)
    identity_references = config.get("assets", {}).get("identity_references", {})
    configured_reference = identity_references.get(speaker_key) if isinstance(identity_references, dict) else None
    assets_setting = config.get("assets", {}) if isinstance(config.get("assets", {}), dict) else {}
    assets_root = Path(str(assets_setting.get("directory", "input/story_workflow2")))
    if not assets_root.is_absolute():
        assets_root = ROOT / assets_root
    configured_path = Path(str(configured_reference)) if configured_reference else None
    external_legacy_reference = bool(configured_path and configured_path.is_absolute() and not _is_relative_to(configured_path, assets_root))
    has_identity_reference = bool(configured_reference) and not is_legacy_shared_identity_value(config, configured_reference) and not external_legacy_reference
    speaker_definition = dict(speaker_subject)
    if has_identity_reference:
        speaker_definition["reference_picture"] = 1
    else:
        speaker_definition.pop("reference_picture", None)
    local_config["subjects"] = [
        speaker_definition
    ]
    profiles = voice_profiles(config)
    if speaker_key in profiles:
        local_config["voice_references"] = {speaker_key: profiles[speaker_key]}
    local_config["scenes"] = [local_scene]
    local_config["chains"] = [[int(local_scene["id"])]]
    local_config.setdefault("assets", {})["reference_subjects"] = ""
    return local_config, local_scene, speaker_key


async def synthesize_voice_refs(config: dict, paths: RuntimePaths, force: bool) -> None:
    profiles = voice_profiles(config)
    if not profiles:
        return
    paths.voice_ref_dir.mkdir(parents=True, exist_ok=True)
    comfy_input_dir = ROOT / "input"
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    for key, profile in profiles.items():
        configured = profile.get("file") or profile.get("path")
        source = resolve_project_path(configured, paths.config_path) if configured else paths.assets_dir / f"{key}_voice_ref.wav"
        if configured and not source.exists() and (paths.assets_dir / str(configured)).exists():
            source = paths.assets_dir / str(configured)
        wav_path = paths.voice_ref_dir / f"{slugify(key)}.wav"
        if source.exists() and valid_audio_reference(source):
            shutil.copy2(source, wav_path)
        elif force or not valid_audio_reference(wav_path):
            text = str(profile.get("sample_text", profile.get("text", key)))
            voice = str(profile.get("voice", "zh-CN-YunxiNeural"))
            mp3_path = paths.voice_ref_dir / f"{slugify(key)}.mp3"
            print(f"VOICE REF {key}", flush=True)
            await edge_tts.Communicate(text, voice, rate=str(profile.get("rate", "-4%")), pitch=str(profile.get("pitch", "+0Hz")), volume=str(profile.get("volume", "+0%"))).save(str(mp3_path))
            run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp3_path), "-af", "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-45dB:stop_periods=-1:stop_duration=0.18:stop_threshold=-45dB,apad=pad_dur=0.25", "-ar", "32000", "-ac", "2", "-c:a", "pcm_s16le", str(wav_path)])
        if not valid_audio_reference(wav_path):
            raise RuntimeError(f"Voice reference {key} is missing or invalid: {source}")
        destination = comfy_input_dir / f"workflow2_{paths.slug}_{slugify(key)}_voice_ref.wav"
        shutil.copy2(wav_path, destination)
        paths.comfy_voice_refs[key] = destination.name


def submit_and_wait(base_url: str, graph: dict, chain_index: int) -> None:
    response = requests.post(f"{base_url}/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=60)
    if not response.ok:
        raise RuntimeError(f"Chain {chain_index} rejected: {response.status_code} {response.text}")
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Chain {chain_index} rejected: {json.dumps(payload, ensure_ascii=False)}")
    prompt_id = payload["prompt_id"]
    print(f"CHAIN {chain_index:02d} QUEUED {prompt_id}", flush=True)
    while True:
        history = requests.get(f"{base_url}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"Chain {chain_index} failed: {json.dumps(status.get('messages', []), ensure_ascii=False)}")
            if status.get("completed"):
                print(f"CHAIN {chain_index:02d} COMPLETED", flush=True)
                return
        time.sleep(8)


def newest_streamed_master(started_at: float) -> Path:
    candidates = [path for path in STREAM_DIR.glob("master_*.mp4") if path.stat().st_mtime >= started_at - 2]
    if not candidates:
        raise RuntimeError("H3 streamed master was not found after completed render")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def prepare_h3_generation(config: dict, paths: RuntimePaths, base_url: str) -> None:
    validate_story_identity_policy(config, paths.config_path)
    write_story_identity_manifest(config, paths)
    register_story_identity(config, paths)
    model_path = ROOT / "models" / "diffusion_models" / UNET
    if not model_path.exists() or model_path.stat().st_size < 20_000_000_000:
        raise RuntimeError(f"Required Ref2VA model is incomplete: {model_path}")
    stage_identity(paths)
    stage_subject_identity_references(config, paths)
    asyncio.run(synthesize_voice_refs(config, paths, force=False))
    asyncio.run(synthesize_dialogues(config, paths, force=False))
    if not paths.comfy_voice_refs:
        raise RuntimeError("No voice_references are configured; add at least one voice profile for H3 audio conditioning")
    STREAM_DIR.mkdir(parents=True, exist_ok=True)
    paths.chains_dir.mkdir(parents=True, exist_ok=True)
    paths.workflows_dir.mkdir(parents=True, exist_ok=True)
    ensure_server(base_url)


def split_chain_for_speaker_lock(config: dict, scene_ids: list[int]) -> list[tuple[list[int], bool]]:
    """Split dialogue and high-risk action shots into isolated H3 contexts."""
    scenes_by_id = {int(scene["id"]): scene for scene in config["scenes"]}
    generation = config.get("generation", {})
    scene_mode = str(generation.get("scene_mode", "single_shot") if isinstance(generation, dict) else "single_shot").lower()
    if scene_mode in {"single_shot", "one_scene_per_take", "isolated"}:
        # H3's multishot node is excellent for one coherent take, but asking it
        # to depict several unrelated plot beats in one prompt invites it to
        # invent bridges and makes the external narration disagree with the
        # rendered action.  Keep every authored scene as one 15s take by
        # default; the final assembler still preserves the story's chain order.
        return [([int(scene_id)], bool(scenes_by_id[int(scene_id)].get("dialogue"))) for scene_id in scene_ids]
    parts: list[tuple[list[int], bool]] = []
    silent_ids: list[int] = []
    for scene_id in scene_ids:
        scene = scenes_by_id[int(scene_id)]
        contract = scene.get("shot_contract")
        isolated = bool(scene.get("isolated", False)) or (
            isinstance(contract, dict) and bool(contract.get("isolated", False))
        )
        if scene.get("dialogue") or isolated:
            if silent_ids:
                parts.append((silent_ids, False))
                silent_ids = []
            parts.append(([int(scene_id)], bool(scene.get("dialogue"))))
        else:
            silent_ids.append(int(scene_id))
    if silent_ids:
        parts.append((silent_ids, False))
    return parts


def scoped_component_context(
    config: dict,
    paths: RuntimePaths,
    source_scenes: list[dict],
    subject_identity_inputs: dict[str, str],
) -> tuple[dict, RuntimePaths, list[dict], int]:
    """Give each component only its declared cast and matching identity batch."""
    all_subjects = normalize_subjects(config)
    requested = {
        str(subject_id)
        for scene in source_scenes
        for subject_id in scene.get("subject_ids", [])
    }
    local_subjects = [subject for subject in all_subjects if subject["id"] in requested]
    if not local_subjects:
        local_subjects = list(all_subjects)
    referenced_ids = {subject["id"] for subject in local_subjects if subject["id"] in subject_identity_inputs}
    referenced_subjects = [subject for subject in local_subjects if subject["id"] in referenced_ids]
    local_config = copy.deepcopy(config)
    local_config["subjects"] = []
    for subject in local_subjects:
        item = copy.deepcopy(subject)
        if item["id"] in referenced_ids:
            item["reference_picture"] = next(
                index for index, ref in enumerate(referenced_subjects, start=1) if ref["id"] == item["id"]
            )
        else:
            item.pop("reference_picture", None)
        local_config["subjects"].append(item)
    local_config["scenes"] = copy.deepcopy(source_scenes)
    local_config["chains"] = [[int(scene["id"]) for scene in source_scenes]]
    local_config["assets"] = copy.deepcopy(local_config.get("assets", {}))
    local_config["assets"]["reference_subjects"] = "1,1" if len(referenced_subjects) > 1 else ""
    local_paths = copy.copy(paths)
    if len(referenced_subjects) == 1:
        local_paths.comfy_identity_ref = subject_identity_inputs[referenced_subjects[0]["id"]]
    elif not referenced_subjects:
        # Never pass another character's portrait as a substitute identity.
        # Text-only subjects are safer than making a host/coachman inherit the
        # protagonist's face and then attaching the wrong dialogue to it.
        local_paths.comfy_identity_ref = None
    return local_config, local_paths, local_subjects, len(referenced_subjects)


def join_generated_components(sources: list[Path], destination: Path) -> None:
    """Join independently generated components without changing their pixels."""
    if not sources:
        raise ValueError("cannot join an empty component list")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(sources) == 1:
        shutil.copy2(sources[0], destination)
        return
    manifest = destination.with_suffix(".components.txt")
    manifest.write_text("".join(f"file '{source}'\n" for source in sources))
    try:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(manifest), "-c", "copy", "-movflags", "+faststart", str(destination),
        ])
    finally:
        manifest.unlink(missing_ok=True)


def generate(config: dict, paths: RuntimePaths, base_url: str, force: bool) -> None:
    prepare_h3_generation(config, paths, base_url)
    write_scene_manifest(config, paths)
    subject_identity_inputs = stage_subject_identity_references(config, paths)
    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    for chain_index, scene_ids in enumerate(config["chains"], start=1):
        destination = paths.chains_dir / f"chain_{chain_index:02d}.mp4"
        expected = len(scene_ids) * shot_seconds
        chain_signature = render_signature(config, [int(scene_id) for scene_id in scene_ids])
        if not force and valid_video(destination, expected - 3) and signature_matches(destination, chain_signature):
            print(f"CHAIN {chain_index:02d} SKIPPED {destination}", flush=True)
            continue
        component_paths: list[Path] = []
        components_dir = paths.chains_dir / f"chain_{chain_index:02d}_components"
        for component_index, (component_scene_ids, speaker_locked) in enumerate(
            split_chain_for_speaker_lock(config, [int(scene_id) for scene_id in scene_ids]), start=1
        ):
            component_name = "_".join(f"{scene_id:02d}" for scene_id in component_scene_ids)
            component_destination = components_dir / f"component_{component_index:02d}_{component_name}.mp4"
            component_expected = len(component_scene_ids) * shot_seconds
            component_paths.append(component_destination)
            component_signature = render_signature(config, [int(scene_id) for scene_id in component_scene_ids])
            if not force and valid_video(component_destination, component_expected - 3) and signature_matches(component_destination, component_signature):
                print(f"CHAIN {chain_index:02d} COMPONENT {component_index:02d} SKIPPED {component_destination}", flush=True)
                continue

            local_config = config
            local_paths = paths
            if speaker_locked:
                source_scene = next(scene for scene in config["scenes"] if int(scene["id"]) == component_scene_ids[0])
                local_config, _, speaker_key = speaker_locked_dialogue_config(config, source_scene)
                local_paths = copy.copy(paths)
                local_paths.comfy_identity_ref = subject_identity_inputs.get(speaker_key)
                if speaker_key in paths.comfy_voice_refs:
                    local_paths.comfy_voice_refs = {speaker_key: paths.comfy_voice_refs[speaker_key]}
            else:
                component_scenes = [
                    next(scene for scene in config["scenes"] if int(scene["id"]) == int(scene_id))
                    for scene_id in component_scene_ids
                ]
                local_config, local_paths, _, _ = scoped_component_context(
                    config, paths, component_scenes, subject_identity_inputs
                )

            guide_audio = stage_audio_spine(
                config,
                paths,
                key=f"chain_{chain_index:02d}_component_{component_index:02d}",
                scene_ids=component_scene_ids,
                force=force,
            )
            graph = graph_for_chain(
                local_config,
                local_paths,
                chain_index * 100 + component_index,
                component_scene_ids,
                guide_audio,
            )
            workflow_path = paths.workflows_dir / f"chain_{chain_index:02d}_component_{component_index:02d}_api.json"
            workflow_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
            started_at = time.time()
            submit_and_wait(base_url, graph, chain_index * 100 + component_index)
            streamed = newest_streamed_master(started_at)
            component_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(streamed, component_destination)
            if not valid_video(component_destination, component_expected - 3):
                raise RuntimeError(
                    f"Chain {chain_index:02d} component {component_index:02d} failed duration validation: {component_destination}"
                )
            write_signature(component_destination, component_signature)
            print(f"CHAIN {chain_index:02d} COMPONENT {component_index:02d} SAVED {component_destination}", flush=True)

        join_generated_components(component_paths, destination)
        if not valid_video(destination, expected - 3):
            raise RuntimeError(f"Chain {chain_index} failed duration validation: {destination}")
        write_signature(destination, chain_signature)
        print(f"CHAIN {chain_index:02d} SAVED {destination}", flush=True)


def generate_audio_spine_dialogue_scenes(
    config: dict,
    paths: RuntimePaths,
    base_url: str,
    scene_ids: list[int],
    force: bool,
) -> dict[int, Path]:
    """Re-render only the authored-dialogue shots with H3's Audio Spine.

    This is the targeted repair path for an existing film.  New stories use
    the ordinary full-chain visual canvas, which wires the same guide audio to
    each applicable H3 sampler automatically.
    """
    prepare_h3_generation(config, paths, base_url)
    subject_identity_inputs = stage_subject_identity_references(config, paths)
    scenes_by_id = {int(scene["id"]): scene for scene in config["scenes"]}
    requested = list(dict.fromkeys(int(scene_id) for scene_id in scene_ids))
    if not requested:
        raise ValueError("at least one dialogue scene is required")
    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    output_dir = paths.chains_dir / "audio_spine_dialogue"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[int, Path] = {}
    for scene_id in requested:
        scene = scenes_by_id.get(scene_id)
        if scene is None:
            raise ValueError(f"Scene {scene_id} does not exist")
        dialogue = scene.get("dialogue")
        if not dialogue:
            raise ValueError(f"Scene {scene_id} has no authored dialogue")
        if not audio_spine_enabled(config, dialogue):
            raise ValueError(f"Scene {scene_id} has audio_spine disabled in its story configuration")
        destination = output_dir / f"scene_{scene_id:02d}.mp4"
        if not force and valid_video(destination, shot_seconds - 0.2):
            rendered[scene_id] = destination
            print(f"AUDIO SPINE SCENE {scene_id:02d} SKIPPED {destination}", flush=True)
            continue
        guide_audio = stage_audio_spine(
            config,
            paths,
            key=f"scene_{scene_id:02d}",
            scene_ids=[scene_id],
            force=force,
        )
        if not guide_audio:
            raise AssertionError(f"Scene {scene_id} unexpectedly has no audio spine")
        local_config, _, speaker_key = speaker_locked_dialogue_config(config, scene)
        local_paths = copy.copy(paths)
        local_paths.comfy_identity_ref = subject_identity_inputs.get(speaker_key)
        local_paths.comfy_voice_refs = {
            speaker_key: paths.comfy_voice_refs[speaker_key]
        } if speaker_key in paths.comfy_voice_refs else dict(paths.comfy_voice_refs)
        graph = graph_for_chain(local_config, local_paths, 1000 + scene_id, [scene_id], guide_audio)
        (paths.workflows_dir / f"audio_spine_scene_{scene_id:02d}_api.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2) + "\n"
        )
        started_at = time.time()
        submit_and_wait(base_url, graph, scene_id)
        streamed = newest_streamed_master(started_at)
        shutil.copy2(streamed, destination)
        if not valid_video(destination, shot_seconds - 0.2):
            raise RuntimeError(f"Audio Spine scene {scene_id} failed duration validation: {destination}")
        rendered[scene_id] = destination
        print(f"AUDIO SPINE SCENE {scene_id:02d} SAVED {destination}", flush=True)
    return rendered


def prepare_visual_workflow(config: dict, paths: RuntimePaths, force_voice_refs: bool = False) -> tuple[Path, Path]:
    """Stage assets and create a zero-intervention speaker-locked canvas.

    Every authored-dialogue scene is rendered as its own H3 component.  The
    component receives only the declared speaker's identity image, voice
    reference, and Audio Spine.  The canvas then joins those components back
    into their original narrative chains before the final assembler runs.
    """
    validate_story_identity_policy(config, paths.config_path)
    write_story_identity_manifest(config, paths)
    register_story_identity(config, paths)
    stage_identity(paths)
    write_scene_manifest(config, paths)
    bgm_input = stage_bgm(config, paths, force=force_voice_refs)
    subject_identity_inputs = stage_subject_identity_references(config, paths)
    asyncio.run(synthesize_voice_refs(config, paths, force=force_voice_refs))
    asyncio.run(synthesize_dialogues(config, paths, force=force_voice_refs))
    if not paths.comfy_voice_refs:
        raise RuntimeError("No voice_references are configured; add at least one voice profile for the H3 visual workflow")
    profiles = voice_profiles(config)
    ordered_keys = scene_voice_keys(config, config["scenes"])
    voice_inputs = [(key, paths.comfy_voice_refs[key]) for key in ordered_keys if key in profiles][:3]
    generation = config.get("generation", {}) if isinstance(config.get("generation", {}), dict) else {}
    scenes_by_id = {int(scene["id"]): scene for scene in config["scenes"]}
    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    chain_specs: list[dict[str, Any]] = []
    chain_part_indices: list[list[int]] = []

    for original_chain_index, original_scene_ids in enumerate(config["chains"], start=1):
        parts = split_chain_for_speaker_lock(config, [int(scene_id) for scene_id in original_scene_ids])

        part_indices: list[int] = []
        for part_number, (scene_ids, speaker_locked) in enumerate(parts, start=1):
            source_scenes = [copy.deepcopy(scenes_by_id[scene_id]) for scene_id in scene_ids]
            if speaker_locked:
                local_config, local_scene, speaker_key = speaker_locked_dialogue_config(config, source_scenes[0])
                local_scenes = [local_scene]
                local_subjects = normalize_subjects(local_config)
                voice_keys = [speaker_key] if speaker_key in paths.comfy_voice_refs else []
                identity_input = subject_identity_inputs.get(speaker_key)
            else:
                local_config, _, local_subjects, reference_subject_count = scoped_component_context(
                    config, paths, source_scenes, subject_identity_inputs
                )
                local_scenes = source_scenes
                voice_keys = [subject["id"] for subject in local_subjects if subject["id"] in profiles]
                identity_input = (
                    subject_identity_inputs[next(
                        subject["id"] for subject in local_subjects if subject["id"] in subject_identity_inputs
                    )]
                    if reference_subject_count == 1
                    else None
                )
            if speaker_locked:
                reference_subject_count = sum(
                    1 for subject in normalize_subjects(local_config) if "reference_picture" in subject
                )
            audio_label_map = {key: index + 1 for index, key in enumerate(voice_keys)}
            guide_audio = stage_audio_spine(
                config,
                paths,
                key=f"visual_chain_{original_chain_index:02d}_part_{part_number:02d}",
                scene_ids=scene_ids,
                force=force_voice_refs,
            )
            part_indices.append(len(chain_specs))
            chain_specs.append(
                {
                    "scene_ids": scene_ids,
                    "chain_index": original_chain_index,
                    "part_number": part_number,
                    "speaker_locked": speaker_locked,
                    "script": "\n---\n".join(
                        build_prompt(local_config, scene, shot_seconds, audio_label_map) for scene in local_scenes
                    ),
                    "voice_keys": voice_keys,
                    "identity_input": identity_input,
                    "guide_audio": guide_audio,
                    "width": int(video_value(config, "width", 768)),
                    "height": int(video_value(config, "height", 1344)),
                    "frames_per_shot": int(video_value(config, "frames_per_shot", 362)),
                    "fps": int(video_value(config, "fps", 24)),
                    "steps": int(video_value(config, "steps", 14)),
                    "sampler": str(video_value(config, "sampler", "euler")),
                    "scheduler": str(video_value(config, "scheduler", "beta57")),
                    **({
                        key: generation_value
                        for key, generation_value in (
                            ("continuity", generation.get("continuity", "cut")),
                            ("chain_gain_control", generation.get("chain_gain_control", "off")),
                            ("bank_pinned", generation.get("bank_pinned", 1)),
                            ("bank_clip_frames", generation.get("bank_clip_frames", 22)),
                            ("color_level", generation.get("color_level", "off")),
                            ("join_anchor_noise", generation.get("join_anchor_noise", 0.0)),
                            ("join_blend", generation.get("join_blend", False)),
                            ("handoff_release", generation.get("handoff_release", 0.30)),
                            ("bank_ref_noise", generation.get("bank_ref_noise", 0.0)),
                            ("end_anchor", generation.get("end_anchor", False)),
                            ("audio_lock", generation.get("audio_lock", False)),
                            ("handoff_taper", generation.get("handoff_taper", 0)),
                            ("handoff_depth", generation.get("handoff_depth", "block")),
                            ("self_anchor_voice", generation.get("self_anchor_voice", False)),
                            ("master_normalize", generation.get("master_normalize", "luma+contrast")),
                            ("pin_frames", generation.get("pin_frames", "22")),
                            ("pin_noise", generation.get("pin_noise", 0.0)),
                            ("pin_renorm", generation.get("pin_renorm", False)),
                            ("low_ram_master", generation.get("low_ram_master", True)),
                        )
                    }),
                    "reference_subjects": "1,1" if reference_subject_count > 1 else "",
                    "seed": int(config.get("seed", 920260901)) + original_chain_index * 104729 + part_number * 7919,
                }
            )
        chain_part_indices.append(part_indices)
    from visual_workflow import build_visual_workflow

    visual_path = paths.workflows_dir / f"{paths.slug}_comfyui_visual.json"
    workflow_name = slugify(str(config.get("workflow_name") or f"{paths.slug}_Workflow2_可视化"))
    user_path = ROOT / "user" / "default" / "workflows" / "MiniMax-H3-Workflow2" / f"{workflow_name}.json"
    return build_visual_workflow(
        title=str(config["title"]),
        slug=paths.slug,
        config_path=paths.config_path,
        identity_input=paths.comfy_identity_ref,
        voice_inputs=voice_inputs,
        bgm_input=bgm_input,
        chain_specs=chain_specs,
        chain_part_indices=chain_part_indices,
        model_names={"unet": UNET, "clip": CLIP, "video_vae": VIDEO_VAE, "audio_vae": AUDIO_VAE},
        output_path=visual_path,
        user_workflow_path=user_path,
    )


def narration_voice(config: dict) -> dict:
    voice = config.get("narration_voice", {})
    if isinstance(voice, str):
        return {"voice": voice}
    return {"voice": "zh-CN-YunyangNeural", "rate": "+8%", "pitch": "-2Hz", "volume": "+0%", **voice}


async def synthesize_tts(
    text: str,
    voice: str,
    output: Path,
    *,
    rate: str,
    pitch: str,
    volume: str,
    attempts: int = 4,
) -> None:
    """Write Edge TTS output atomically and retry transient network failures."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            temporary.unlink(missing_ok=True)
            await edge_tts.Communicate(
                text, voice, rate=rate, pitch=pitch, volume=volume
            ).save(str(temporary))
            if not temporary.exists() or temporary.stat().st_size < 1024:
                raise RuntimeError(f"Edge TTS returned an empty file: {temporary}")
            temporary.replace(output)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                print(f"TTS RETRY {attempt}/{attempts - 1}: {output.name}", flush=True)
                await asyncio.sleep(float(2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


async def synthesize(config: dict, paths: RuntimePaths, force: bool) -> None:
    paths.narration_dir.mkdir(parents=True, exist_ok=True)
    spec = narration_voice(config)
    for scene in config["scenes"]:
        output = paths.narration_dir / f"scene_{int(scene['id']):02d}.mp3"
        if output.exists() and not force:
            try:
                if media_duration(output) > 2:
                    continue
            except (subprocess.CalledProcessError, ValueError):
                pass
        print(f"NARRATION {int(scene['id']):02d}", flush=True)
        await synthesize_tts(
            str(scene["narration"]), str(spec["voice"]), output,
            rate=str(spec.get("rate", "+8%")), pitch=str(spec.get("pitch", "-2Hz")),
            volume=str(spec.get("volume", "+0%")),
        )


async def synthesize_dialogues(config: dict, paths: RuntimePaths, force: bool) -> None:
    """Always create the designed dialogue track instead of trusting H3 speech."""
    profiles = voice_profiles(config)
    paths.dialogue_fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_voice = narration_voice(config)
    for scene in config["scenes"]:
        dialogue = scene.get("dialogue")
        if not dialogue:
            continue
        key = dialogue_speaker_key(config, dialogue)
        profile = profiles.get(key, fallback_voice)
        output = paths.dialogue_fallback_dir / f"scene_{int(scene['id']):02d}.mp3"
        if output.exists() and not force:
            try:
                if media_duration(output) > 1:
                    continue
            except (subprocess.CalledProcessError, ValueError):
                pass
        print(f"DIALOGUE {int(scene['id']):02d}", flush=True)
        await synthesize_tts(
            str(dialogue["text"]), str(profile.get("voice", "zh-CN-YunxiNeural")), output,
            rate=str(profile.get("fallback_rate", "+20%")), pitch=str(profile.get("pitch", "+0Hz")),
            volume=str(profile.get("volume", "+0%")),
        )


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def subtitle_chunks(text: str, maximum: int = 15) -> list[str]:
    clauses = [part for part in re.split(r"(?<=[，。！？；：、,.!?;:])", text) if part]
    chunks: list[str] = []
    current = ""
    for clause in clauses:
        if current and len(current) + len(clause) > maximum:
            chunks.append(current)
            current = clause
        else:
            current += clause
    if current:
        chunks.append(current)
    return chunks or [text]


def audio_tempo_filters(speed: float) -> str:
    """Return a valid ffmpeg atempo chain for a positive playback speed."""
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors if not math.isclose(factor, 1.0))


def build_audio_plan(config: dict, paths: RuntimePaths, timeline: list[dict[str, float | int]]) -> list[dict[str, Any]]:
    """Build one authoritative timeline for narration, dialogue, and subtitles."""
    scene_by_id = {int(scene["id"]): scene for scene in config["scenes"]}
    total = float(timeline[-1]["end"])
    events: list[dict[str, Any]] = []
    for timing in timeline:
        scene_id = int(timing["scene_id"])
        scene = scene_by_id[scene_id]
        scene_start, scene_end = float(timing["start"]), float(timing["end"])
        dialogue = scene.get("dialogue")
        dialogue_start = scene_end
        dialogue_limit = scene_end
        dialogue_sync_source = ""
        if dialogue:
            if audio_spine_enabled(config, dialogue):
                dialogue_start = scene_start + float(dialogue["start"])
                dialogue_sync_source = "audio_spine"
            else:
                lip_start, lip_end, dialogue_sync_source = dialogue_lip_window(scene, float(timing["duration"]))
                dialogue_start = scene_start + lip_start
                dialogue_limit = min(scene_end - 0.12, scene_start + lip_end)

        narration_source = paths.narration_dir / f"scene_{scene_id:02d}.mp3"
        narration_duration = media_duration(narration_source)
        narration_start = max(scene_start + 0.28, scene_start + float(scene["narration_offset"]))
        narration_limit = min(scene_end - 0.32, dialogue_start - 0.34)
        narration_start = min(narration_start, max(scene_start + 0.28, narration_limit - narration_duration))
        narration_available = narration_limit - narration_start
        if narration_available < 0.8:
            raise RuntimeError(f"Scene {scene_id} has no safe narration window before its dialogue")
        narration_speed = max(1.0, narration_duration / narration_available)
        if narration_speed > 1.45:
            raise RuntimeError(f"Scene {scene_id} narration needs {narration_speed:.2f}x speed; shorten the narration text")
        narration_end = narration_start + narration_duration / narration_speed
        events.append(
            {
                "kind": "narration", "scene_id": scene_id, "source": narration_source,
                "text": str(scene["narration"]), "start": narration_start, "end": narration_end,
                "speed": narration_speed, "lufs": -18,
            }
        )

        if not dialogue:
            continue
        dialogue_source = paths.dialogue_fallback_dir / f"scene_{scene_id:02d}.mp3"
        dialogue_duration = media_duration(dialogue_source)
        if audio_spine_enabled(config, dialogue):
            dialogue_speed = 1.0
            dialogue_limit = dialogue_start + dialogue_duration
            if dialogue_limit > scene_end - 0.12:
                raise RuntimeError(
                    f"Scene {scene_id} audio-spine dialogue exceeds its shot; shorten the line or use a faster dialogue voice"
                )
        else:
            dialogue_available = dialogue_limit - dialogue_start
            if dialogue_available < 0.8:
                raise RuntimeError(f"Scene {scene_id} has no safe dialogue window")
            dialogue_speed = dialogue_duration / dialogue_available
            if not 0.65 <= dialogue_speed <= 1.45:
                raise RuntimeError(
                    f"Scene {scene_id} dialogue needs {dialogue_speed:.2f}x speed to match its visible lip window; "
                    "adjust the line or its lip_sync range"
                )
        dialogue_end_at = dialogue_start + dialogue_duration / dialogue_speed
        events.append(
            {
                "kind": "dialogue", "scene_id": scene_id, "source": dialogue_source,
                "text": str(dialogue["text"]), "speaker": str(dialogue.get("speaker", "")),
                "speaker_id": dialogue_speaker_key(config, dialogue),
                "start": dialogue_start, "end": dialogue_end_at, "speed": dialogue_speed, "lufs": -16,
                "visual_start": dialogue_start, "visual_end": dialogue_limit,
                "sync_source": dialogue_sync_source,
            }
        )
    if not math.isclose(total, float(timeline[-1]["end"])):
        raise AssertionError("invalid audio timeline total")
    return events


def render_speech_track(events: list[dict[str, Any]], total: float, destination: Path) -> None:
    """Render scheduled speech clips into a single 48 kHz stereo track."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    for event in events:
        command.extend(["-i", str(event["source"])])
    filters = [f"[0:a]atrim=0:{total:.6f},asetpts=PTS-STARTPTS[base]"]
    labels = ["[base]"]
    for index, event in enumerate(events, start=1):
        tempo = audio_tempo_filters(float(event["speed"]))
        duration = float(event["end"]) - float(event["start"])
        delay_ms = round(float(event["start"]) * 1000)
        parts = [
            f"[{index}:a]aresample=48000",
            "aformat=sample_fmts=fltp:channel_layouts=stereo",
            f"loudnorm=I={float(event['lufs']):.0f}:TP=-2:LRA=7",
            "aresample=48000",
        ]
        if tempo:
            parts.append(tempo)
        parts.extend([f"atrim=0:{duration:.6f}", f"adelay={delay_ms}:all=1[e{index}]"])
        filters.append(",".join(parts))
        labels.append(f"[e{index}]")
    filters.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:normalize=0,alimiter=limit=0.96[track]")
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[track]", "-t", f"{total:.6f}",
        "-c:a", "pcm_s24le", "-ar", "48000", str(destination),
    ])
    run(command)


def write_subtitles(config: dict, audio_events: list[dict[str, Any]], total: float, path: Path) -> None:
    title = str(config["title"])
    subtitle = str(config.get("subtitle", "AI电影短片"))
    lesson = str(config.get("lesson", ""))
    width, height = int(config.get("output_width", 1080)), int(config.get("output_height", 1920))
    header = f"""[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nScaledBorderAndShadow: yes\nWrapStyle: 0\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Narration,Noto Sans CJK SC,66,&H00FFFFFF,&H000000FF,&H00101010,&H76000000,0,0,0,0,100,100,1,0,1,4.0,0.9,2,64,64,142,1\nStyle: Character,Noto Sans CJK SC,70,&H003FE7FF,&H000000FF,&H00101010,&H76000000,-1,0,0,0,100,100,1,0,1,4.2,1.0,2,64,64,142,1\nStyle: Title,Noto Serif CJK SC,98,&H00F4E4BE,&H000000FF,&H0024190D,&H70000000,-1,0,0,0,100,100,5,0,1,4.5,1.5,5,60,60,0,1\nStyle: Subtitle,Noto Sans CJK SC,44,&H00E8D6AE,&H000000FF,&H00101010,&H70000000,0,0,0,0,100,100,1,0,1,2.5,0.5,5,60,60,0,1\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"""
    subtitle_events = [
        f"Dialogue: 1,{ass_time(1.0)},{ass_time(5.8)},Title,,0,0,0,,{title}",
        f"Dialogue: 1,{ass_time(2.8)},{ass_time(5.8)},Subtitle,,0,0,0,,{subtitle}",
    ]
    for event in audio_events:
        chunks = subtitle_chunks(str(event["text"]))
        total_chars = max(1, sum(len(chunk) for chunk in chunks))
        cursor = float(event["start"])
        event_end = float(event["end"])
        for chunk in chunks:
            duration = (event_end - float(event["start"])) * len(chunk) / total_chars
            end = min(event_end, cursor + duration)
            display_text = chunk
            if event["kind"] == "dialogue" and event.get("speaker"):
                display_text = f"{event['speaker']}：{display_text}"
            safe = display_text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            style = "Narration" if event["kind"] == "narration" else "Character"
            speaker = str(event.get("speaker", ""))
            subtitle_events.append(f"Dialogue: {0 if style == 'Narration' else 2},{ass_time(cursor)},{ass_time(end)},{style},{speaker},0,0,0,,{safe}")
            cursor = end
    if lesson:
        subtitle_events.append(f"Dialogue: 1,{ass_time(max(0, total - 8))},{ass_time(max(0, total - 2))},Title,,0,0,0,,{title}")
        subtitle_events.append(f"Dialogue: 1,{ass_time(max(0, total - 6))},{ass_time(max(0, total - 2))},Subtitle,,0,0,0,,{lesson}")
    path.write_text(header + "\n".join(subtitle_events) + "\n")


def existing_chain_paths(config: dict, paths: RuntimePaths) -> list[Path]:
    """Locate the saved H3 masters that the current release was built from."""
    output_dir = ROOT / "output" / "video" / "workflow2" / paths.slug
    result: list[Path] = []
    shot_seconds = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    for chain_index, scene_ids in enumerate(config["chains"], start=1):
        expected = len(scene_ids) * shot_seconds
        candidates = [paths.chains_dir / f"chain_{chain_index:02d}.mp4"]
        candidates.extend(sorted(output_dir.glob(f"chain_{chain_index:02d}_visual_*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True))
        source = next((item for item in candidates if valid_video(item, expected - 3)), None)
        if source is None:
            raise RuntimeError(f"Could not locate the original H3 master for chain {chain_index:02d}")
        result.append(source.resolve())
    return result


def render_video_segment(
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    """Make a frame-exact, self-contained segment for safe concatenation."""
    if end_frame <= start_frame:
        raise ValueError("video segment must contain at least one frame")
    start_seconds = start_frame / fps
    end_seconds = end_frame / fps
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS[v];"
        f"[0:a]atrim=start={start_seconds:.6f}:end={end_seconds:.6f},asetpts=PTS-STARTPTS[a]"
    )
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-filter_complex", filters, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
        "-r", f"{fps:.6f}", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", str(destination),
    ])


def replace_dialogue_scenes(
    config: dict,
    paths: RuntimePaths,
    scene_videos: dict[int, Path],
    original_chain_paths: list[Path] | None = None,
) -> list[Path]:
    """Replace selected shots without changing any other existing picture.

    The old masters provide all non-dialogue footage.  Each replacement is
    trimmed to the old shot's exact frame range, so the narrative pacing and
    downstream subtitle timeline remain stable.
    """
    original_chain_paths = original_chain_paths or existing_chain_paths(config, paths)
    timeline = build_scene_timeline(config, original_chain_paths)
    timeline_by_id = {int(item["scene_id"]): item for item in timeline}
    unknown = sorted(set(scene_videos) - set(timeline_by_id))
    if unknown:
        raise ValueError(f"Replacement scenes are not in the story timeline: {unknown}")
    fps = float(video_value(config, "fps", 24))
    output_dir = paths.chains_dir / "audio_spine_replacements"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for chain_index, (scene_ids, original) in enumerate(zip(config["chains"], original_chain_paths), start=1):
        chain_timeline = [timeline_by_id[int(scene_id)] for scene_id in scene_ids]
        chain_start_frame = int(chain_timeline[0]["start_frame"])
        segment_paths: list[Path] = []
        for position, timing in enumerate(chain_timeline, start=1):
            scene_id = int(timing["scene_id"])
            frame_count = int(timing["end_frame"]) - int(timing["start_frame"])
            source = scene_videos.get(scene_id, original)
            source_start = 0 if scene_id in scene_videos else int(timing["start_frame"]) - chain_start_frame
            segment = output_dir / f"chain_{chain_index:02d}_scene_{scene_id:02d}.mp4"
            render_video_segment(
                source,
                segment,
                start_frame=source_start,
                end_frame=source_start + frame_count,
                fps=fps,
            )
            segment_paths.append(segment)
        manifest = output_dir / f"chain_{chain_index:02d}.txt"
        manifest.write_text("".join(f"file '{path}'\n" for path in segment_paths))
        destination = output_dir / f"chain_{chain_index:02d}.mp4"
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(manifest), "-c", "copy", "-movflags", "+faststart", str(destination),
        ])
        expected = len(scene_ids) * float(video_value(config, "frames_per_shot", 362)) / fps
        if not valid_video(destination, expected - 3):
            raise RuntimeError(f"Replacement chain {chain_index:02d} failed duration validation: {destination}")
        output_paths.append(destination)
    return output_paths


def render_cover(config: dict, final_video: Path) -> Path | None:
    """Create a title-safe vertical cover from the finished master.

    The platform picker may use a frame where the opening title has already
    faded. Keeping this in the common assembler guarantees every story has a
    deliberate, readable cover instead of relying on an arbitrary frame.
    """
    cover = config.get("cover", {})
    if cover is False:
        return None
    if not isinstance(cover, dict):
        raise ValueError("cover must be an object or false")
    video_width = int(config.get("output_width", 1080))
    # Douyin displays a 3:4 crop for a vertical-video cover.  Generate that
    # crop directly so a title cannot disappear below the platform crop.
    width = int(cover.get("width", video_width))
    height = int(cover.get("height", round(width * 4 / 3)))
    if width <= 0 or height <= 0:
        raise ValueError("cover width and height must be positive")
    title = str(cover.get("title", config["title"])).strip()
    if not title:
        raise ValueError("cover.title cannot be empty")
    font = Path(str(cover.get("fontfile", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")))
    if not font.is_file():
        raise RuntimeError(f"Cover font is unavailable: {font}")
    filename = str(cover.get("output_filename", f"{final_video.stem}_cover.jpg"))
    destination = final_video.parent / filename
    title_filter = title.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    font_filter = str(font).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    title_size = int(cover.get("font_size", max(112, round(width * 0.145))))
    title_y = int(cover.get("title_y", height * 0.15))
    cover_time = max(0.0, float(cover.get("time", 2.0)))
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,"
        f"drawtext=fontfile='{font_filter}':text='{title_filter}':fontcolor=0xF8E8C0:"
        f"fontsize={title_size}:x=(w-text_w)/2:y={title_y}:"
        "borderw=7:bordercolor=black@0.96:shadowcolor=black@0.92:shadowx=3:shadowy=3"
    )
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{cover_time:.3f}",
        "-i", str(final_video), "-frames:v", "1", "-vf", video_filter,
        "-q:v", "2", str(destination),
    ])
    return destination




def assemble(
    config: dict,
    paths: RuntimePaths,
    force_tts: bool,
    chain_paths: list[Path] | None = None,
    bgm_path: Path | None = None,
) -> Path:
    asyncio.run(synthesize(config, paths, force_tts))
    asyncio.run(synthesize_dialogues(config, paths, force_tts))
    paths.final_dir.mkdir(parents=True, exist_ok=True)
    if chain_paths is None:
        chain_paths = [paths.chains_dir / f"chain_{index:02d}.mp4" for index in range(1, len(config["chains"]) + 1)]
    chain_paths = [path.expanduser().resolve() for path in chain_paths]
    chain_paths = normalize_chain_paths(config, paths, chain_paths)
    shot_duration = float(video_value(config, "frames_per_shot", 362)) / float(video_value(config, "fps", 24))
    for index, path in enumerate(chain_paths):
        expected = len(config["chains"][index]) * shot_duration
        if not valid_video(path, expected - 3):
            raise RuntimeError(f"Missing or invalid workflow2 chain: {path}")
    timeline = build_scene_timeline(config, chain_paths)
    total = float(timeline[-1]["end"])
    audio_events = build_audio_plan(config, paths, timeline)
    bgm_source = resolve_bgm_source(config, paths, bgm_path)
    bgm_setting = config.get("bgm", {}) if isinstance(config.get("bgm", {}), dict) else {}
    timeline_path = paths.final_dir / "workflow2_audio_timeline.json"
    timeline_path.write_text(json.dumps({
        "video": timeline,
        "speech": audio_events,
        "bgm": {
            "enabled": bool(bgm_source),
            "source_type": str(bgm_setting.get("source_type", "file")),
            "source": str(bgm_source) if bgm_source else None,
            "sha256": hashlib.sha256(bgm_source.read_bytes()).hexdigest() if bgm_source else None,
            "volume": float(bgm_setting.get("volume", 0.10)),
            "duck_ratio": float(bgm_setting.get("duck_ratio", 18.0)),
            "source_url": bgm_setting.get("source_url"),
            "credit": bgm_setting.get("credit"),
        },
    }, ensure_ascii=False, indent=2, default=str) + "\n")
    concat_list = paths.final_dir / "chains.txt"
    concat_list.write_text("".join(f"file '{path}'\n" for path in chain_paths))
    raw_master = paths.final_dir / "workflow2_native_master_raw.mkv"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(raw_master)])
    clean_master = paths.final_dir / "workflow2_native_master_clean.mp4"
    from remove_h3_speech import clean_video
    # H3 vocal separation is CPU-heavy.  Reuse a clean master when none of
    # the source chains changed; this keeps audio-only mix iterations fast.
    newest_chain_mtime = max(path.stat().st_mtime_ns for path in chain_paths)
    if clean_master.exists() and clean_master.stat().st_mtime_ns >= newest_chain_mtime:
        print(f"CLEAN H3 AUDIO SKIPPED {clean_master}", flush=True)
    else:
        clean_video(raw_master, clean_master)
    narration_track = paths.final_dir / "workflow2_narration_track.wav"
    dialogue_track = paths.final_dir / "workflow2_dialogue_track.wav"
    render_speech_track([event for event in audio_events if event["kind"] == "narration"], total, narration_track)
    render_speech_track([event for event in audio_events if event["kind"] == "dialogue"], total, dialogue_track)
    subtitles = paths.final_dir / "workflow2_vertical.ass"
    write_subtitles(config, audio_events, total, subtitles)
    filename = str(config.get("output_filename", f"{paths.slug}_workflow2_vertical_{int(config.get('output_width', 1080))}x{int(config.get('output_height', 1920))}.mp4"))
    final_video = paths.final_dir / filename
    subtitle_filter = str(subtitles).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    output_width, output_height = int(config.get("output_width", 1080)), int(config.get("output_height", 1920))
    filter_complex = f"[0:v]scale={output_width}:{output_height}:force_original_aspect_ratio=increase:flags=lanczos,crop={output_width}:{output_height},setsar=1,fps={int(video_value(config, 'output_fps', 24))},eq=contrast=1.02:saturation=1.025,ass='{subtitle_filter}'[v];[0:a]aresample=48000:async=1:first_pts=0,volume=1.50,highpass=f=42,lowpass=f=15500,alimiter=limit=0.55[native];[1:a]aresample=48000:async=1:first_pts=0,volume=1.00,highpass=f=75,lowpass=f=12500,acompressor=threshold=0.12:ratio=2.5:attack=8:release=180[narration];[2:a]aresample=48000:async=1:first_pts=0,volume=1.04,highpass=f=75,lowpass=f=12500,acompressor=threshold=0.12:ratio=2.5:attack=8:release=180[dialogue];[narration][dialogue]amix=inputs=2:duration=first:normalize=0,asplit=2[voice_sc][voice_mix]"
    ffmpeg_inputs = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(clean_master), "-i", str(narration_track), "-i", str(dialogue_track)]
    if bgm_source:
        bgm_volume = max(0.0, min(0.35, float(bgm_setting.get("volume", 0.10))))
        fade_in = max(0.0, float(bgm_setting.get("fade_in", 4.0)))
        fade_out = max(0.0, float(bgm_setting.get("fade_out", 8.0)))
        fade_out_start = max(0.0, total - fade_out)
        duck_ratio = max(1.0, min(30.0, float(bgm_setting.get("duck_ratio", 18.0))))
        filter_complex += (
            f";[3:a]aresample=48000:async=1:first_pts=0,aformat=channel_layouts=stereo,"
            f"volume={bgm_volume:.4f},lowpass=f=9000,afade=t=in:st=0:d={fade_in:.3f},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}[bgm];"
            f"[native][bgm]amix=inputs=2:duration=first:normalize=0[bed];"
            f"[bed][voice_sc]sidechaincompress=threshold=0.018:ratio={duck_ratio:.2f}:attack=25:release=650:knee=5[ducked];"
            f"[ducked][voice_mix]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.84[a]"
        )
        ffmpeg_inputs.extend(["-stream_loop", "-1", "-i", str(bgm_source)])
    else:
        filter_complex += ";[native][voice_sc]sidechaincompress=threshold=0.018:ratio=14:attack=25:release=500:knee=4[ducked];[ducked][voice_mix]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.84[a]"
    run(ffmpeg_inputs + ["-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]", "-t", f"{total:.6f}", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(final_video)])
    render_cover(config, final_video)
    return final_video


def verify(path: Path, config: dict) -> None:
    probe = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels", "-of", "json", str(path)], capture_output=True, text=True)
    info = json.loads(probe.stdout)
    duration = float(info["format"]["duration"])
    video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
    minimum = float(config.get("minimum_duration", 120))
    expected_size = (int(config.get("output_width", 1080)), int(config.get("output_height", 1920)))
    if duration < minimum or (video.get("width"), video.get("height")) != expected_size:
        raise RuntimeError(f"Final validation failed: {json.dumps(info, ensure_ascii=False)}")
    run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"VERIFIED duration={duration:.3f}s video={video['width']}x{video['height']}@{video.get('r_frame_rate')} audio={audio.get('codec_name')}/{audio.get('sample_rate')}Hz/{audio.get('channels')}ch", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic MiniMax H3 workflow; visual ComfyUI canvas is the default mode")
    parser.add_argument("--config", "--story", dest="config", type=Path, required=True, help="JSON story/project configuration")
    parser.add_argument("--mode", choices=("visual", "api"), default="visual", help="visual opens a ComfyUI canvas; api runs unattended")
    parser.add_argument("--url", default="http://127.0.0.1:8190")
    parser.add_argument("--no-start-server", action="store_true", help="visual mode: create the canvas without starting ComfyUI")
    parser.add_argument("--no-open-browser", action="store_true", help="visual mode: do not open the ComfyUI page")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-tts", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--chain-path", action="append", type=Path, help="existing H3 chain master; repeat once per configured chain")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--audio-spine-dialogue-scene",
        action="append",
        type=int,
        help="re-render one authored-dialogue scene against its final clean audio; repeat for several scenes",
    )
    parser.add_argument("--skip-ai-audio-qc", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without starting ComfyUI")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        parser.error(f"config does not exist: {config_path}")
    config = json.loads(config_path.read_text())
    validate_config(config)
    validate_story_identity_policy(config, config_path)
    paths = build_paths(config, config_path)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "title": config["title"], "slug": paths.slug, "identity_namespace": identity_namespace(config), "unique_story_cast": bool(identity_policy(config).get("require_unique_story_cast", False)), "scenes": len(config["scenes"]), "chains": len(config["chains"]), "scene_mode": (config.get("generation", {}) or {}).get("scene_mode", "single_shot"), "bgm_source_type": (config.get("bgm", {}) or {}).get("source_type", "file"), "output_dir": str(paths.final_dir), "output_filename": config.get("output_filename", f"{paths.slug}_workflow2_vertical_1080x1920.mp4")}, ensure_ascii=False, indent=2))
        return
    if args.audio_spine_dialogue_scene:
        rendered = generate_audio_spine_dialogue_scenes(
            config,
            paths,
            args.url,
            args.audio_spine_dialogue_scene,
            args.force_generate,
        )
        replacement_chains = replace_dialogue_scenes(config, paths, rendered)
        final_video = assemble(config, paths, args.force_tts, replacement_chains)
        verify(final_video, config)
        print(final_video, flush=True)
        return
    if args.mode == "visual":
        visual_path, user_path = prepare_visual_workflow(config, paths, args.force_tts)
        if not args.no_start_server:
            ensure_server(args.url)
        if not args.no_open_browser:
            webbrowser.open(args.url)
        print(f"COMFYUI VISUAL WORKFLOW {visual_path}", flush=True)
        print(f"COMFYUI WORKFLOW LIBRARY {user_path}", flush=True)
        print(f"COMFYUI PAGE {args.url}", flush=True)
        return
    if args.chain_path:
        if len(args.chain_path) != len(config["chains"]):
            parser.error("--chain-path must be supplied once per configured chain")
        final_video = assemble(config, paths, args.force_tts, args.chain_path)
        verify(final_video, config)
        print(final_video, flush=True)
        return
    if not args.assemble_only:
        generate(config, paths, args.url, args.force_generate)
        if not args.skip_ai_audio_qc:
            from qc_native_audio import run_qc
            report = run_qc(config_path, paths.chains_dir, paths.audio_qc_path)
            print(f"AUDIO QC authored={len(report['authored_dialogue'])} missing={len(report['missing_authored_dialogue'])} unexpected={len(report['confirmed_speech'])}; unexpected speech will be removed during assembly", flush=True)
    if args.generate_only:
        return
    final_video = assemble(config, paths, args.force_tts)
    verify(final_video, config)
    print(final_video, flush=True)


if __name__ == "__main__":
    main()
