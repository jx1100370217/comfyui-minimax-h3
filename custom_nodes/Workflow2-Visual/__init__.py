"""Small visual-workflow helpers for joining low-RAM H3 streamed masters."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path, PurePosixPath

import folder_paths
import numpy as np
from comfy_api.latest import InputImpl


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW2_DIR = ROOT / "story" / "workflow2"
if str(WORKFLOW2_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW2_DIR))


class Workflow2VideoFromPaths:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"path_{letter}": ("STRING", {"forceInput": True})
            for letter in "bcdefgh"
        }
        return {
            "required": {
                "path_a": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "video/workflow2/native_master"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "saved_path")
    FUNCTION = "join"
    CATEGORY = "video/minimax/workflow2"
    DESCRIPTION = "Load one H3 low-RAM streamed master, or safely concatenate up to eight masters, as a previewable ComfyUI VIDEO."

    @staticmethod
    def _safe_source(value: str) -> Path:
        output_root = Path(folder_paths.get_output_directory()).resolve()
        source = Path(value).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"H3 streamed master does not exist: {source}")
        if not source.is_relative_to(output_root):
            raise ValueError(f"Refusing video outside the ComfyUI output directory: {source}")
        return source

    @staticmethod
    def _destination(prefix: str) -> Path:
        output_root = Path(folder_paths.get_output_directory()).resolve()
        relative = PurePosixPath(str(prefix).replace("\\", "/").lstrip("/"))
        if ".." in relative.parts:
            raise ValueError("filename_prefix must stay inside the ComfyUI output directory")
        base = output_root.joinpath(*relative.parts).resolve()
        if not base.is_relative_to(output_root):
            raise ValueError("filename_prefix must stay inside the ComfyUI output directory")
        base.parent.mkdir(parents=True, exist_ok=True)
        for counter in range(1, 100000):
            candidate = base.parent / f"{base.name}_{counter:05d}.mp4"
            if not candidate.exists():
                return candidate
        raise RuntimeError("No free output filename was found")

    @staticmethod
    def _concat(sources: list[Path], destination: Path) -> None:
        if len(sources) == 1:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(sources[0]), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", str(destination)],
                check=True,
            )
            return
        with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="workflow2_concat_", delete=False) as handle:
            manifest = Path(handle.name)
            for source in sources:
                escaped = str(source).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        try:
            command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", str(destination)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(destination)],
                    check=True,
                )
        finally:
            try:
                os.unlink(manifest)
            except FileNotFoundError:
                pass

    def join(self, path_a, filename_prefix, **kwargs):
        raw_paths = [path_a] + [kwargs.get(f"path_{letter}") for letter in "bcdefgh"]
        sources = [self._safe_source(value) for value in raw_paths if value]
        destination = self._destination(filename_prefix)
        self._concat(sources, destination)
        return (InputImpl.VideoFromFile(str(destination)), str(destination))

    @classmethod
    def IS_CHANGED(cls, path_a, filename_prefix, **kwargs):
        values = [path_a] + [kwargs.get(f"path_{letter}") for letter in "bcdefgh"]
        stamps = []
        for value in values:
            if value and Path(value).exists():
                stamps.append((str(value), Path(value).stat().st_mtime_ns))
        return (tuple(stamps), filename_prefix)


class Workflow2StoryAssembler:
    """Create the release video from visual H3 chain masters and a story config."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"path_{letter}": ("STRING", {"forceInput": True})
            for letter in "bcdefgh"
        }
        optional["bgm"] = ("AUDIO",)
        return {
            "required": {
                "config_path": ("STRING", {"default": "story/workflow2/stories/story.json"}),
                "path_a": ("STRING", {"forceInput": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "saved_path")
    FUNCTION = "assemble"
    CATEGORY = "video/minimax/workflow2"
    OUTPUT_NODE = True
    DESCRIPTION = "Uses actual H3 video frames; Audio Spine stories drive H3 lip motion from the final dialogue waveform, while legacy dialogue.lip_sync windows remain supported for existing clips."

    @staticmethod
    def _safe_config(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.is_file() or not candidate.is_relative_to(ROOT):
            raise ValueError("story config must be a file inside the ComfyUI project")
        return candidate

    @staticmethod
    def _write_audio_input(audio, destination: Path) -> Path:
        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = int(audio.get("sample_rate", 48000)) if isinstance(audio, dict) else 48000
        if waveform is None:
            raise ValueError("BGM AUDIO input is empty")
        if getattr(waveform, "ndim", 0) == 3:
            waveform = waveform[0]
        if getattr(waveform, "ndim", 0) == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform[:2].detach().cpu().clamp(-1.0, 1.0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        pcm = (waveform.transpose(0, 1).contiguous().numpy() * 32767.0).astype("<i2").tobytes()
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(int(waveform.shape[0]))
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm)
        return destination

    def assemble(self, config_path, path_a, **kwargs):
        from run_workflow2 import assemble, build_paths, validate_config, verify

        config_file = self._safe_config(config_path)
        raw_paths = [path_a] + [kwargs.get(f"path_{letter}") for letter in "bcdefgh"]
        sources = [Workflow2VideoFromPaths._safe_source(value) for value in raw_paths if value]
        import json

        config = json.loads(config_file.read_text())
        validate_config(config)
        if len(sources) != len(config["chains"]):
            raise ValueError("the number of H3 chain paths must match story config.chains")
        paths = build_paths(config, config_file)
        bgm_audio = kwargs.get("bgm")
        bgm_path = None
        if bgm_audio is not None:
            paths.final_dir.mkdir(parents=True, exist_ok=True)
            handle, temporary_name = tempfile.mkstemp(prefix=".workflow2_bgm_", suffix=".wav", dir=paths.final_dir)
            os.close(handle)
            bgm_path = self._write_audio_input(bgm_audio, Path(temporary_name))
        try:
            final_path = assemble(config, paths, force_tts=False, chain_paths=sources, bgm_path=bgm_path)
        finally:
            if bgm_path:
                bgm_path.unlink(missing_ok=True)
        verify(final_path, config)
        return (InputImpl.VideoFromFile(str(final_path)), str(final_path))

    @classmethod
    def IS_CHANGED(cls, config_path, path_a, **kwargs):
        values = [config_path, path_a] + [kwargs.get(f"path_{letter}") for letter in "bcdefgh"]
        stamps = []
        for value in values:
            if value and Path(value).exists():
                stamps.append((str(value), Path(value).stat().st_mtime_ns))
        return tuple(stamps)


NODE_CLASS_MAPPINGS = {
    "Workflow2VideoFromPaths": Workflow2VideoFromPaths,
    "Workflow2StoryAssembler": Workflow2StoryAssembler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Workflow2VideoFromPaths": "Workflow2 · 合并/预览 H3 流式视频",
    "Workflow2StoryAssembler": "Workflow2 · 精准旁白/对白成片",
}
