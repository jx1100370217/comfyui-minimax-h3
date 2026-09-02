#!/usr/bin/env python3
"""Remove hallucinated speech from MiniMax H3 audio while retaining diegetic SFX."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def separate_non_vocals(
    waveform: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: int,
    segment_seconds: float = 8.0,
    overlap_seconds: float = 1.0,
    preserve_windows: Iterable[tuple[float, float]] = (),
) -> torch.Tensor:
    segment = int(round(segment_seconds * sample_rate))
    overlap = int(round(overlap_seconds * sample_rate))
    step = segment - overlap
    length = waveform.shape[-1]
    output = torch.zeros_like(waveform)
    weight = torch.zeros(length, dtype=waveform.dtype)

    for start in range(0, length, step):
        stop = min(start + segment, length)
        valid = stop - start
        chunk = waveform[:, start:stop]
        if valid < segment:
            chunk = F.pad(chunk, (0, segment - valid))

        reference = chunk.mean(dim=0)
        mean = reference.mean()
        std = reference.std().clamp_min(1e-5)
        normalized = (chunk - mean) / std
        with torch.inference_mode():
            sources = model(normalized.unsqueeze(0))[0]
        sources = sources * std + mean
        non_vocals = sources[:3].sum(dim=0)[:, :valid]

        window = torch.ones(valid, dtype=waveform.dtype)
        fade = min(overlap, valid // 2)
        if fade:
            if start > 0:
                window[:fade] = torch.linspace(0.0, 1.0, fade)
            if stop < length:
                window[-fade:] = torch.linspace(1.0, 0.0, fade)
        output[:, start:stop] += non_vocals * window
        weight[start:stop] += window
        if stop == length:
            break

    output /= weight.clamp_min(1e-6).unsqueeze(0)
    original_rms = waveform.square().mean().sqrt().item()
    clean_rms = output.square().mean().sqrt().item()
    peak = output.abs().max().item()
    if clean_rms > 1e-7 and peak > 1e-7:
        target_rms = original_rms * 0.75
        gain = min(target_rms / clean_rms, 0.70 / peak)
        output *= gain
    preserve = torch.zeros(length, dtype=waveform.dtype)
    fade = max(1, round(sample_rate * 0.12))
    for start_seconds, end_seconds in preserve_windows:
        start = max(0, min(length, round(start_seconds * sample_rate)))
        end = max(start, min(length, round(end_seconds * sample_rate)))
        left = max(0, start - fade)
        right = min(length, end + fade)
        if start > left:
            preserve[left:start] = torch.maximum(
                preserve[left:start], torch.linspace(0.0, 1.0, start - left)
            )
        preserve[start:end] = 1.0
        if right > end:
            preserve[end:right] = torch.maximum(
                preserve[end:right], torch.linspace(1.0, 0.0, right - end)
            )
    if preserve.any():
        mask = preserve.unsqueeze(0)
        output = output * (1.0 - mask) + waveform * mask
    return output


def clean_video(
    input_video: Path,
    output_video: Path,
    preserve_windows: Iterable[tuple[float, float]] = (),
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    with tempfile.TemporaryDirectory(prefix="h3_speech_clean_") as temp_dir:
        temp = Path(temp_dir)
        original_wav = temp / "original.wav"
        clean_wav = temp / "clean.wav"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_video),
                "-vn",
                "-ac",
                "2",
                "-ar",
                str(HDEMUCS_HIGH_MUSDB_PLUS.sample_rate),
                "-c:a",
                "pcm_f32le",
                str(original_wav),
            ]
        )

        samples, sample_rate = sf.read(original_wav, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())
        model = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().cpu()
        clean = separate_non_vocals(
            waveform, model, sample_rate, preserve_windows=preserve_windows
        )
        sf.write(clean_wav, clean.T.numpy(), sample_rate, subtype="PCM_24")

        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_video),
                "-i",
                str(clean_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(output_video),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument(
        "--preserve-window",
        action="append",
        default=[],
        metavar="START:END",
        help="Keep the original audio in this authored dialogue time window; repeatable.",
    )
    args = parser.parse_args()
    windows = []
    for value in args.preserve_window:
        start, end = value.split(":", 1)
        windows.append((float(start), float(end)))
    clean_video(args.input_video.resolve(), args.output_video.resolve(), windows)


if __name__ == "__main__":
    main()
