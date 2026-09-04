#!/usr/bin/env python3
"""Render a small, license-free cinematic pentatonic music bed.

This is deliberately an offline fallback for stories that do not have a
licensed music file.  It creates a real musical arrangement (tempo, chord
changes, melody, plucks and restrained frame-drum pulses), not a single-tone
drone.  The generic Workflow2 runner only needs the resulting audio file.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 48_000
PENTATONIC = np.array([293.6648, 349.2282, 391.9954, 440.0000, 523.2511], dtype=np.float64)
PAD_CHORDS = (
    (146.8324, 220.0000, 293.6648),
    (130.8128, 196.0000, 261.6256),
    (116.5409, 174.6141, 233.0819),
    (97.9989, 146.8324, 195.9977),
)

# Small, deliberately different score profiles keep the offline fallback from
# producing the same loop for every story.  They share the same safe mixer but
# vary register, tempo, pulse density, and the lead/pluck balance.
PROFILES = {
    "ye_gong": {"tempo": 72.0, "scale": (261.626, 329.628, 392.000, 493.883, 587.330), "lead": 0.19, "pluck": 0.14, "drum": 0.06, "octave": 1.0},
    "nan_yuan": {"tempo": 86.0, "scale": (293.665, 369.994, 440.000, 554.365, 659.255), "lead": 0.13, "pluck": 0.23, "drum": 0.15, "octave": 1.0},
    "cup_snake": {"tempo": 58.0, "scale": (246.942, 293.665, 369.994, 440.000, 554.365), "lead": 0.15, "pluck": 0.12, "drum": 0.035, "octave": 0.5},
    "paoding": {"tempo": 78.0, "scale": (261.626, 311.127, 392.000, 466.164, 622.254), "lead": 0.14, "pluck": 0.20, "drum": 0.11, "octave": 1.0},
    "zhuang_zhou": {"tempo": 62.0, "scale": (277.183, 329.628, 415.305, 493.883, 622.254), "lead": 0.22, "pluck": 0.10, "drum": 0.025, "octave": 2.0},
}


def _pan(voice: np.ndarray, position: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-power stereo placement."""
    angle = (position + 1.0) * np.pi / 4.0
    return voice * np.cos(angle), voice * np.sin(angle)


def _add(buffer: np.ndarray, start: float, voice: np.ndarray, position: float, gain: float) -> None:
    first = max(0, int(round(start * SAMPLE_RATE)))
    last = min(buffer.shape[1], first + len(voice))
    if last <= first:
        return
    left, right = _pan(voice[: last - first] * gain, position)
    buffer[0, first:last] += left
    buffer[1, first:last] += right


def _envelope(length: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(length, dtype=np.float64)
    attack_n = min(length, max(1, int(attack * SAMPLE_RATE)))
    release_n = min(length, max(1, int(release * SAMPLE_RATE)))
    env[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False)
    env[-release_n:] *= np.linspace(1.0, 0.0, release_n)
    return env


def _flute(freq: float, seconds: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(max(1, int(seconds * SAMPLE_RATE)), dtype=np.float64) / SAMPLE_RATE
    vibrato = 0.012 * np.sin(2 * np.pi * 5.1 * t)
    phase = 2 * np.pi * freq * (1.0 + vibrato) * t
    voice = 0.72 * np.sin(phase) + 0.20 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
    breath = rng.normal(0.0, 0.012, len(t))
    return (voice + breath) * _envelope(len(t), 0.055, min(0.26, seconds * 0.45))


def _pluck(freq: float, seconds: float = 1.25) -> np.ndarray:
    t = np.arange(max(1, int(seconds * SAMPLE_RATE)), dtype=np.float64) / SAMPLE_RATE
    decay = np.exp(-3.8 * t)
    voice = np.sin(2 * np.pi * freq * t) + 0.34 * np.sin(4 * np.pi * freq * t) + 0.12 * np.sin(6 * np.pi * freq * t)
    return voice * decay


def _drum(seconds: float = 0.32) -> np.ndarray:
    t = np.arange(max(1, int(seconds * SAMPLE_RATE)), dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(77)
    body = np.sin(2 * np.pi * 82.0 * t) * np.exp(-12.0 * t)
    skin = rng.normal(0.0, 1.0, len(t)) * np.exp(-24.0 * t)
    return 0.78 * body + 0.22 * skin


def render(duration: float, seed: int = 920261117, profile: str = "ye_gong") -> np.ndarray:
    total = max(1, int(round(duration * SAMPLE_RATE)))
    audio = np.zeros((2, total), dtype=np.float64)
    spec = PROFILES.get(profile, PROFILES["ye_gong"])
    tempo = float(spec["tempo"])
    scale = np.asarray(spec["scale"], dtype=np.float64)
    beat = 60.0 / tempo
    bar = 4.0 * beat
    rng = np.random.default_rng(seed)
    bar_count = int(np.ceil(duration / bar))

    # Four-bar movement gives the underscore a clear beginning, development,
    # and return while remaining calm enough to sit under spoken narration.
    for bar_index in range(bar_count):
        chord = PAD_CHORDS[bar_index // 4 % len(PAD_CHORDS)]
        start = bar_index * bar
        remaining = min(4.0 * bar, duration - start)
        if remaining <= 0:
            break
        for note_index, freq in enumerate(chord):
            t = np.arange(int(remaining * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
            slow = 0.76 + 0.24 * np.sin(2 * np.pi * t / (bar * 2.0) + note_index)
            pad = (0.72 * np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)) * slow
            pad *= _envelope(len(t), 1.0, 1.2)
            _add(audio, start, pad, -0.30 + note_index * 0.30, 0.12)

        # Guqin-like plucks mark the pulse without competing with speech.
        for beat_index in (0, 2):
            degree = (bar_index + beat_index // 2) % len(PENTATONIC)
            _add(audio, start + beat_index * beat, _pluck(scale[degree] / 2.0), -0.45 if beat_index == 0 else 0.45, float(spec["pluck"]))

        # A restrained frame-drum pulse enters after the intro and returns at
        # the close, so the arrangement has movement without becoming busy.
        if 7 <= bar_index < bar_count - 3:
            _add(audio, start, _drum(), 0.0, float(spec["drum"]))
            if bar_index % 2 == 1:
                _add(audio, start + 2 * beat, _drum(0.20), 0.10, float(spec["drum"]) * 0.5)

        # A singable pentatonic motif.  The middle section lifts one octave;
        # the final section simplifies it again for a calm resolution.
        motif = ((0, 0.00, 0.72), (1, 0.90, 0.46), (2, 1.50, 0.56), (3, 2.18, 0.80), (2, 3.12, 0.48))
        if bar_index % 4 == 3:
            motif = ((4, 0.00, 0.60), (3, 0.82, 0.44), (2, 1.44, 0.58), (1, 2.15, 0.68), (0, 3.05, 0.70))
        if bar_index < 4 and bar_index % 2:
            motif = motif[::2]
        octave = float(spec["octave"]) * (2.0 if 20 <= bar_index < 32 else 1.0)
        for degree, beat_offset, note_beats in motif:
            note_start = start + beat_offset * beat
            if note_start >= duration:
                continue
            seconds = min(note_beats * beat, duration - note_start)
            _add(audio, note_start, _flute(scale[degree] * octave, seconds, rng), 0.18 if degree % 2 else -0.18, float(spec["lead"]) if bar_index < 4 else float(spec["lead"]) * 1.35)

    # Short room reflections make the synthetic instruments feel like a score,
    # while the final Workflow2 mixer still handles speech ducking.
    dry = audio.copy()
    for delay, gain in ((0.31, 0.16), (0.57, 0.10), (0.91, 0.055)):
        samples = int(delay * SAMPLE_RATE)
        if samples < total:
            audio[0, samples:] += dry[1, : total - samples] * gain
            audio[1, samples:] += dry[0, : total - samples] * gain
    fade = min(total // 2, int(3.0 * SAMPLE_RATE))
    audio[:, :fade] *= np.linspace(0.0, 1.0, fade)
    audio[:, -fade:] *= np.linspace(1.0, 0.0, fade)
    peak = float(np.max(np.abs(audio))) or 1.0
    return np.clip(audio / peak * 0.58, -0.95, 0.95)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio.T * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an original cinematic pentatonic BGM")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--seed", type=int, default=920261117)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="ye_gong")
    args = parser.parse_args()
    write_wav(args.output.expanduser().resolve(), render(args.duration, args.seed, args.profile))


if __name__ == "__main__":
    main()
