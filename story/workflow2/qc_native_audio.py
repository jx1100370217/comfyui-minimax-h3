#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import pipeline


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parent

CLASSIFIER_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
ASR_MODEL = "openai/whisper-tiny"

SPEECH_LABELS = ("speech", "conversation", "narration", "whispering", "talking")
MUSIC_LABELS = ("music", "musical instrument", "singing", "choir", "orchestra")
COMMON_SILENCE_HALLUCINATIONS = {
    "thankyou", "thanksforwatching", "subtitlesby", "you", "谢谢", "感谢观看"
}


def label_score(results: list[dict], needles: tuple[str, ...]) -> float:
    return sum(
        float(item["score"])
        for item in results
        if any(needle in item["label"].lower() for needle in needles)
    )


def normalized_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text).lower()


def extract_audio(source: Path, start: float, duration: float, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.6f}", "-t", f"{duration:.6f}",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )


def run_qc(
    config_path: Path,
    chains_dir: Path,
    report_path: Path | None = None,
) -> dict:
    config = json.loads(config_path.read_text())
    report_path = report_path or config_path.with_name("audio_qc.json")
    video = config.get("video", {})
    shot_duration = video.get("frames_per_shot", config.get("frames_per_shot", 362)) / video.get("fps", config.get("fps", 24))
    scene_to_chain = {}
    for chain_index, scene_ids in enumerate(config["chains"], start=1):
        for within_chain, scene_id in enumerate(scene_ids):
            scene_to_chain[int(scene_id)] = (chain_index, within_chain)
    classifier = pipeline("audio-classification", model=CLASSIFIER_MODEL, device=-1)
    asr = None
    report: dict = {
        "model": CLASSIFIER_MODEL,
        "scenes": [],
        "confirmed_speech": [],
        "authored_dialogue": [],
        "missing_authored_dialogue": [],
    }

    with tempfile.TemporaryDirectory(prefix="workflow2_audio_qc_") as temporary:
        temporary_dir = Path(temporary)
        for scene_index, scene in enumerate(config["scenes"]):
            chain_index, within_chain = scene_to_chain[int(scene["id"])]
            chain_path = chains_dir / f"chain_{chain_index:02d}.mp4"
            scene_result = {"scene_id": scene["id"], "windows": []}
            dialogue = scene.get("dialogue")
            dialogue_detected = False
            for half in range(2):
                local_start = half * shot_duration / 2
                local_end = local_start + shot_duration / 2
                start = within_chain * shot_duration + local_start
                duration = shot_duration / 2
                wav_path = temporary_dir / f"scene_{scene['id']:02d}_{half}.wav"
                extract_audio(chain_path, start, duration, wav_path)
                waveform, sample_rate = sf.read(wav_path, dtype="float32")
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                result = classifier(
                    {"raw": np.asarray(waveform), "sampling_rate": sample_rate},
                    top_k=20,
                )
                speech_score = label_score(result, SPEECH_LABELS)
                music_score = label_score(result, MUSIC_LABELS)
                entry = {
                    "half": half + 1,
                    "speech_score": round(speech_score, 4),
                    "music_score": round(music_score, 4),
                    "top_events": [
                        {"label": item["label"], "score": round(float(item["score"]), 4)}
                        for item in result[:8]
                    ],
                }
                if speech_score >= 0.35:
                    if asr is None:
                        asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=-1)
                    transcript = asr(
                        {"raw": np.asarray(waveform), "sampling_rate": sample_rate},
                        generate_kwargs={"task": "transcribe"},
                    )["text"].strip()
                    entry["transcript"] = transcript
                    normalized = normalized_text(transcript)
                    expected_dialogue = bool(
                        dialogue
                        and local_end >= dialogue["start"]
                        and local_start <= dialogue["start"] + 3.2
                    )
                    if expected_dialogue:
                        dialogue_detected = True
                        entry["expected_text"] = dialogue["text"]
                        report["authored_dialogue"].append(
                            {
                                "scene_id": scene["id"],
                                "expected": dialogue["text"],
                                "transcript": transcript,
                            }
                        )
                    elif len(normalized) >= 3 and normalized not in COMMON_SILENCE_HALLUCINATIONS:
                        report["confirmed_speech"].append(
                            {"scene_id": scene["id"], "half": half + 1, "text": transcript}
                        )
                scene_result["windows"].append(entry)
            if dialogue and not dialogue_detected:
                report["missing_authored_dialogue"].append(
                    {"scene_id": scene["id"], "expected": dialogue["text"]}
                )
            report["scenes"].append(scene_result)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect unwanted speech/music in raw H3 audio")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chains-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_qc(args.config, args.chains_dir, args.report)
    print(args.report or args.config.with_name("audio_qc.json"))
    if report["confirmed_speech"]:
        print(json.dumps(report["confirmed_speech"], ensure_ascii=False, indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
