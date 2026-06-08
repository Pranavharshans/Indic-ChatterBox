import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.generate_openrouter_pilot_dataset import (  # noqa: E402
    DEFAULT_MODEL,
    maybe_normalize_wav,
    request_speech,
    write_pcm_wav,
)


DEFAULT_DATASET = "./IndicFinetuning/datasets/OpenRouterManualReview800"


def load_manifest(dataset_dir: Path):
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def selected_range(total: int, start_index: int, limit: int | None):
    start = max(start_index, 0)
    end = total if limit is None else min(total, start + max(limit, 0))
    return range(start, end)


def save_audio_bytes(wav_path: Path, audio_bytes: bytes, response_format: str):
    if response_format == "pcm":
        write_pcm_wav(wav_path, audio_bytes)
    else:
        mp3_path = wav_path.with_suffix(".mp3")
        mp3_path.write_bytes(audio_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp3_path), "-ar", "24000", "-ac", "1", str(wav_path)],
            check=True,
        )
        mp3_path.unlink(missing_ok=True)
    maybe_normalize_wav(wav_path)


def main():
    parser = argparse.ArgumentParser(description="Generate OpenRouter TTS audio for the manual Malayalam dataset manifest.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--response-format", choices=["pcm", "mp3"], default="pcm")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY before generating audio.")

    dataset_dir = Path(args.dataset)
    rows = load_manifest(dataset_dir)
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    indexes = list(selected_range(len(rows), args.start_index, args.limit))
    for position, index in enumerate(indexes, start=1):
        row = rows[index]
        wav_path = wav_dir / f"{row['id']}.wav"
        if args.skip_existing and wav_path.exists() and wav_path.stat().st_size > 0:
            print(f"[{position}/{len(indexes)}] skip existing {row['id']}")
            continue
        print(
            f"[{position}/{len(indexes)}] index={index} id={row['id']} "
            f"voice={row['voice']} gender={row['gender']} category={row['category']} emotion={row['emotion_type'] or '-'}"
        )
        audio_bytes = request_speech(api_key, args.model, row["voice"], row["tts_input"], args.response_format)
        save_audio_bytes(wav_path, audio_bytes, args.response_format)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Done. WAVs: {wav_dir}")


if __name__ == "__main__":
    main()
