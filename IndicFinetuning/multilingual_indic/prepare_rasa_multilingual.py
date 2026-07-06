import argparse
import csv
import json
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_text import normalize_indic_text


SELECTED_LANGUAGES = ["hi", "ta", "te", "ml", "kn", "bn", "mr", "gu", "pa", "ur", "or", "as"]
LANGUAGE_NAMES = {
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "ml": "malayalam",
    "kn": "kannada",
    "bn": "bengali",
    "mr": "marathi",
    "gu": "gujarati",
    "pa": "punjabi",
    "ur": "urdu",
    "or": "odia",
    "as": "assamese",
}


def nested_value(item: dict, key: Optional[str]) -> Any:
    if not key:
        return None
    value = item
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def pick_column(columns: list[str], explicit: Optional[str], candidates: list[str], role: str, required: bool = True) -> Optional[str]:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"{role} column '{explicit}' not found. Available columns: {columns}")
        return explicit
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise ValueError(f"Could not infer {role} column. Available columns: {columns}")
    return None


def get_audio_array_and_sr(audio_value):
    if isinstance(audio_value, dict):
        array = audio_value.get("array")
        sampling_rate = audio_value.get("sampling_rate")
        if array is not None and sampling_rate is not None:
            return np.asarray(array), int(sampling_rate)
        audio_bytes = audio_value.get("bytes")
        if audio_bytes:
            array, sampling_rate = sf.read(BytesIO(audio_bytes), always_2d=False)
            return np.asarray(array), int(sampling_rate)
        path = audio_value.get("path")
        if path:
            array, sampling_rate = sf.read(path, always_2d=False)
            return np.asarray(array), int(sampling_rate)
    if isinstance(audio_value, str):
        array, sampling_rate = sf.read(audio_value, always_2d=False)
        return np.asarray(array), int(sampling_rate)
    raise ValueError("Unsupported audio value. Expected HF Audio dict, audio bytes, or audio file path.")


def normalize_gender(value: Any) -> str:
    gender = str(value or "").strip().lower()
    if gender.startswith("f") or gender in {"woman", "female_speaker"}:
        return "female"
    if gender.startswith("m") or gender in {"man", "male_speaker"}:
        return "male"
    return "unknown"


def parse_config_map(value: Optional[str]) -> dict[str, str]:
    if not value:
        return {}
    mapping = {}
    for part in value.split(","):
        if not part.strip():
            continue
        language, config = part.split("=", 1)
        mapping[language.strip()] = config.strip()
    return mapping


def choose_config(language: str, config_names: list[str], config_map: dict[str, str]) -> Optional[str]:
    if language in config_map:
        return config_map[language]
    if not config_names:
        return None
    if len(config_names) == 1 and config_names[0].lower() in {"default", "main", "all"}:
        return config_names[0]

    language_name = LANGUAGE_NAMES[language]
    normalized = {config.lower(): config for config in config_names}
    for key in (language, language_name):
        if key in normalized:
            return normalized[key]

    matches = [
        config
        for config in config_names
        if language in config.lower().split("_") or language_name in config.lower()
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Could not pick Rasa config for language '{language}'. "
        f"Pass --config-map {language}=CONFIG. Available configs: {config_names[:50]}"
    )


def load_language_dataset(args, language: str, config_name: Optional[str]):
    from datasets import Audio, load_dataset

    kwargs = {
        "split": args.split,
        "streaming": args.streaming,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.hf_token:
        kwargs["token"] = args.hf_token

    dataset = load_dataset(args.dataset, config_name, **kwargs) if config_name else load_dataset(args.dataset, **kwargs)
    columns = list(dataset.features.keys()) if hasattr(dataset, "features") and dataset.features else list(dataset.column_names)
    audio_column = pick_column(columns, args.audio_column, ["audio", "wav", "speech", "audio_filepath", "path"], "audio")
    text_column = pick_column(
        columns,
        args.text_column,
        ["normalized_text", "text", "transcript", "sentence", "raw_text", "verbatim_text"],
        "text",
    )
    gender_column = pick_column(columns, args.gender_column, ["gender", "speaker_gender", "sex"], "gender", required=False)
    language_column = pick_column(columns, args.language_column, ["language_id", "lang", "language"], "language", required=False)

    try:
        dataset = dataset.cast_column(audio_column, Audio(decode=False))
    except Exception:
        pass

    return dataset, audio_column, text_column, gender_column, language_column


def export_language(args, language: str, writer, wav_dir: Path, config_name: Optional[str]) -> dict:
    dataset, audio_column, text_column, gender_column, language_column = load_language_dataset(args, language, config_name)
    target_seconds = args.hours_per_language * 3600.0
    gender_targets = {"female": target_seconds / 2.0, "male": target_seconds / 2.0}
    seconds_by_gender = {"female": 0.0, "male": 0.0, "unknown": 0.0}
    exported = 0
    skipped = 0

    print(
        f"[{language}] columns: audio={audio_column}, text={text_column}, "
        f"gender={gender_column or 'none'}, language={language_column or 'none'}",
        flush=True,
    )

    if args.streaming:
        rows = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    else:
        rows = dataset.shuffle(seed=args.seed)

    for index, item in enumerate(rows):
        if index and args.progress_every and index % args.progress_every == 0:
            total_hours = sum(seconds_by_gender.values()) / 3600.0
            print(
                f"[{language}] scanned={index} exported={exported} skipped={skipped} "
                f"hours={total_hours:.2f}/{args.hours_per_language:.2f}",
                flush=True,
            )

        if language_column:
            item_language = str(nested_value(item, language_column) or language).strip().lower()
            if item_language and item_language not in {language, LANGUAGE_NAMES[language]}:
                skipped += 1
                continue

        gender = normalize_gender(nested_value(item, gender_column)) if gender_column else "unknown"
        if gender in gender_targets and seconds_by_gender[gender] >= gender_targets[gender]:
            if all(seconds_by_gender[key] >= value for key, value in gender_targets.items()):
                break
            continue
        if gender == "unknown" and seconds_by_gender["unknown"] >= target_seconds:
            break

        text = normalize_indic_text(nested_value(item, text_column) or "")
        if not text:
            skipped += 1
            continue

        try:
            audio_array, sampling_rate = get_audio_array_and_sr(nested_value(item, audio_column))
            audio_array = np.asarray(audio_array, dtype=np.float32)
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1 if audio_array.shape[0] > audio_array.shape[1] else 0)
            duration = float(audio_array.shape[0]) / float(sampling_rate)
            if duration < args.min_duration or duration > args.max_duration:
                skipped += 1
                continue
        except Exception as exc:
            skipped += 1
            print(f"[{language}] skipping row {index}: audio decode failed: {exc}")
            continue

        file_id = f"{language}_{gender}_{exported:06d}"
        sf.write(str(wav_dir / f"{file_id}.wav"), audio_array, sampling_rate)
        writer.writerow([file_id, text, text, language])
        seconds_by_gender[gender] += duration
        exported += 1

        if args.max_samples_per_language and exported >= args.max_samples_per_language:
            break
        if gender_column:
            if all(seconds_by_gender[key] >= value for key, value in gender_targets.items()):
                break
        elif seconds_by_gender["unknown"] >= target_seconds:
            break

    total_hours = sum(seconds_by_gender.values()) / 3600.0
    print(
        f"[{language}] done exported={exported} skipped={skipped} "
        f"hours={total_hours:.2f}/{args.hours_per_language:.2f}",
        flush=True,
    )

    return {
        "language": language,
        "config": config_name,
        "exported": exported,
        "skipped": skipped,
        "hours": {key: round(value / 3600.0, 3) for key, value in seconds_by_gender.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare a balanced multilingual Rasa dataset for Indic Chatterbox finetuning.")
    parser.add_argument("--dataset", default="ai4bharat/Rasa")
    parser.add_argument("--output", required=True)
    parser.add_argument("--languages", nargs="+", default=SELECTED_LANGUAGES)
    parser.add_argument("--hours-per-language", type=float, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--config-map", default=None, help="Comma map if config auto-detect fails, e.g. hi=hindi,ta=tamil")
    parser.add_argument("--audio-column", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--gender-column", default=None)
    parser.add_argument("--language-column", default=None)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--max-samples-per-language", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=256, help="Shuffle buffer for streaming mode.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress after scanning this many rows per language.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from datasets import get_dataset_config_names

    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    summary_path = output_dir / "summary.json"

    try:
        config_names = get_dataset_config_names(
            args.dataset,
            token=args.hf_token,
            trust_remote_code=args.trust_remote_code,
        )
    except TypeError:
        # Some datasets versions do not accept trust_remote_code here, even
        # though load_dataset accepts it. Keep the loader path configurable.
        config_names = get_dataset_config_names(args.dataset, token=args.hf_token)
    config_map = parse_config_map(args.config_map)
    summary = []

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        for language in args.languages:
            config_name = choose_config(language, config_names, config_map)
            print(f"Preparing {language} from config={config_name}", flush=True)
            summary.append(export_language(args, language, writer, wav_dir, config_name))

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadata: {metadata_path}")
    print(f"Wavs: {wav_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
