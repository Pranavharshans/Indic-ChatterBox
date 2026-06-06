import argparse
import csv
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from IndicFinetuning.indic_text import normalize_indic_text, resolve_language


def get_nested_value(item: dict, key: Optional[str]) -> Any:
    if not key:
        return None
    value = item
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def get_audio_array_and_sr(audio_value):
    if isinstance(audio_value, dict):
        array = audio_value.get("array")
        sampling_rate = audio_value.get("sampling_rate")
        if array is not None and sampling_rate is not None:
            return np.asarray(array), int(sampling_rate)
        path = audio_value.get("path")
        if path:
            array, sampling_rate = sf.read(path, always_2d=False)
            return np.asarray(array), int(sampling_rate)
    if isinstance(audio_value, str):
        array, sampling_rate = sf.read(audio_value, always_2d=False)
        return np.asarray(array), int(sampling_rate)
    raise ValueError("Unsupported audio value. Expected HF Audio dict or audio file path.")


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def main():
    parser = argparse.ArgumentParser(description="Convert a Hugging Face audio dataset into IndicFinetuning metadata.csv + wavs.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name or local dataset script/path.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="./IndicFinetuning/datasets/MalayalamDataset")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--normalized-text-column", default=None)
    parser.add_argument("--language-column", default=None)
    parser.add_argument("--default-language", default="ml")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--language-filter", default=None, help="Only export rows matching this language code.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=1.0, help="Skip clips shorter than this many seconds.")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Skip clips longer than this many seconds.")
    parser.add_argument("--unicode-form", default="NFC")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from datasets import Audio, load_dataset

    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    dataset = load_dataset(args.dataset, split=args.split, trust_remote_code=args.trust_remote_code)
    if args.audio_column in dataset.column_names:
        dataset = dataset.cast_column(args.audio_column, Audio(decode=True))

    exported = 0
    skipped = 0

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        for index, item in enumerate(tqdm(dataset, desc="Exporting HF dataset")):
            language_id = get_nested_value(item, args.language_column) if args.language_column else args.default_language
            language_id = resolve_language(safe_text(language_id), args.default_language)

            if args.language_filter and language_id != args.language_filter:
                skipped += 1
                continue

            text = safe_text(get_nested_value(item, args.text_column))
            normalized_text = safe_text(get_nested_value(item, args.normalized_text_column)) if args.normalized_text_column else text
            text = normalize_indic_text(text, args.unicode_form)
            normalized_text = normalize_indic_text(normalized_text, args.unicode_form)
            if not text:
                skipped += 1
                continue

            file_id = safe_text(get_nested_value(item, args.id_column)) if args.id_column else f"segment_{exported:06d}"
            file_id = Path(file_id).stem.replace("|", "_").replace(" ", "_")
            wav_path = wav_dir / f"{file_id}.wav"

            try:
                audio_value = get_nested_value(item, args.audio_column)
                audio_array, sampling_rate = get_audio_array_and_sr(audio_value)
                audio_array = np.asarray(audio_array, dtype=np.float32)
                if audio_array.ndim > 2:
                    raise ValueError(f"Audio has unsupported shape: {audio_array.shape}")
                duration = audio_array.shape[0] / sampling_rate
                if duration < args.min_duration or duration > args.max_duration:
                    skipped += 1
                    continue
                sf.write(str(wav_path), audio_array, sampling_rate)
            except Exception as exc:
                skipped += 1
                print(f"Skipping row {index}: audio export failed: {exc}")
                continue

            writer.writerow([file_id, text, normalized_text, language_id])
            exported += 1
            if args.max_samples is not None and exported >= args.max_samples:
                break

    print(f"Metadata: {metadata_path}")
    print(f"WAV directory: {wav_dir}")
    print(f"Exported rows: {exported}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
