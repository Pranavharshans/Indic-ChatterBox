from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_text import normalize_indic_text
from IndicFinetuning.single_language_curriculum.curriculum import (
    CurriculumSample,
    parse_optional_float,
)


def get_audio_array_and_sr(audio_value):
    import numpy as np
    import soundfile as sf

    if isinstance(audio_value, dict):
        array = audio_value.get("array")
        sample_rate = audio_value.get("sampling_rate")
        if array is not None and sample_rate is not None:
            return np.asarray(array), int(sample_rate)
        audio_bytes = audio_value.get("bytes")
        if audio_bytes:
            from io import BytesIO

            array, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=False)
            return np.asarray(array), int(sample_rate)
        path = audio_value.get("path")
        if path:
            array, sample_rate = sf.read(path, always_2d=False)
            return np.asarray(array), int(sample_rate)
    if isinstance(audio_value, str):
        array, sample_rate = sf.read(audio_value, always_2d=False)
        return np.asarray(array), int(sample_rate)
    raise ValueError("Unsupported audio value. Expected decoded audio, bytes, or a local path.")


def pick_column(columns: list[str], candidates: list[str], role: str, required: bool = True) -> str | None:
    lowered = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    if required:
        raise ValueError(f"Could not infer {role} column from: {columns}")
    return None


def value(row: dict, column: str | None, default=None):
    return row.get(column, default) if column else default


def normalized_gender(raw: Any) -> str:
    text = str(raw or "").strip().casefold()
    if text.startswith("f") or text == "woman":
        return "female"
    if text.startswith("m") or text == "man":
        return "male"
    return "unknown"


def source_schema(dataset, source: str) -> dict[str, str | None]:
    columns = list(dataset.features) if getattr(dataset, "features", None) else list(dataset.column_names)
    common = {
        "audio": pick_column(columns, ["audio", "wav", "speech"], "audio"),
        "text": pick_column(columns, ["normalized", "normalized_text", "text", "transcript"], "text"),
        "speaker": pick_column(columns, ["speaker_id", "speaker", "speaker_name", "spk_id"], "speaker", False),
        "gender": pick_column(columns, ["gender", "speaker_gender", "sex"], "gender", False),
        "duration": pick_column(columns, ["duration", "audio_duration"], "duration", False),
    }
    if source == "ivr":
        common.update(
            {
                "scenario": pick_column(columns, ["scenario"], "scenario", False),
                "cer": pick_column(columns, ["cer"], "CER", False),
                "snr": pick_column(columns, ["snr"], "SNR", False),
                "c50": pick_column(columns, ["c50"], "C50", False),
                "pitch_std": pick_column(columns, ["utterance_pitch_std", "pitch_std"], "pitch std", False),
                "speaking_rate": pick_column(columns, ["speaking_rate"], "speaking rate", False),
            }
        )
    return common


def metadata_passes_ivr_filters(row: dict, schema: dict, args) -> bool:
    duration = parse_optional_float(value(row, schema["duration"]))
    cer = parse_optional_float(value(row, schema.get("cer")))
    snr = parse_optional_float(value(row, schema.get("snr")))
    c50 = parse_optional_float(value(row, schema.get("c50")))
    return (
        duration is not None
        and args.ivr_min_duration <= duration <= args.ivr_max_duration
        and cer is not None
        and cer <= args.max_ivr_cer
        and snr is not None
        and snr >= args.min_ivr_snr
        and c50 is not None
        and c50 >= args.min_ivr_c50
    )


def load_source(args, source: str):
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi, get_token

    if source == "rasa":
        dataset_id = args.rasa_dataset
        config = args.rasa_config
        revision = args.rasa_revision
    else:
        dataset_id = args.ivr_dataset
        config = args.ivr_config
        revision = args.ivr_revision
    resolved_token = args.hf_token or get_token()
    if source == "rasa" and not resolved_token:
        raise RuntimeError(
            "Rasa is gated. Accept its Hub terms and run `hf auth login` before exporting."
        )
    resolved_revision = HfApi(token=resolved_token).dataset_info(
        dataset_id,
        revision=revision,
    ).sha
    kwargs = {
        "split": "train",
        "streaming": True,
        "revision": resolved_revision,
    }
    if resolved_token:
        kwargs["token"] = resolved_token
    dataset = load_dataset(dataset_id, config, **kwargs)
    schema = source_schema(dataset, source)
    dataset = dataset.cast_column(schema["audio"], Audio(decode=False))
    return dataset, schema, dataset_id, config, resolved_revision


def export_source(args, source: str) -> dict:
    import numpy as np
    import soundfile as sf

    dataset, schema, dataset_id, config, revision = load_source(args, source)
    source_dir = Path(args.output).resolve() / source
    wav_dir = source_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = source_dir / "catalog.jsonl"
    metadata_path = source_dir / "source_meta.json"
    source_metadata = {
        "source": source,
        "dataset": dataset_id,
        "config": config,
        "revision": revision,
        "duration": (
            [args.rasa_min_duration, args.rasa_max_duration]
            if source == "rasa"
            else [args.ivr_min_duration, args.ivr_max_duration]
        ),
        "max_ivr_cer": args.max_ivr_cer if source == "ivr" else None,
        "min_ivr_snr": args.min_ivr_snr if source == "ivr" else None,
        "min_ivr_c50": args.min_ivr_c50 if source == "ivr" else None,
    }
    if args.resume and metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous_metadata != source_metadata:
            raise ValueError(
                f"Cannot resume {source}: dataset revision or filters changed. "
                f"Existing={previous_metadata}, requested={source_metadata}"
            )
    metadata_path.write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    completed = set()
    if args.resume and catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(json.loads(line)["id"])

    written = len(completed)
    skipped = 0
    mode = "a" if args.resume else "w"
    with catalog_path.open(mode, encoding="utf-8") as catalog:
        for index, row in enumerate(dataset):
            sample_id = f"{source}_{index:06d}"
            if sample_id in completed:
                continue
            if source == "ivr" and not metadata_passes_ivr_filters(row, schema, args):
                skipped += 1
                continue

            text = normalize_indic_text(value(row, schema["text"], ""))
            if not text:
                skipped += 1
                continue
            try:
                audio, sample_rate = get_audio_array_and_sr(value(row, schema["audio"]))
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)
                duration = len(audio) / float(sample_rate)
            except Exception as exc:
                skipped += 1
                print(f"[{source}] skip row {index}: audio decode failed: {exc}", flush=True)
                continue
            min_duration = args.rasa_min_duration if source == "rasa" else args.ivr_min_duration
            max_duration = args.rasa_max_duration if source == "rasa" else args.ivr_max_duration
            if not min_duration <= duration <= max_duration:
                skipped += 1
                continue

            gender = normalized_gender(value(row, schema["gender"]))
            speaker_id = str(value(row, schema["speaker"], "") or "").strip()
            if not speaker_id and source == "rasa" and gender != "unknown":
                # Rasa contains one released voice for each gender/language pair.
                speaker_id = f"rasa_{gender}"
            if not speaker_id:
                raise ValueError(
                    f"Cannot create leakage-safe splits: {source} row {index} has no speaker ID or usable gender"
                )

            audio_path = wav_dir / f"{sample_id}.wav"
            sf.write(str(audio_path), audio, sample_rate)
            sample = CurriculumSample(
                id=sample_id,
                source=source,
                text=text,
                audio_path=str(audio_path),
                speaker_id=speaker_id,
                gender=gender,
                duration=duration,
                scenario=str(value(row, schema.get("scenario"), "") or ""),
                cer=parse_optional_float(value(row, schema.get("cer"))),
                snr=parse_optional_float(value(row, schema.get("snr"))),
                c50=parse_optional_float(value(row, schema.get("c50"))),
                pitch_std=parse_optional_float(value(row, schema.get("pitch_std"))),
                speaking_rate=parse_optional_float(value(row, schema.get("speaking_rate"))),
            )
            catalog.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                catalog.flush()
                print(f"[{source}] written={written} skipped={skipped}", flush=True)
            if args.limit_per_source and written >= args.limit_per_source:
                break

    return {
        "source": source,
        "dataset": dataset_id,
        "config": config,
        "revision": revision,
        "catalog": str(catalog_path),
        "written": written,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Malayalam Rasa and IV-R to canonical local catalogs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--rasa-dataset", default="ai4bharat/Rasa")
    parser.add_argument("--rasa-config", default="Malayalam")
    parser.add_argument("--rasa-revision", default="main")
    parser.add_argument("--ivr-dataset", default="trysem/indicvoices_r-ML")
    parser.add_argument("--ivr-config", default="default")
    parser.add_argument("--ivr-revision", default="main")
    parser.add_argument("--rasa-min-duration", type=float, default=1.0)
    parser.add_argument("--rasa-max-duration", type=float, default=20.0)
    parser.add_argument("--ivr-min-duration", type=float, default=2.0)
    parser.add_argument("--ivr-max-duration", type=float, default=20.0)
    parser.add_argument("--max-ivr-cer", type=float, default=0.15)
    parser.add_argument("--min-ivr-snr", type=float, default=20.0)
    parser.add_argument("--min-ivr-c50", type=float, default=10.0)
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = [export_source(args, source) for source in ("rasa", "ivr")]
    report = {
        "filters": {
            "rasa_duration": [args.rasa_min_duration, args.rasa_max_duration],
            "ivr_duration": [args.ivr_min_duration, args.ivr_max_duration],
            "max_ivr_cer": args.max_ivr_cer,
            "min_ivr_snr": args.min_ivr_snr,
            "min_ivr_c50": args.min_ivr_c50,
        },
        "sources": summaries,
    }
    (output / "export_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
