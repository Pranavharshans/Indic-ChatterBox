import argparse
import json
import os
from collections import Counter
from pathlib import Path


DEFAULT_DATASET_DIR = "./IndicFinetuning/datasets/LaughterCompositeTest/omni_malayalam_speech"
DEFAULT_SAMPLE_RATE = 24000


def load_manifest(dataset_dir: Path) -> list[dict]:
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path} at line {line_no}: {exc}") from exc
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {manifest_path}")
    return rows


def resolve_audio_path(dataset_dir: Path, row: dict) -> Path:
    row_id = row.get("id")
    candidates = []

    if row.get("path"):
        candidates.append(Path(row["path"]))
    if row.get("audio"):
        candidates.append(Path(row["audio"]))
    if row_id:
        candidates.append(dataset_dir / "wavs" / f"{row_id}.wav")

    checked = []
    for candidate in candidates:
        expanded = candidate.expanduser()
        checked.append(str(expanded))

        if expanded.is_absolute() and expanded.exists():
            return expanded

        cwd_path = Path.cwd() / expanded
        checked.append(str(cwd_path))
        if cwd_path.exists():
            return cwd_path.resolve()

        dataset_path = dataset_dir / expanded
        checked.append(str(dataset_path))
        if dataset_path.exists():
            return dataset_path.resolve()

    raise FileNotFoundError(f"Audio file not found for row {row_id}. Checked: {checked}")


def build_rows(dataset_dir: Path, manifest_rows: list[dict], max_rows: int | None) -> list[dict]:
    output_rows = []
    selected_rows = manifest_rows[:max_rows] if max_rows else manifest_rows

    for index, row in enumerate(selected_rows, start=1):
        row_id = row.get("id") or f"sample_{index:06d}"
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"Missing text for row {row_id}")

        audio_path = resolve_audio_path(dataset_dir, row)
        output_rows.append(
            {
                "id": row_id,
                "audio": str(audio_path),
                "text": text,
                "language_id": row.get("language_id", "ml"),
            }
        )
    return output_rows


def validate_rows(rows: list[dict], expected_count: int | None) -> None:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")

    ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    duplicate_texts = [item for item, count in Counter(texts).items() if count > 1]

    if duplicate_ids:
        raise ValueError(f"Duplicate ids found: {duplicate_ids[:10]}")
    if duplicate_texts:
        raise ValueError(f"Duplicate text rows found: {duplicate_texts[:3]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a generated local wav/text manifest as a proper Hugging Face audio dataset."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--hf-namespace", help="HF user or org name, for example Praha-Labs")
    parser.add_argument("--dataset-name", help="HF dataset repo name, for example PrahaTTS-ML-Laughter-Test")
    parser.add_argument("--repo-id", help="Full HF dataset repo id. Overrides --hf-namespace/--dataset-name.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--private", action="store_true", help="Create/upload as a private dataset repo.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Upload only the first N rows. Useful for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print a preview without uploading.")
    parser.add_argument("--commit-message", default="Upload Malayalam audio/text dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    repo_id = args.repo_id
    if not repo_id:
        if not args.hf_namespace or not args.dataset_name:
            raise ValueError("Pass either --repo-id or both --hf-namespace and --dataset-name.")
        repo_id = f"{args.hf_namespace.strip('/')}/{args.dataset_name.strip('/')}"

    manifest_rows = load_manifest(dataset_dir)
    rows = build_rows(dataset_dir, manifest_rows, args.max_rows)
    validate_rows(rows, args.expected_count)

    print(f"Dataset dir: {dataset_dir}")
    print(f"Repo id: {repo_id}")
    print(f"Rows: {len(rows)}")
    print("Columns: id, audio, text, language_id")
    print("First row:")
    print(json.dumps({**rows[0], "audio": Path(rows[0]["audio"]).name}, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("Dry run complete. No upload performed.")
        return

    if not args.token:
        raise ValueError("HF token missing. Pass --token or set HF_TOKEN.")

    try:
        from datasets import Audio, Dataset
    except ImportError as exc:
        raise ImportError('Install dependencies first: pip install -U "datasets[audio]" huggingface_hub') from exc

    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))
    dataset.push_to_hub(
        repo_id,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
