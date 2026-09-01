import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.single_language_curriculum.curriculum import (
    assert_disjoint,
    build_stage_manifests,
    deduplicate,
    expressive_clean_subset,
    filter_ivr,
    read_jsonl,
    split_ivr_speaker_disjoint,
    split_rasa_stratified,
    summarize,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="Build deterministic Malayalam curriculum manifests.")
    parser.add_argument("--rasa-catalog", required=True)
    parser.add_argument("--ivr-catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ivr-cer", type=float, default=0.15)
    parser.add_argument("--min-ivr-snr", type=float, default=20.0)
    parser.add_argument("--min-ivr-c50", type=float, default=10.0)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rasa_raw = read_jsonl(args.rasa_catalog)
    ivr_raw = read_jsonl(args.ivr_catalog)
    rasa = deduplicate(rasa_raw)
    ivr_catalog = deduplicate(ivr_raw)
    if any(row.source != "rasa" for row in rasa):
        raise ValueError("Every row in --rasa-catalog must have source='rasa'")
    if any(row.source != "ivr" for row in ivr_catalog):
        raise ValueError("Every row in --ivr-catalog must have source='ivr'")
    ivr = filter_ivr(
        ivr_catalog,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_cer=args.max_ivr_cer,
        min_snr=args.min_ivr_snr,
        min_c50=args.min_ivr_c50,
    )
    rasa_splits = split_rasa_stratified(rasa, seed=args.seed)
    ivr_splits = split_ivr_speaker_disjoint(ivr, seed=args.seed)
    assert_disjoint(rasa_splits, speaker_disjoint=False)
    assert_disjoint(ivr_splits, speaker_disjoint=True)

    expressive = expressive_clean_subset(ivr_splits["train"])
    stages = build_stage_manifests(
        rasa_splits["train"],
        ivr_splits["train"],
        expressive,
        seed=args.seed,
    )

    for source, splits in (("rasa", rasa_splits), ("ivr", ivr_splits)):
        for split, rows in splits.items():
            write_jsonl(output / "splits" / f"{source}_{split}.jsonl", rows)
    write_jsonl(output / "splits" / "ivr_expressive_clean_train.jsonl", expressive)
    for stage, rows in stages.items():
        write_jsonl(output / "stages" / f"{stage}.jsonl", rows)

    validation = rasa_splits["validation"] + ivr_splits["validation"]
    write_jsonl(output / "splits" / "validation_combined.jsonl", validation)

    summary = {
        "seed": args.seed,
        "inputs": {
            "rasa_catalog": len(rasa_raw),
            "rasa_after_deduplication": len(rasa),
            "ivr_catalog": len(ivr_raw),
            "ivr_after_deduplication": len(ivr_catalog),
            "ivr_after_filtering": len(ivr),
        },
        "filters": {
            "duration": [args.min_duration, args.max_duration],
            "max_ivr_cer": args.max_ivr_cer,
            "min_ivr_snr": args.min_ivr_snr,
            "min_ivr_c50": args.min_ivr_c50,
            "speaking_rate_trim_fraction": 0.01,
        },
        "rasa": {name: summarize(rows) for name, rows in rasa_splits.items()},
        "ivr": {name: summarize(rows) for name, rows in ivr_splits.items()},
        "ivr_expressive_clean_train": summarize(expressive),
        "stages": {name: summarize(rows) for name, rows in stages.items()},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
