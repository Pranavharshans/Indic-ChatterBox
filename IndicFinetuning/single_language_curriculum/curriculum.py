from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


CER_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


@dataclass(frozen=True)
class CurriculumSample:
    id: str
    source: str
    text: str
    audio_path: str
    speaker_id: str
    gender: str = "unknown"
    language_id: str = "ml"
    duration: float = 0.0
    scenario: str = ""
    cer: float | None = None
    snr: float | None = None
    c50: float | None = None
    pitch_std: float | None = None
    speaking_rate: float | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "CurriculumSample":
        fields = cls.__dataclass_fields__
        payload = {key: value[key] for key in fields if key in value}
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = CER_NUMBER.search(str(value))
    return float(match.group(0)) if match else None


def normalized_text_key(text: str) -> str:
    return " ".join(str(text).casefold().split())


def stable_seed(seed: int, *parts: str) -> int:
    payload = "\x1f".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def cumulative_interval_step(local_step: int, step_offset: int, interval: int) -> int | None:
    if interval <= 0:
        raise ValueError("interval must be positive")
    cumulative_step = int(local_step) + int(step_offset)
    return cumulative_step if cumulative_step > 0 and cumulative_step % interval == 0 else None


def read_jsonl(path: str | Path) -> list[CurriculumSample]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(CurriculumSample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid curriculum row at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[CurriculumSample]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def deduplicate(rows: Sequence[CurriculumSample]) -> list[CurriculumSample]:
    """Remove duplicate IDs and per-speaker duplicate transcripts deterministically."""
    kept = []
    seen_ids = set()
    seen_text = set()
    for row in sorted(rows, key=lambda item: (item.source, item.id)):
        id_key = (row.source, row.id)
        text_key = (row.source, row.speaker_id, normalized_text_key(row.text))
        if id_key in seen_ids or not text_key[-1] or text_key in seen_text:
            continue
        seen_ids.add(id_key)
        seen_text.add(text_key)
        kept.append(row)
    return kept


def filter_ivr(
    rows: Sequence[CurriculumSample],
    *,
    min_duration: float = 2.0,
    max_duration: float = 20.0,
    max_cer: float = 0.15,
    min_snr: float = 20.0,
    min_c50: float = 10.0,
    speaking_rate_trim_fraction: float = 0.01,
) -> list[CurriculumSample]:
    kept = []
    for row in rows:
        if not min_duration <= row.duration <= max_duration:
            continue
        if row.cer is None or row.cer > max_cer:
            continue
        if row.snr is None or row.snr < min_snr:
            continue
        if row.c50 is None or row.c50 < min_c50:
            continue
        kept.append(row)
    rates = sorted(row.speaking_rate for row in kept if row.speaking_rate is not None)
    if len(rates) >= 100 and speaking_rate_trim_fraction > 0:
        trim = max(1, int(len(rates) * speaking_rate_trim_fraction))
        lower = rates[trim]
        upper = rates[-trim - 1]
        kept = [
            row
            for row in kept
            if row.speaking_rate is not None and lower <= row.speaking_rate <= upper
        ]
    return kept


def _split_counts(size: int, validation_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    validation = int(round(size * validation_fraction))
    test = int(round(size * test_fraction))
    train = size - validation - test
    if size >= 3:
        validation = max(validation, 1)
        test = max(test, 1)
        train = size - validation - test
    return train, validation, test


def split_rasa_stratified(
    rows: Sequence[CurriculumSample],
    *,
    seed: int,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> dict[str, list[CurriculumSample]]:
    """Split each Rasa speaker independently so both voices remain in every split."""
    by_speaker: dict[str, list[CurriculumSample]] = defaultdict(list)
    for row in rows:
        if not row.speaker_id:
            raise ValueError(f"Rasa sample {row.id} has no speaker_id")
        by_speaker[row.speaker_id].append(row)

    splits = {"train": [], "validation": [], "test": []}
    for speaker, speaker_rows in sorted(by_speaker.items()):
        shuffled = sorted(speaker_rows, key=lambda row: stable_seed(seed, "rasa", speaker, row.id))
        train_count, validation_count, test_count = _split_counts(
            len(shuffled), validation_fraction, test_fraction
        )
        splits["train"].extend(shuffled[:train_count])
        splits["validation"].extend(shuffled[train_count : train_count + validation_count])
        splits["test"].extend(shuffled[train_count + validation_count : train_count + validation_count + test_count])
    return splits


def split_ivr_speaker_disjoint(
    rows: Sequence[CurriculumSample],
    *,
    seed: int,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> dict[str, list[CurriculumSample]]:
    """Keep each IV-R speaker wholly inside train, validation, or test."""
    by_speaker: dict[str, list[CurriculumSample]] = defaultdict(list)
    for row in rows:
        if not row.speaker_id:
            raise ValueError(f"IV-R sample {row.id} has no speaker_id")
        by_speaker[row.speaker_id].append(row)
    if len(by_speaker) < 3:
        raise ValueError("Speaker-disjoint IV-R splitting requires at least three speakers")

    ordered_speakers = sorted(
        by_speaker,
        key=lambda speaker: stable_seed(seed, "ivr-speaker", speaker),
    )
    validation_target = max(1, int(round(len(rows) * validation_fraction)))
    test_target = max(1, int(round(len(rows) * test_fraction)))
    splits = {"train": [], "validation": [], "test": []}

    destination = "validation"
    for speaker in ordered_speakers:
        if destination == "validation" and len(splits["validation"]) >= validation_target:
            destination = "test"
        if destination == "test" and len(splits["test"]) >= test_target:
            destination = "train"
        splits[destination].extend(by_speaker[speaker])

    if not all(splits.values()):
        raise ValueError("IV-R speaker split produced an empty train, validation, or test split")
    return splits


def expressive_clean_subset(rows: Sequence[CurriculumSample]) -> list[CurriculumSample]:
    """Select clean extempore rows without introducing a learned quality pipeline."""
    candidates = [
        row
        for row in rows
        if row.scenario.casefold() == "extempore"
        and row.cer is not None
        and row.cer <= 0.08
        and row.snr is not None
        and row.snr >= 25.0
        and row.c50 is not None
        and row.c50 >= 20.0
        and row.pitch_std is not None
    ]
    if not candidates:
        return []
    pitch_values = sorted(row.pitch_std for row in candidates if row.pitch_std is not None)
    median_pitch_std = pitch_values[len(pitch_values) // 2]
    return [row for row in candidates if row.pitch_std >= median_pitch_std]


def deterministic_take(
    rows: Sequence[CurriculumSample], count: int, *, seed: int, label: str
) -> list[CurriculumSample]:
    if not rows:
        raise ValueError(f"Cannot build {label}: source pool is empty")
    shuffled = list(rows)
    random.Random(stable_seed(seed, label)).shuffle(shuffled)
    return [shuffled[index % len(shuffled)] for index in range(count)]


def build_stage_manifests(
    rasa_train: Sequence[CurriculumSample],
    ivr_train: Sequence[CurriculumSample],
    ivr_expressive_clean: Sequence[CurriculumSample],
    *,
    seed: int,
) -> dict[str, list[CurriculumSample]]:
    """Build fixed 100/0, 50/50, and 80/20 source manifests."""
    stage1 = deterministic_take(ivr_train, len(ivr_train), seed=seed, label="stage1-ivr")

    stage2_per_source = max(len(rasa_train), len(ivr_train))
    stage2 = deterministic_take(rasa_train, stage2_per_source, seed=seed, label="stage2-rasa")
    stage2.extend(deterministic_take(ivr_train, stage2_per_source, seed=seed, label="stage2-ivr"))
    random.Random(stable_seed(seed, "stage2-merge")).shuffle(stage2)

    stage3_rasa_count = len(rasa_train)
    stage3_ivr_count = math.ceil(stage3_rasa_count / 4)
    stage3 = deterministic_take(rasa_train, stage3_rasa_count, seed=seed, label="stage3-rasa")
    stage3.extend(
        deterministic_take(
            ivr_expressive_clean,
            stage3_ivr_count,
            seed=seed,
            label="stage3-ivr-expressive",
        )
    )
    random.Random(stable_seed(seed, "stage3-merge")).shuffle(stage3)
    return {"stage1": stage1, "stage2": stage2, "stage3": stage3}


def summarize(rows: Sequence[CurriculumSample]) -> dict:
    source_counts = Counter(row.source for row in rows)
    return {
        "samples": len(rows),
        "sources": dict(sorted(source_counts.items())),
        "hours": round(sum(row.duration for row in rows) / 3600.0, 3),
        "speakers": len({(row.source, row.speaker_id) for row in rows}),
    }


def assert_disjoint(splits: dict[str, Sequence[CurriculumSample]], *, speaker_disjoint: bool) -> None:
    ids_by_split = {
        name: {(row.source, row.id) for row in rows}
        for name, rows in splits.items()
    }
    names = list(ids_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = ids_by_split[left] & ids_by_split[right]
            if overlap:
                raise AssertionError(f"Sample leakage between {left} and {right}: {len(overlap)} rows")
    if speaker_disjoint:
        speakers = {
            name: {row.speaker_id for row in rows}
            for name, rows in splits.items()
        }
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = speakers[left] & speakers[right]
                if overlap:
                    raise AssertionError(f"Speaker leakage between {left} and {right}: {len(overlap)} speakers")
