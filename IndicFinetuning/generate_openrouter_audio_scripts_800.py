import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_text import normalize_indic_text


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_OUTPUT = "./IndicFinetuning/datasets/OpenRouterScript800Fixed"

FEMALE_VOICES = ["Callirrhoe", "Aoede", "Kore", "Despina", "Laomedeia", "Leda"]
MALE_VOICES = ["Puck", "Charon", "Fenrir", "Orus", "Iapetus", "Algenib"]
SUPPORTED_TAGS = {"[laughter]", "[giggle]", "[whisper]", "[cry]", "[sigh]", "[cough]"}
TAG_RE = re.compile(r"\[[^\]]+\]")
WORD_RE = re.compile(r"[\u0D00-\u0D7F]+")

TOPICS = [
    "home and family routines",
    "tea shop and evening snacks",
    "office conversations",
    "bus and auto travel",
    "school and college memories",
    "bank and document work",
    "clinic and pharmacy visits",
    "market and grocery shopping",
    "gym and morning walk",
    "park and neighborhood talk",
    "phone calls and messages",
    "small household mistakes",
    "rain, traffic, and waiting",
    "friendship and casual plans",
    "cooking and kitchen stories",
    "local events and announcements",
    "work deadlines and relief",
    "family functions and guests",
    "repair shop and service center",
    "train station and travel plans",
    "library and study routine",
    "cinema, music, and weekend plans",
    "rent, bills, and apartment life",
    "childhood memories",
    "sports practice and playground talk",
    "festival preparation",
    "customer support style replies",
    "appointment reminders",
    "simple directions",
    "weather and schedule updates",
]

STYLE_BY_CATEGORY = {
    "conversation": "Speak in natural everyday Malayalam, casual and human, not like a newsreader.",
    "neutral_replay": "Speak in plain neutral Malayalam with clear pronunciation, steady pacing, and no drama.",
    "emotion": "Speak Malayalam in the specified emotional delivery while keeping every word clean and understandable.",
}

TAG_AUDIO_DIRECTIONS = {
    "[laughter]": "Produce a short audible natural laugh at the control-tag point, then continue speaking cheerfully.",
    "[giggle]": "Produce a small audible giggle at the control-tag point, then continue speaking lightly.",
    "[whisper]": "Switch into a clear quiet whisper at the control-tag point. Keep the Malayalam understandable.",
    "[cry]": "Use a tearful, shaky emotional voice from the control-tag point, with a brief crying catch in the voice.",
    "[sigh]": "Produce a clearly audible sigh at the control-tag point, then continue in the requested tired or nervous tone.",
    "[cough]": "Produce one short natural cough at the control-tag point, then continue speaking normally.",
}

EMOTION_HINTS = {
    "laughter": "funny, amused, relaxed; the [laughter] tag must mark the laugh position",
    "giggle": "small shy amusement; the [giggle] tag must mark the giggle position",
    "whisper": "private, secretive, quiet; the [whisper] tag must mark where whispering starts",
    "cry": "sad, tearful, hurt; the [cry] tag must mark where the crying tone starts",
    "sigh_frustration_tired": "tired or frustrated; the [sigh] tag must mark the sigh position",
    "sigh_nervous_uncertain": "nervous or unsure; the [sigh] tag must mark the sigh position",
    "positive_excited": "positive and excited but no bracket tag",
    "curious_confused": "curious or confused; no bracket tag unless the row explicitly asks for one",
    "cough": "brief cough; the [cough] tag must mark the cough position",
}


@dataclass(frozen=True)
class PlanItem:
    category: str
    gender: str
    count: int
    emotion_type: str = ""
    tag: str = ""


SPLIT_PLAN: List[PlanItem] = [
    PlanItem("conversation", "female", 200),
    PlanItem("conversation", "male", 200),
    PlanItem("neutral_replay", "female", 100),
    PlanItem("neutral_replay", "male", 100),
    PlanItem("emotion", "female", 15, "laughter", "[laughter]"),
    PlanItem("emotion", "male", 15, "laughter", "[laughter]"),
    PlanItem("emotion", "female", 13, "giggle", "[giggle]"),
    PlanItem("emotion", "male", 12, "giggle", "[giggle]"),
    PlanItem("emotion", "female", 15, "whisper", "[whisper]"),
    PlanItem("emotion", "male", 15, "whisper", "[whisper]"),
    PlanItem("emotion", "female", 12, "cry", "[cry]"),
    PlanItem("emotion", "male", 13, "cry", "[cry]"),
    PlanItem("emotion", "female", 13, "sigh_frustration_tired", "[sigh]"),
    PlanItem("emotion", "male", 12, "sigh_frustration_tired", "[sigh]"),
    PlanItem("emotion", "female", 10, "sigh_nervous_uncertain", "[sigh]"),
    PlanItem("emotion", "male", 10, "sigh_nervous_uncertain", "[sigh]"),
    PlanItem("emotion", "female", 13, "positive_excited", ""),
    PlanItem("emotion", "male", 12, "positive_excited", ""),
    PlanItem("emotion", "female", 7, "curious_confused", ""),
    PlanItem("emotion", "male", 8, "curious_confused", ""),
    PlanItem("emotion", "female", 2, "cough", "[cough]"),
    PlanItem("emotion", "male", 3, "cough", "[cough]"),
]


def planned_rows() -> List[Dict[str, str]]:
    rows = []
    running = 1
    per_gender_index = defaultdict(int)
    for item in SPLIT_PLAN:
        for _ in range(item.count):
            per_gender_index[item.gender] += 1
            voices = FEMALE_VOICES if item.gender == "female" else MALE_VOICES
            label = item.emotion_type or item.category
            topic = TOPICS[(running - 1) % len(TOPICS)]
            rows.append(
                {
                    "id": f"script_{running:04d}_{label}_{item.gender}",
                    "category": item.category,
                    "gender": item.gender,
                    "voice": voices[(per_gender_index[item.gender] - 1) % len(voices)],
                    "emotion_type": item.emotion_type,
                    "tag": item.tag,
                    "topic": topic,
                }
            )
            running += 1
    return rows


def clean_transcript_for_tts(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def split_tagged_text(text: str, tag: str) -> tuple[str, str]:
    before, after = text.split(tag, 1)
    return before.strip(), after.strip()


def make_tts_input(row: Dict[str, str]) -> str:
    text = row["text"]
    style = STYLE_BY_CATEGORY[row["category"]]
    tag = row.get("tag", "")
    if not tag:
        return (
            f"{style}\n\n"
            "Say exactly the Malayalam text below. Keep the delivery natural, clean, single-speaker, "
            "and free of background music or English additions.\n\n"
            f'Text: "{text}"'
        )

    before, after = split_tagged_text(text, tag)
    direction = TAG_AUDIO_DIRECTIONS[tag]
    sequence = []
    if before:
        sequence.append(f'First speak this Malayalam text naturally: "{before}"')
    sequence.append(direction)
    if after:
        sequence.append(f'Then speak this Malayalam text naturally: "{after}"')

    return (
        f"{style}\n\n"
        f"The training transcript contains the control tag {tag}. Do not pronounce the bracketed tag. "
        "Instead, perform the audio event exactly at that position.\n\n"
        + "\n".join(f"{idx}. {step}" for idx, step in enumerate(sequence, start=1))
        + "\n\nKeep it single-speaker, clean, natural Malayalam, no background music, no extra words."
    )


def normalize_generated_text(text: str) -> str:
    text = text.strip().replace("|", " ")
    text = re.sub(r"\s+", " ", text)
    return normalize_indic_text(text)


def row_prompt_specs(rows: List[Dict[str, str]]) -> str:
    lines = []
    for row in rows:
        tag_rule = (
            f'Include the exact tag {row["tag"]} exactly once at a natural audio-event position.'
            if row["tag"]
            else "Do not include any bracket tag."
        )
        emotion_rule = EMOTION_HINTS.get(row["emotion_type"], "natural everyday Malayalam")
        lines.append(
            json.dumps(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "gender": row["gender"],
                    "topic": row["topic"],
                    "emotion_type": row["emotion_type"],
                    "tag_rule": tag_rule,
                    "emotion_hint": emotion_rule,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def build_prompt(rows: List[Dict[str, str]], previous_errors: Optional[List[str]] = None) -> str:
    errors = ""
    if previous_errors:
        errors = "\nPrevious attempt failed validation. Fix these issues:\n" + "\n".join(f"- {err}" for err in previous_errors[:20])
    return f"""
Generate Malayalam TTS training transcript rows as JSON only.

Rules:
- Return a JSON array only. No markdown, no explanation.
- Each object must contain exactly: "id" and "text".
- Use natural spoken Malayalam, not news style and not literary essay style.
- Each text should be one or two sentences, around 10 to 28 Malayalam words.
- Do not repeat sentence structures across rows.
- Avoid English words, transliteration, emojis, numbering, quotes, and pipe characters.
- The text should match the row topic, category, gender, and emotion hint.
- For tagged rows, include the exact bracket tag exactly once. Do not add unsupported tags.
- For non-tag rows, include no bracket tags at all.
- Put tags at a natural location in the text, not always at the beginning.
- The tag is a control token for audio. The surrounding Malayalam should make the event plausible.

Rows to generate:
{row_prompt_specs(rows)}
{errors}
""".strip()


def extract_json_array(content: str) -> List[Dict[str, str]]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        raise ValueError("No JSON array found in model response.")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Model response JSON is not an array.")
    return data


def request_chat(api_key: str, model: str, prompt: str, retries: int = 3) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You generate high-quality Malayalam TTS dataset transcripts and return strict JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "top_p": 0.95,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Pranavharshans/Indic-ChatterBox",
        "X-Title": "Indic Chatterbox Malayalam Script Generator",
    }
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(OPENROUTER_CHAT_URL, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenRouter chat request failed after {retries} attempts: {last_error}")


def validate_batch(expected_rows: List[Dict[str, str]], generated_rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[str]]:
    expected_by_id = {row["id"]: row for row in expected_rows}
    generated_by_id = {}
    errors = []
    for item in generated_rows:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            errors.append(f"Invalid generated object: {item}")
            continue
        row_id = str(item["id"])
        if row_id in generated_by_id:
            errors.append(f"Duplicate generated id: {row_id}")
        generated_by_id[row_id] = item

    missing = sorted(set(expected_by_id) - set(generated_by_id))
    extra = sorted(set(generated_by_id) - set(expected_by_id))
    if missing:
        errors.append(f"Missing ids: {missing[:10]}")
    if extra:
        errors.append(f"Unexpected ids: {extra[:10]}")

    merged = []
    for row_id, expected in expected_by_id.items():
        if row_id not in generated_by_id:
            continue
        text = normalize_generated_text(str(generated_by_id[row_id]["text"]))
        row = dict(expected)
        row["text"] = text
        row["spoken_text"] = clean_transcript_for_tts(text)
        row["style"] = STYLE_BY_CATEGORY[row["category"]]
        row["tts_input"] = make_tts_input(row)
        merged.append(row)

        words = WORD_RE.findall(text)
        found_tags = TAG_RE.findall(text)
        unsupported = [tag for tag in found_tags if tag not in SUPPORTED_TAGS]
        if "|" in text:
            errors.append(f"{row_id}: pipe character found")
        if unsupported:
            errors.append(f"{row_id}: unsupported tags {unsupported}")
        if len(words) < 10:
            errors.append(f"{row_id}: too short")
        if len(words) > 34:
            errors.append(f"{row_id}: too long")
        if row["tag"]:
            if found_tags.count(row["tag"]) != 1:
                errors.append(f"{row_id}: expected tag {row['tag']} exactly once, found {found_tags}")
        elif found_tags:
            errors.append(f"{row_id}: no tag expected, found {found_tags}")
        if not re.search(r"[\u0D00-\u0D7F]", text):
            errors.append(f"{row_id}: no Malayalam characters found")
    return merged, errors


def repeated_chunks(rows: List[Dict[str, str]], chunk_size: int = 6) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        words = WORD_RE.findall(TAG_RE.sub("", row["text"]))
        for index in range(len(words) - chunk_size + 1):
            counts[" ".join(words[index : index + chunk_size])] += 1
    return {chunk: count for chunk, count in counts.items() if count > 2}


def validate_full_dataset(rows: List[Dict[str, str]]):
    errors = []
    if len(rows) != 800:
        errors.append(f"Expected 800 rows, got {len(rows)}")
    ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate ids found")
    if len(texts) != len(set(texts)):
        errors.append("Duplicate full texts found")

    category_counts = Counter(row["category"] for row in rows)
    gender_counts = Counter(row["gender"] for row in rows)
    tag_counts = Counter(row["tag"] for row in rows if row["tag"])
    emotion_counts = Counter(row["emotion_type"] for row in rows if row["category"] == "emotion")
    expected_category = {"conversation": 400, "neutral_replay": 200, "emotion": 200}
    expected_gender = {"female": 400, "male": 400}
    expected_tag = {"[laughter]": 30, "[giggle]": 25, "[whisper]": 30, "[cry]": 25, "[sigh]": 45, "[cough]": 5}
    if dict(category_counts) != expected_category:
        errors.append(f"Category count mismatch: {dict(category_counts)}")
    if dict(gender_counts) != expected_gender:
        errors.append(f"Gender count mismatch: {dict(gender_counts)}")
    if dict(tag_counts) != expected_tag:
        errors.append(f"Tag count mismatch: {dict(tag_counts)}")

    repeats = repeated_chunks(rows)
    if repeats:
        worst = sorted(repeats.items(), key=lambda item: item[1], reverse=True)[:10]
        errors.append(f"Repeated 6-word chunks found: {worst}")
    if errors:
        raise AssertionError(json.dumps(errors, ensure_ascii=False, indent=2))

    return {
        "rows": len(rows),
        "category": dict(category_counts),
        "gender": dict(gender_counts),
        "tags": dict(tag_counts),
        "emotion_type": dict(emotion_counts),
    }


def write_outputs(output: Path, rows: List[Dict[str, str]], audit: Dict[str, object]):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metadata.csv").open("w", encoding="utf-8", newline="") as metadata_handle:
        writer = csv.writer(metadata_handle, delimiter="|", lineterminator="\n")
        for row in rows:
            writer.writerow([row["id"], row["text"], row["text"], "ml"])

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest_handle:
        for row in rows:
            full_row = dict(row)
            full_row["language_id"] = "ml"
            manifest_handle.write(json.dumps(full_row, ensure_ascii=False) + "\n")

    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "split_plan.json").write_text(
        json.dumps([item.__dict__ for item in SPLIT_PLAN], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_existing(output: Path) -> List[Dict[str, str]]:
    manifest = output / "manifest.jsonl"
    if not manifest.exists():
        return []
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate validated Malayalam audio-script rows through OpenRouter Gemini Lite.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    args = parser.parse_args()

    plan = planned_rows()
    output = Path(args.output)
    if args.dry_run_plan:
        audit = {
            "planned_rows": len(plan),
            "category": dict(Counter(row["category"] for row in plan)),
            "gender": dict(Counter(row["gender"] for row in plan)),
            "tags": dict(Counter(row["tag"] for row in plan if row["tag"])),
            "emotion_type": dict(Counter(row["emotion_type"] for row in plan if row["category"] == "emotion")),
        }
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY before running this script.")

    completed = {row["id"]: row for row in load_existing(output)} if args.resume else {}
    raw_dir = output / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(plan), args.batch_size):
        batch = plan[start : start + args.batch_size]
        missing_batch = [row for row in batch if row["id"] not in completed]
        if not missing_batch:
            print(f"Batch {start // args.batch_size + 1}: reused {len(batch)} rows")
            continue

        previous_errors = None
        accepted_rows = None
        for attempt in range(1, args.retries + 1):
            prompt = build_prompt(missing_batch, previous_errors)
            content = request_chat(api_key, args.model, prompt)
            (raw_dir / f"batch_{start:04d}_attempt_{attempt}.txt").write_text(content, encoding="utf-8")
            try:
                generated = extract_json_array(content)
                accepted_rows, errors = validate_batch(missing_batch, generated)
            except Exception as exc:
                accepted_rows, errors = None, [str(exc)]
            if not errors and accepted_rows is not None:
                previous_errors = None
                break
            previous_errors = errors
            print(f"Batch {start // args.batch_size + 1} attempt {attempt} failed: {errors[:3]}")
        if accepted_rows is None or previous_errors:
            raise RuntimeError(f"Could not generate valid batch starting at {start}: {previous_errors}")

        for row in accepted_rows:
            completed[row["id"]] = row
        audit_partial = {
            "completed": len(completed),
            "category": dict(Counter(row["category"] for row in completed.values())),
            "gender": dict(Counter(row["gender"] for row in completed.values())),
            "tags": dict(Counter(row["tag"] for row in completed.values() if row["tag"])),
        }
        write_outputs(output, [completed[row["id"]] for row in plan if row["id"] in completed], audit_partial)
        print(f"Batch {start // args.batch_size + 1}: accepted {len(accepted_rows)} rows, completed {len(completed)}/800")

    ordered_rows = [completed[row["id"]] for row in plan]
    audit = validate_full_dataset(ordered_rows)
    write_outputs(output, ordered_rows, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Metadata: {output / 'metadata.csv'}")
    print(f"Manifest: {output / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
