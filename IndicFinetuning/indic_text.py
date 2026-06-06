import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from unicodedata import normalize

from IndicFinetuning.indic_languages import get_language


def normalize_indic_text(text: str, unicode_form: str = "NFC") -> str:
    text = normalize(unicode_form, str(text))
    return " ".join(text.split()).strip()


def apply_language_tag(text: str, language_id: Optional[str], enabled: bool = True) -> str:
    if not enabled or not language_id:
        return text
    tag = get_language(language_id).tag
    return text if text.startswith(tag) else f"{tag}{text}"


def resolve_language(language_id: Optional[str], default_language: str) -> str:
    language = language_id or default_language
    return get_language(language).code


def read_ljspeech_rows(csv_path: str, default_language: str, language_column: Optional[int] = 3) -> List[Dict[str, str]]:
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if len(row) < 2:
                continue
            text = row[2] if len(row) > 2 and row[2].strip() else row[1]
            language_id = row[language_column].strip() if language_column is not None and len(row) > language_column else default_language
            rows.append({"id": row[0].strip(), "text": text.strip(), "language_id": resolve_language(language_id, default_language)})
    return rows


def read_json_rows(metadata_path: str, default_language: str) -> List[Dict[str, str]]:
    data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", data.get("data", []))
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        file_id = item.get("id") or item.get("file_id") or item.get("audio_id")
        text = item.get("formatted_text") or item.get("normalized_text") or item.get("text")
        language_id = item.get("language_id") or item.get("lang") or item.get("language") or default_language
        if file_id and text:
            rows.append({"id": str(file_id), "text": str(text), "language_id": resolve_language(str(language_id), default_language)})
    return rows


def read_file_based_rows(wav_dir: str, default_language: str) -> List[Dict[str, str]]:
    rows = []
    for text_path in sorted(Path(wav_dir).glob("*.txt")):
        file_id = text_path.stem
        text = text_path.read_text(encoding="utf-8").strip()
        if text:
            rows.append({"id": file_id, "text": text, "language_id": resolve_language(default_language, default_language)})
    return rows


def collect_languages(rows: Iterable[Dict[str, str]]) -> List[str]:
    languages = []
    seen = set()
    for row in rows:
        language_id = row.get("language_id")
        if language_id and language_id not in seen:
            languages.append(language_id)
            seen.add(language_id)
    return languages

