import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable, List, Optional
from unicodedata import normalize

from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_languages import get_graphemes, get_language_tags
from IndicFinetuning.emotion_tags import TAG_GROUPS, get_emotion_tags


def read_csv_texts(path: Path, text_column: int, language_column: Optional[int]) -> List[tuple]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if not row or len(row) <= text_column:
                continue
            language_id = row[language_column].strip() if language_column is not None and len(row) > language_column else None
            rows.append((row[text_column].strip(), language_id))
    return rows


def read_json_texts(path: Path) -> List[tuple]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", data.get("data", []))
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("formatted_text") or item.get("normalized_text") or item.get("text") or ""
        language_id = item.get("language_id") or item.get("lang") or item.get("language")
        if text:
            rows.append((str(text), str(language_id) if language_id else None))
    return rows


def unique_chars(texts: Iterable[str], unicode_form: str) -> Counter:
    counts = Counter()
    for text in texts:
        normalized = normalize(unicode_form, text)
        counts.update(char for char in normalized if char.strip())
    return counts


def main():
    parser = argparse.ArgumentParser(description="Audit Indic tokenizer coverage against language registry and transcripts.")
    parser.add_argument("--tokenizer", default="./pretrained_models/tokenizer.json")
    parser.add_argument("--languages", nargs="+", default=["ml"])
    parser.add_argument("--emotion-tags", choices=sorted(TAG_GROUPS), default="none")
    parser.add_argument("--extra-token", action="append", default=[])
    parser.add_argument("--metadata")
    parser.add_argument("--metadata-format", choices=["csv", "json"], default="csv")
    parser.add_argument("--text-column", type=int, default=2)
    parser.add_argument("--language-column", type=int, default=3)
    parser.add_argument("--unicode-form", default="NFC")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab = set(tokenizer.get_vocab().keys())

    emotion_tags = get_emotion_tags(args.emotion_tags, args.extra_token)
    expected_tokens = get_language_tags(args.languages) + get_graphemes(args.languages) + emotion_tags
    missing_expected = [token for token in expected_tokens if token not in vocab]

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Emotion tag group: {args.emotion_tags}")
    print(f"Emotion/control tags: {', '.join(emotion_tags) if emotion_tags else 'none'}")
    print(f"Missing registry tokens: {len(missing_expected)}")
    if missing_expected:
        print("".join(missing_expected[:200]))

    if args.metadata:
        metadata_path = Path(args.metadata)
        if args.metadata_format == "json":
            rows = read_json_texts(metadata_path)
        else:
            rows = read_csv_texts(metadata_path, args.text_column, args.language_column)
        chars = unique_chars((text for text, _language_id in rows), args.unicode_form)
        missing_from_data = [char for char in chars if char not in vocab]

        print(f"Metadata rows: {len(rows)}")
        print(f"Unique transcript chars: {len(chars)}")
        print(f"Missing transcript chars: {len(missing_from_data)}")
        for char in missing_from_data:
            print(f"{char}\tU+{ord(char):04X}\tcount={chars[char]}")


if __name__ == "__main__":
    main()
