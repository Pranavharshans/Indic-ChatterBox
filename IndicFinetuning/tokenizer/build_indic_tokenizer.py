import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, List

from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_languages import get_graphemes, get_language_tags


def unique_in_order(values: Iterable[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def load_existing_vocab(tokenizer_path: Path) -> set:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return set(tokenizer.get_vocab().keys())


def build_token_list(languages: List[str], include_punctuation: bool) -> List[str]:
    return unique_in_order(
        get_language_tags(languages)
        + get_graphemes(languages, include_common_punctuation=include_punctuation)
    )


def write_manifest(output_path: Path, languages: List[str], added_tokens: List[str], final_vocab_size: int):
    manifest = {
        "languages": languages,
        "added_token_count": len(added_tokens),
        "final_vocab_size": final_vocab_size,
        "added_tokens": added_tokens,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build an Indic-extended Chatterbox tokenizer.")
    parser.add_argument("--base-tokenizer", default="./pretrained_models/tokenizer.json")
    parser.add_argument("--output-tokenizer", default="./IndicFinetuning/tokenizer/tokenizer_indic.json")
    parser.add_argument("--languages", nargs="+", default=["ml"])
    parser.add_argument("--no-common-punctuation", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_tokenizer)
    output_path = Path(args.output_tokenizer)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(base_path))
    existing_vocab = load_existing_vocab(base_path)
    candidate_tokens = build_token_list(args.languages, include_punctuation=not args.no_common_punctuation)
    tokens_to_add = [token for token in candidate_tokens if token not in existing_vocab]

    added_count = tokenizer.add_tokens(tokens_to_add)
    tokenizer.save(str(output_path))
    final_vocab_size = len(tokenizer.get_vocab())
    write_manifest(output_path, args.languages, tokens_to_add, final_vocab_size)

    print(f"Base tokenizer: {base_path}")
    print(f"Output tokenizer: {output_path}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Candidate tokens: {len(candidate_tokens)}")
    print(f"Added tokens: {added_count}")
    print(f"Final vocab size: {final_vocab_size}")
    print("Update IndicFinetuning/config_indic.py new_vocab_size to this final vocab size.")


if __name__ == "__main__":
    main()

