from typing import Iterable, List


SUPPORTED_BASE_TAGS = [
    "[laughter]",
    "[giggle]",
    "[sigh]",
    "[cough]",
    "[cry]",
    "[whisper]",
]


CORE_EMOTION_TAGS = [
    "[laughter]",
    "[giggle]",
    "[sigh]",
    "[cry]",
    "[whisper]",
]


OPTIONAL_EMOTION_TAGS = [
    "[cough]",
    "[laugh]",
    "[chuckle]",
    "[sob]",
    "[breath]",
    "[pause]",
    "[angry]",
    "[sad]",
    "[happy]",
    "[excited]",
    "[nervous]",
    "[frustrated]",
]


TAG_GROUPS = {
    "none": [],
    "base": SUPPORTED_BASE_TAGS,
    "core": CORE_EMOTION_TAGS,
    "extended": CORE_EMOTION_TAGS + OPTIONAL_EMOTION_TAGS,
}


def get_emotion_tags(group: str, extra_tokens: Iterable[str] = ()) -> List[str]:
    if group not in TAG_GROUPS:
        supported = ", ".join(sorted(TAG_GROUPS))
        raise ValueError(f"Unsupported emotion tag group '{group}'. Supported: {supported}")

    tags = []
    seen = set()
    for token in list(TAG_GROUPS[group]) + list(extra_tokens):
        token = token.strip()
        if not token:
            continue
        if not (token.startswith("[") and token.endswith("]")):
            raise ValueError(f"Emotion/control token must be bracketed, got: {token}")
        if token not in seen:
            tags.append(token)
            seen.add(token)
    return tags

