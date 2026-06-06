from dataclasses import dataclass
from typing import Dict, Iterable, List


COMMON_PUNCTUATION = list(" .,!?;:'\"-()[]/")


@dataclass(frozen=True)
class IndicLanguage:
    code: str
    name: str
    script: str
    unicode_blocks: List[range]
    extra_graphemes: str = ""

    @property
    def tag(self) -> str:
        return f"[{self.code}]"

    def graphemes(self) -> List[str]:
        chars = []
        seen = set()
        for block in self.unicode_blocks:
            for codepoint in block:
                char = chr(codepoint)
                if char not in seen:
                    chars.append(char)
                    seen.add(char)
        for char in self.extra_graphemes:
            if char not in seen:
                chars.append(char)
                seen.add(char)
        return chars


INDIC_LANGUAGES: Dict[str, IndicLanguage] = {
    "hi": IndicLanguage("hi", "Hindi", "Devanagari", [range(0x0900, 0x0980)]),
    "mr": IndicLanguage("mr", "Marathi", "Devanagari", [range(0x0900, 0x0980)]),
    "ne": IndicLanguage("ne", "Nepali", "Devanagari", [range(0x0900, 0x0980)]),
    "bn": IndicLanguage("bn", "Bengali", "Bengali", [range(0x0980, 0x0A00)]),
    "as": IndicLanguage("as", "Assamese", "Bengali", [range(0x0980, 0x0A00)]),
    "pa": IndicLanguage("pa", "Punjabi", "Gurmukhi", [range(0x0A00, 0x0A80)]),
    "gu": IndicLanguage("gu", "Gujarati", "Gujarati", [range(0x0A80, 0x0B00)]),
    "or": IndicLanguage("or", "Odia", "Odia", [range(0x0B00, 0x0B80)]),
    "ta": IndicLanguage("ta", "Tamil", "Tamil", [range(0x0B80, 0x0C00)]),
    "te": IndicLanguage("te", "Telugu", "Telugu", [range(0x0C00, 0x0C80)]),
    "kn": IndicLanguage("kn", "Kannada", "Kannada", [range(0x0C80, 0x0D00)]),
    "ml": IndicLanguage("ml", "Malayalam", "Malayalam", [range(0x0D00, 0x0D80)]),
    "si": IndicLanguage("si", "Sinhala", "Sinhala", [range(0x0D80, 0x0E00)]),
    "ur": IndicLanguage("ur", "Urdu", "Arabic", [range(0x0600, 0x0700), range(0x0750, 0x0780), range(0x08A0, 0x0900)]),
}


def get_language(code: str) -> IndicLanguage:
    normalized = code.lower().strip()
    if normalized not in INDIC_LANGUAGES:
        supported = ", ".join(sorted(INDIC_LANGUAGES))
        raise ValueError(f"Unsupported language code '{code}'. Supported: {supported}")
    return INDIC_LANGUAGES[normalized]


def get_language_tags(codes: Iterable[str]) -> List[str]:
    return [get_language(code).tag for code in codes]


def get_graphemes(codes: Iterable[str], include_common_punctuation: bool = True) -> List[str]:
    tokens = []
    seen = set()
    if include_common_punctuation:
        for char in COMMON_PUNCTUATION:
            tokens.append(char)
            seen.add(char)
    for code in codes:
        for char in get_language(code).graphemes():
            if char not in seen:
                tokens.append(char)
                seen.add(char)
    return tokens


def list_supported_languages() -> Dict[str, str]:
    return {code: language.name for code, language in INDIC_LANGUAGES.items()}

