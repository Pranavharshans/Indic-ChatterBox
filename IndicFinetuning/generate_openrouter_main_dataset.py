import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.generate_openrouter_pilot_dataset import (  # noqa: E402
    DEFAULT_MODEL,
    make_tts_input,
    maybe_normalize_wav,
    request_speech,
    write_pcm_wav,
)
from IndicFinetuning.indic_text import normalize_indic_text  # noqa: E402


DEFAULT_OUTPUT = "./IndicFinetuning/datasets/OpenRouterMain800"
DEFAULT_FEMALE_VOICES = ["Callirrhoe", "Aoede", "Kore", "Despina", "Laomedeia", "Leda"]
DEFAULT_MALE_VOICES = ["Puck", "Charon", "Fenrir", "Orus", "Iapetus", "Algenib"]


CONVERSATION_SPLIT = {"female": 200, "male": 200}
NEUTRAL_REPLAY_SPLIT = {"female": 100, "male": 100}
EMOTION_SPLIT = [
    ("laughter", "[laughter]", 15, 15),
    ("giggle", "[giggle]", 13, 12),
    ("whisper", "[whisper]", 15, 15),
    ("cry", "[cry]", 13, 12),
    ("sigh_frustration_tired", "[sigh]", 13, 12),
    ("sigh_nervous_uncertain", "[sigh]", 10, 10),
    ("positive_excited", "", 12, 13),
    ("curious_confused", "", 7, 8),
    ("cough", "[cough]", 2, 3),
]


CONVERSATION_OPENERS = [
    "ഇന്ന് രാവിലെ വീട്ടിൽ നിന്ന് ഇറങ്ങുമ്പോൾ തന്നെ ചെറിയ മഴയുടെ മണം ഉണ്ടായിരുന്നു",
    "ഓഫീസിലെത്തിയപ്പോൾ എല്ലാവരും പതുക്കെ ദിവസത്തിന്റെ കാര്യങ്ങളിലേക്ക് കയറുകയായിരുന്നു",
    "ഇന്നലെ രാത്രി കുറച്ച് വൈകിയാണ് ഉറങ്ങിയത്, അതുകൊണ്ട് രാവിലെ എഴുന്നേൽക്കാൻ ചെറിയ ബുദ്ധിമുട്ടുണ്ടായി",
    "ബസിൽ ഇരിക്കുമ്പോൾ അടുത്ത സീറ്റിലെ ആളുകൾ സംസാരിക്കുന്നത് കേട്ട് പഴയ ചില ഓർമ്മകൾ വന്നു",
    "വീട്ടിൽ ചായ കുടിച്ച് ഇരിക്കുമ്പോൾ അമ്മ സാധാരണ പോലെ ദിവസത്തെ കാര്യങ്ങൾ ചോദിച്ചു",
    "കടയിൽ പോയപ്പോൾ പതിവായി കാണുന്ന ചേട്ടൻ ഇന്ന് കുറച്ച് സന്തോഷത്തിലാണ് തോന്നിയത്",
    "വൈകുന്നേരം ജോലി കഴിഞ്ഞ് പുറത്തിറങ്ങിയപ്പോൾ റോഡിൽ പതിവിനെക്കാൾ കുറച്ച് തിരക്ക് കുറവായിരുന്നു",
    "സുഹൃത്ത് ഫോൺ വിളിച്ചപ്പോൾ ആദ്യം സാധാരണ കാര്യമാണെന്ന് കരുതി",
    "ഇന്ന് ലഞ്ച് കഴിക്കാൻ ഇറങ്ങിയപ്പോൾ അടുത്തുള്ള ചെറിയ ഹോട്ടലിൽ നല്ല തിരക്കുണ്ടായിരുന്നു",
    "പുതിയ കാര്യം തുടങ്ങുമ്പോൾ ആദ്യം ചെറിയൊരു മടി തോന്നുന്നത് സ്വാഭാവികമാണ്",
    "വീട്ടിൽ എല്ലാവരും കൂടിയിരുന്നാൽ ചെറിയ കാര്യങ്ങൾ പോലും സംസാരിച്ച് വലിയ ചിരിയാകും",
    "രാവിലെ നടന്ന കാര്യങ്ങൾ വൈകുന്നേരം ഓർത്തപ്പോൾ അത്ര വലിയ പ്രശ്നമൊന്നുമല്ലെന്ന് തോന്നി",
]

CONVERSATION_MIDDLES = [
    "പക്ഷേ കുറച്ച് സമയം കഴിഞ്ഞപ്പോൾ മനസ്സ് പതുക്കെ ശാന്തമായി, പിന്നെ കാര്യങ്ങൾ എല്ലാം സാധാരണയായി മുന്നോട്ട് പോയി",
    "അങ്ങനെ രണ്ട് മിനിറ്റ് സംസാരിച്ചപ്പോൾ തന്നെ ദിവസത്തിന്റെ മൂട് ഒന്ന് മാറി",
    "വലിയ പ്ലാൻ ഒന്നുമില്ലെങ്കിലും ഇങ്ങനെ ചെറിയ ചെറിയ നിമിഷങ്ങളാണ് ചിലപ്പോൾ ദിവസം നല്ലതാക്കുന്നത്",
    "ആദ്യം ബുദ്ധിമുട്ടായി തോന്നിയ കാര്യം പോലും ആരെയെങ്കിലും കൂടെ പറഞ്ഞാൽ കുറച്ച് ലളിതമാകും",
    "പതുക്കെ ചായ കുടിച്ച് ഇരുന്നപ്പോൾ രാവിലെ ഉണ്ടായിരുന്ന തിരക്ക് ഒന്ന് കുറഞ്ഞുപോയി",
    "അവരുടെ സംസാരത്തിൽ നിന്ന് തന്നെ ഓരോ വീട്ടിലും ഒരുപോലെ ചെറിയ പ്രശ്നങ്ങളും സന്തോഷങ്ങളും ഉണ്ടെന്ന് തോന്നി",
    "അപ്പോൾ എനിക്ക് തോന്നിയത്, എല്ലാ കാര്യവും ഉടനെ ശരിയാക്കേണ്ട ആവശ്യമില്ല, കുറച്ച് സമയം കൊടുത്താൽ മതി എന്നാണ്",
    "ചിലപ്പോൾ ആരെങ്കിലും സുഖമാണോ എന്ന് ചോദിക്കുന്നത് പോലും വലിയൊരു ആശ്വാസമായി തോന്നും",
    "ഞാൻ കരുതിയതിനെക്കാൾ കാര്യങ്ങൾ നന്നായി പോയതുകൊണ്ട് പിന്നെ മുഴുവൻ ദിവസം ഒന്ന് ലളിതമായി",
    "ഇങ്ങനെ സാധാരണമായ ഒരു ദിവസം തന്നെയാണ് ചിലപ്പോൾ ഏറ്റവും ഓർമ്മയിൽ നിൽക്കുന്നത്",
]

CONVERSATION_ENDINGS = [
    "അതുകൊണ്ട് നാളെയും ഇങ്ങനെ പതുക്കെ തുടങ്ങാമെന്ന് വിചാരിച്ചു.",
    "അവസാനം എല്ലാം ശരിയാകും എന്നൊരു വിശ്വാസം മനസ്സിൽ ഉണ്ടായിരുന്നു.",
    "വലിയ കാര്യമൊന്നുമല്ല, പക്ഷേ പറയുമ്പോൾ തന്നെ മനസ്സിന് ഭാരം കുറയുന്നു.",
    "ഇത് കേൾക്കുമ്പോൾ നിനക്കും അങ്ങനെ തോന്നിയിട്ടുണ്ടാകും എന്ന് കരുതുന്നു.",
    "അങ്ങനെ ചെറിയൊരു സന്തോഷം കിട്ടിയാൽ മതി, ദിവസം മുഴുവൻ മെച്ചപ്പെടും.",
    "പിന്നെ വീട്ടിലെത്തുമ്പോഴേക്കും ക്ഷീണം ഉണ്ടായിരുന്നെങ്കിലും മനസ്സ് നല്ലതായിരുന്നു.",
    "ഇത് പോലുള്ള ചെറിയ കാര്യങ്ങളാണ് ജീവിതം കുറച്ച് സുഖമായി തോന്നിക്കുന്നത്.",
    "അങ്ങനെ നോക്കുമ്പോൾ ഇന്നത്തെ ദിവസം മോശമായിരുന്നില്ല.",
]

REPLAY_TOPICS = [
    "നാളെ രാവിലെ ഒമ്പത് മണിക്ക് മീറ്റിംഗ് ആരംഭിക്കും",
    "വൈകുന്നേരം മഴയ്ക്ക് സാധ്യതയുള്ളതിനാൽ പുറത്തേക്ക് പോകുന്നവർ കുട കൈയിൽ കരുതണം",
    "റെയിൽവേ സ്റ്റേഷനിലേക്ക് പോകാൻ ഈ വഴിയിലൂടെ നേരെ പോയാൽ മതി",
    "നിങ്ങളുടെ അപേക്ഷ സ്വീകരിച്ചിട്ടുണ്ട്, അടുത്ത ഘട്ടത്തിന്റെ വിവരങ്ങൾ ഉടൻ അറിയിക്കും",
    "ക്ലാസ് തുടങ്ങുന്നതിന് മുമ്പ് വിദ്യാർത്ഥികൾ രജിസ്ട്രേഷൻ പൂർത്തിയാക്കണം",
    "ഈ ഫയൽ അപ്ലോഡ് ചെയ്ത ശേഷം സ്ഥിരീകരണ സന്ദേശം ലഭിക്കുന്നതുവരെ കാത്തിരിക്കണം",
    "ഡോക്ടറെ കാണാൻ വരുന്നവർ പഴയ റിപ്പോർട്ടുകൾ ഉണ്ടെങ്കിൽ കൂടെ കൊണ്ടുവരണം",
    "പുതിയ സമയക്രമം അടുത്ത ആഴ്ച തിങ്കളാഴ്ച മുതൽ പ്രാബല്യത്തിൽ വരും",
    "ബുക്കിംഗ് മാറ്റണമെങ്കിൽ യാത്രയ്ക്ക് കുറഞ്ഞത് ഒരു ദിവസം മുമ്പ് അറിയിക്കണം",
    "ഓഫീസ് ഇന്ന് വൈകുന്നേരം അഞ്ചര വരെ പ്രവർത്തിക്കും",
]

REPLAY_DETAILS = [
    "സമയം പാലിച്ചാൽ അനാവശ്യ തിരക്ക് ഒഴിവാക്കാൻ കഴിയും.",
    "കൂടുതൽ വിവരങ്ങൾ ലഭിച്ചാൽ വീണ്ടും നിങ്ങളെ അറിയിക്കും.",
    "സഹായം ആവശ്യമുണ്ടെങ്കിൽ കൗണ്ടറിലുള്ള ജീവനക്കാരനോട് ചോദിക്കാം.",
    "വിവരങ്ങൾ ശരിയാണെന്ന് ഉറപ്പാക്കിയ ശേഷം മാത്രമേ അപേക്ഷ സമർപ്പിക്കാവൂ.",
    "ഇത് സാധാരണ നടപടിക്രമമാണ്, അതിനാൽ പ്രത്യേകമായി ആശങ്കപ്പെടേണ്ടതില്ല.",
    "മാറ്റങ്ങൾ ഉണ്ടെങ്കിൽ ഔദ്യോഗിക അറിയിപ്പ് വഴി വ്യക്തമാക്കും.",
    "ദയവായി സന്ദേശത്തിലെ നിർദ്ദേശങ്ങൾ ശ്രദ്ധിച്ച് വായിക്കുക.",
    "തിരക്ക് കൂടുതലായ സമയത്ത് കുറച്ച് നേരത്തെ എത്തുന്നത് നല്ലതാണ്.",
]

EMOTION_LINES = {
    "laughter": [
        "അത് പറഞ്ഞപ്പോൾ എനിക്ക് ശരിക്കും ചിരി അടക്കാൻ പറ്റിയില്ല. നീ ഇത്ര അപ്രതീക്ഷിതമായി മറുപടി പറയും എന്ന് കരുതിയില്ല.",
        "ഇന്നലെ നടന്ന ആ ചെറിയ സംഭവം ഓർത്താൽ ഇന്നും ചിരി വരുന്നു. എല്ലാവരും അത്രയും സീരിയസായി നിന്നത് തന്നെയാണ് രസകരം.",
        "അവൻ അത്ര ആത്മവിശ്വാസത്തോടെ പറഞ്ഞിട്ട് അവസാനം തെറ്റിയത് കണ്ടപ്പോൾ എല്ലാവരും പൊട്ടിച്ചിരിച്ചു.",
    ],
    "giggle": [
        "അങ്ങനെ പറയല്ലേ, കേൾക്കുമ്പോൾ തന്നെ എനിക്ക് ചെറിയ ചിരി വരുന്നു. നീ വളരെ സീരിയസായി പറഞ്ഞതുകൊണ്ടാണ് കൂടുതൽ രസം.",
        "ഇത് ആരോടും പറയരുത്, പക്ഷേ ആ രംഗം ഓർത്താൽ എനിക്ക് ഇപ്പോഴും ചിരി നിൽക്കാറില്ല.",
        "നീ പറഞ്ഞ ആ ചെറിയ തമാശ മുഴുവൻ ദിവസവും എന്റെ മനസ്സിൽ തന്നെ കിടന്നു.",
    ],
    "whisper": [
        "ഇത് ആരോടും പറയരുത്. ഞാൻ നിന്നോട് മാത്രം പറയുന്ന കാര്യമാണിത്, അതുകൊണ്ട് പതുക്കെ കേൾക്കണം.",
        "അവിടെ എല്ലാവരും ഇരിക്കുന്നുണ്ട്, അതുകൊണ്ട് ഈ കാര്യം ഇപ്പോൾ ശാന്തമായി പറയാം.",
        "ഒരു ചെറിയ രഹസ്യം പറയാനുണ്ട്. ശബ്ദം കുറച്ച് കേൾക്ക്, പിന്നെ നമുക്ക് പുറത്തു പോയി വിശദമായി സംസാരിക്കാം.",
    ],
    "cry": [
        "അവൻ പറഞ്ഞത് കേട്ടപ്പോൾ മനസ്സിന് വളരെ വേദനയായി. കുറച്ച് നേരം ഒന്നും പറയാൻ കഴിഞ്ഞില്ല.",
        "ഇത്രയും നാൾ കാത്തിരുന്ന കാര്യം ഇങ്ങനെ മാറിപ്പോകുമെന്ന് ഞാൻ ഒരിക്കലും കരുതിയില്ല.",
        "ആ പഴയ ഫോട്ടോ കണ്ടപ്പോൾ ഓർമ്മകൾ എല്ലാം ഒരുമിച്ച് വന്നു. എനിക്ക് സംസാരിക്കാൻ പോലും ബുദ്ധിമുട്ടായി.",
    ],
    "sigh_frustration_tired": [
        "ഇന്ന് എത്ര ശ്രമിച്ചിട്ടും കാര്യം ശരിയായി തീർന്നില്ല. ശരിക്കും ക്ഷീണം തോന്നുന്നുണ്ട്.",
        "ഒരേ കാര്യം വീണ്ടും വീണ്ടും വിശദീകരിക്കേണ്ടി വരുമ്പോൾ മനസ്സിന് നല്ല ബുദ്ധിമുട്ടാണ്.",
        "രാവിലെ മുതൽ ഓടിനടന്നിട്ടും അവസാനം ചെയ്യേണ്ടത് ബാക്കി തന്നെ. ഇനി കുറച്ച് വിശ്രമിക്കണം.",
    ],
    "sigh_nervous_uncertain": [
        "എനിക്ക് ഇതിൽ പൂർണ്ണമായി ഉറപ്പില്ല, പക്ഷേ പറയാതെ ഇരിക്കാൻ പറ്റില്ലെന്ന് തോന്നുന്നു.",
        "ഇത് ശരിയാണോ എന്ന് അറിയില്ല. എന്നാലും ആദ്യം നിന്നോട് പറഞ്ഞ് നോക്കണമെന്ന് തോന്നി.",
        "അവരോട് ഇത് എങ്ങനെ പറയണമെന്ന് എനിക്ക് ഇപ്പോഴും മനസ്സിലാകുന്നില്ല.",
    ],
    "positive_excited": [
        "ഇത് ശരിക്കും നല്ല വാർത്തയാണ്. ഇത്രയും നാളത്തെ പരിശ്രമത്തിന് ഒടുവിൽ നല്ലൊരു ഫലം കിട്ടിയതിൽ എനിക്ക് വളരെ സന്തോഷമുണ്ട്.",
        "ഇന്ന് കേട്ട വാർത്ത കൊണ്ട് മുഴുവൻ ദിവസം തന്നെ പ്രകാശമായി തോന്നുന്നു. ഇനി അടുത്ത ഘട്ടം തുടങ്ങാൻ നല്ല ഉത്സാഹമുണ്ട്.",
        "നമ്മൾ ആലോചിച്ചതുപോലെ തന്നെ കാര്യം നടന്നാൽ ഇത് വളരെ വലിയ മുന്നേറ്റമാകും.",
    ],
    "curious_confused": [
        "അത് എങ്ങനെ സംഭവിച്ചു എന്ന് എനിക്ക് ശരിക്കും അറിയണം. നീ ഒന്ന് വിശദമായി പറഞ്ഞുതരുമോ?",
        "ഞാൻ കേട്ടത് ശരിയാണോ എന്ന് കുറച്ച് സംശയമുണ്ട്. ഒരിക്കൽ കൂടി ശാന്തമായി വിശദീകരിക്കാമോ?",
        "ഇതിലെ ചെറിയ വ്യത്യാസം എനിക്ക് പിടികിട്ടിയില്ല. നീ പറഞ്ഞ ഉദാഹരണം വീണ്ടും പറയാമോ?",
    ],
    "cough": [
        "ക്ഷമിക്കണം, തൊണ്ട കുറച്ച് വരണ്ടുപോയി. ഞാൻ പറഞ്ഞത് വീണ്ടും തുടക്കം മുതൽ പറയാം.",
        "ഒന്ന് കാത്തിരിക്ക്, വെള്ളം കുടിച്ചിട്ട് ഞാൻ വിശദമായി പറയാം. കാര്യത്തിൽ വലിയ പ്രശ്നമൊന്നുമില്ല.",
        "ചെറിയ ചുമയാണ്, പേടിക്കേണ്ട. ബാക്കി കാര്യം ഇപ്പോൾ തന്നെ പറഞ്ഞുതരാം.",
    ],
}

EMOTION_STYLES = {
    "laughter": "Speak Malayalam with a natural laugh at the beginning, then continue cheerfully and conversationally.",
    "giggle": "Speak Malayalam with a small amused giggle, light and friendly.",
    "whisper": "Whisper in Malayalam, quiet and intimate, but still clear.",
    "cry": "Speak Malayalam with a tearful emotional tone, but keep the words understandable.",
    "sigh_frustration_tired": "Speak Malayalam with a tired frustrated sigh at the start, then continue naturally.",
    "sigh_nervous_uncertain": "Speak Malayalam with a nervous uncertain tone and a soft sigh.",
    "positive_excited": "Speak Malayalam with positive excited energy, but do not shout.",
    "curious_confused": "Speak Malayalam with curious or mildly confused interest, natural and clear.",
    "cough": "Speak Malayalam with one short natural cough at the beginning, then continue clearly.",
}


def pick(items: List[str], index: int) -> str:
    return items[index % len(items)]


def clean_file_part(value: str) -> str:
    return value.replace("_", "-").replace(" ", "-")


def parse_voice_list(value: Optional[str], defaults: List[str]) -> List[str]:
    if not value:
        return defaults
    voices = [item.strip() for item in value.split(",") if item.strip()]
    return voices or defaults


def conversation_text(index: int) -> str:
    return normalize_indic_text(
        f"{pick(CONVERSATION_OPENERS, index)}. "
        f"{pick(CONVERSATION_MIDDLES, index * 3)}. "
        f"{pick(CONVERSATION_ENDINGS, index * 5)}"
    )


def neutral_replay_text(index: int) -> str:
    return normalize_indic_text(f"{pick(REPLAY_TOPICS, index)}. {pick(REPLAY_DETAILS, index * 2)}")


def emotion_text(emotion_type: str, tag: str, index: int) -> str:
    spoken = pick(EMOTION_LINES[emotion_type], index)
    return normalize_indic_text(f"{tag} {spoken}" if tag else spoken)


def style_for_sample(category: str, emotion_type: Optional[str]) -> str:
    if category == "conversation":
        return "Speak in natural everyday Malayalam, warm and conversational, not like a newsreader."
    if category == "neutral_replay":
        return "Speak in plain neutral Malayalam with clear pronunciation, steady pacing, and no drama."
    if emotion_type:
        return EMOTION_STYLES[emotion_type]
    raise ValueError(f"Unknown sample style for category={category}")


def add_group(
    plan: List[Dict[str, str]],
    category: str,
    gender: str,
    count: int,
    voices: List[str],
    start_local_index: int = 0,
    emotion_type: Optional[str] = None,
    tag: str = "",
):
    for offset in range(count):
        local_index = start_local_index + offset
        if category == "conversation":
            text = conversation_text(local_index)
        elif category == "neutral_replay":
            text = neutral_replay_text(local_index)
        else:
            if not emotion_type:
                raise ValueError("emotion_type is required for emotion samples")
            text = emotion_text(emotion_type, tag, local_index)

        row_number = len(plan) + 1
        voice = voices[local_index % len(voices)]
        category_part = category if not emotion_type else f"{category}_{emotion_type}"
        file_id = f"synthetic_{row_number:04d}_{clean_file_part(category_part)}_{gender}"
        plan.append(
            {
                "id": file_id,
                "category": category,
                "emotion_type": emotion_type or "",
                "tag": tag,
                "gender": gender,
                "voice": voice,
                "text": text,
                "style": style_for_sample(category, emotion_type),
                "language_id": "ml",
            }
        )


def build_plan(female_voices: List[str], male_voices: List[str]) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = []
    for gender, count in CONVERSATION_SPLIT.items():
        add_group(plan, "conversation", gender, count, female_voices if gender == "female" else male_voices)
    for gender, count in NEUTRAL_REPLAY_SPLIT.items():
        add_group(plan, "neutral_replay", gender, count, female_voices if gender == "female" else male_voices)
    for emotion_type, tag, female_count, male_count in EMOTION_SPLIT:
        add_group(plan, "emotion", "female", female_count, female_voices, emotion_type=emotion_type, tag=tag)
        add_group(plan, "emotion", "male", male_count, male_voices, emotion_type=emotion_type, tag=tag)
    validate_plan(plan)
    return plan


def count_by(items: Iterable[Dict[str, str]], *keys: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        label = "/".join(item[key] for key in keys)
        counts[label] = counts.get(label, 0) + 1
    return counts


def validate_plan(plan: List[Dict[str, str]]):
    if len(plan) != 800:
        raise AssertionError(f"Expected 800 samples, got {len(plan)}")

    category_counts = count_by(plan, "category")
    if category_counts != {"conversation": 400, "neutral_replay": 200, "emotion": 200}:
        raise AssertionError(f"Unexpected category split: {category_counts}")

    gender_counts = count_by(plan, "gender")
    if gender_counts != {"female": 400, "male": 400}:
        raise AssertionError(f"Unexpected gender split: {gender_counts}")

    emotion_counts = count_by([row for row in plan if row["category"] == "emotion"], "emotion_type", "gender")
    expected = {}
    for emotion_type, _, female_count, male_count in EMOTION_SPLIT:
        expected[f"{emotion_type}/female"] = female_count
        expected[f"{emotion_type}/male"] = male_count
    if emotion_counts != expected:
        raise AssertionError(f"Unexpected emotion split: {emotion_counts}")


def write_plan_files(output_dir: Path, plan: List[Dict[str, str]], selected_indexes: Optional[set] = None):
    metadata_path = output_dir / "metadata.csv"
    manifest_path = output_dir / "manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_handle, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_handle:
        writer = csv.writer(metadata_handle, delimiter="|", lineterminator="\n")
        for index, sample in enumerate(plan):
            if selected_indexes is not None and index not in selected_indexes:
                continue
            transcript = normalize_indic_text(sample["text"])
            writer.writerow([sample["id"], transcript, transcript, sample["language_id"]])
            manifest_row = dict(sample)
            manifest_row["tts_input"] = make_tts_input(sample["style"], transcript)
            manifest_handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")


def save_audio_bytes(wav_path: Path, audio_bytes: bytes, response_format: str):
    if response_format == "pcm":
        write_pcm_wav(wav_path, audio_bytes)
    else:
        mp3_path = wav_path.with_suffix(".mp3")
        mp3_path.write_bytes(audio_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp3_path), "-ar", "24000", "-ac", "1", str(wav_path)],
            check=True,
        )
        mp3_path.unlink(missing_ok=True)
    maybe_normalize_wav(wav_path)


def selected_range(total: int, start_index: int, limit: Optional[int]) -> List[int]:
    start = max(start_index, 0)
    end = total if limit is None else min(total, start + max(limit, 0))
    return list(range(start, end))


def main():
    parser = argparse.ArgumentParser(description="Generate the 800-sample Malayalam synthetic continuation dataset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--female-voices", default=",".join(DEFAULT_FEMALE_VOICES))
    parser.add_argument("--male-voices", default=",".join(DEFAULT_MALE_VOICES))
    parser.add_argument("--response-format", choices=["pcm", "mp3"], default="pcm")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based plan index to start generation from.")
    parser.add_argument("--limit", type=int, default=None, help="Generate only this many rows from start-index.")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata and manifest only; do not call OpenRouter.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each generated sample.")
    args = parser.parse_args()

    female_voices = parse_voice_list(args.female_voices, DEFAULT_FEMALE_VOICES)
    male_voices = parse_voice_list(args.male_voices, DEFAULT_MALE_VOICES)
    plan = build_plan(female_voices, male_voices)
    selected_indexes = set(selected_range(len(plan), args.start_index, args.limit))

    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    write_plan_files(output_dir, plan, selected_indexes if args.limit is not None or args.start_index else None)

    print("Plan summary:")
    print(json.dumps(count_by(plan, "category"), ensure_ascii=False, indent=2))
    print(json.dumps(count_by(plan, "gender"), ensure_ascii=False, indent=2))
    print(f"Selected rows: {len(selected_indexes)}")
    print(f"Metadata: {output_dir / 'metadata.csv'}")
    print(f"Manifest: {output_dir / 'manifest.jsonl'}")

    if args.dry_run:
        print("Dry run complete. No audio was requested.")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY before running audio generation.")

    selected = [(index, plan[index]) for index in selected_range(len(plan), args.start_index, args.limit)]
    for position, (index, sample) in enumerate(selected, start=1):
        transcript = normalize_indic_text(sample["text"])
        wav_path = wav_dir / f"{sample['id']}.wav"
        if args.skip_existing and wav_path.exists() and wav_path.stat().st_size > 0:
            print(f"[{position}/{len(selected)}] skip existing {sample['id']}")
            continue

        tts_input = make_tts_input(sample["style"], transcript)
        print(
            f"[{position}/{len(selected)}] index={index} id={sample['id']} "
            f"gender={sample['gender']} voice={sample['voice']} category={sample['category']} emotion={sample['emotion_type'] or '-'}"
        )
        audio_bytes = request_speech(api_key, args.model, sample["voice"], tts_input, args.response_format)
        save_audio_bytes(wav_path, audio_bytes, args.response_format)
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Done. WAVs: {wav_dir}")


if __name__ == "__main__":
    main()
