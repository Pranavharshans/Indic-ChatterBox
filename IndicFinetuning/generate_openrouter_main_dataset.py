import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.generate_openrouter_pilot_dataset import (  # noqa: E402
    DEFAULT_MODEL as DEFAULT_TTS_MODEL,
    make_tts_input,
    maybe_normalize_wav,
    request_speech,
    write_pcm_wav,
)
from IndicFinetuning.indic_text import normalize_indic_text  # noqa: E402


DEFAULT_OUTPUT = "./IndicFinetuning/datasets/OpenRouterMain800"
DEFAULT_FEMALE_VOICES = ["Callirrhoe", "Aoede", "Kore", "Despina", "Laomedeia", "Leda"]
DEFAULT_MALE_VOICES = ["Puck", "Charon", "Fenrir", "Orus", "Iapetus", "Algenib"]
SUPPORTED_TAGS = {"[laughter]", "[giggle]", "[sigh]", "[cry]", "[whisper]", "[cough]"}
TAG_RE = re.compile(r"\[[^\]]+\]")
MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")


CONVERSATION_SPLIT = {"female": 200, "male": 200}
NEUTRAL_REPLAY_SPLIT = {"female": 100, "male": 100}

# The user-requested emotion sub-table summed to 225. This keeps the final set
# at 800 total, 400 female / 400 male, and 200 emotion rows.
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


STYLE_BY_CATEGORY = {
    "conversation": "Speak in natural everyday Malayalam, casual and human, not like a newsreader.",
    "neutral_replay": "Speak in plain neutral Malayalam with clear pronunciation, steady pacing, and no drama.",
}

STYLE_BY_EMOTION = {
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


CONV_OPENERS = [
    "ഇന്ന് രാവിലെ ഫോൺ എടുത്തപ്പോൾ ആദ്യം കണ്ടത് നിന്റെ മെസേജായിരുന്നു",
    "ഓഫീസിലേക്ക് ഇറങ്ങുന്നതിന് മുമ്പ് വീട്ടിൽ ചെറിയൊരു തിരക്കായിരുന്നു",
    "ബസിൽ കയറിയപ്പോൾ പതിവായി കാണുന്ന സീറ്റൊന്നും ഒഴിവില്ലായിരുന്നു",
    "ചായക്കടയിൽ നിന്നപ്പോൾ അടുത്ത മേശയിൽ നടക്കുന്ന സംസാരം കേട്ട് ചിരി വന്നു",
    "വീട്ടിൽ ഉച്ചഭക്ഷണം കഴിക്കുമ്പോൾ അമ്മ ഇന്നത്തെ കാര്യം പതുക്കെ ചോദിച്ചു",
    "വൈകുന്നേരം ജോലി കഴിഞ്ഞിറങ്ങിയപ്പോൾ റോഡിൽ മഴ പെയ്ത് തീർന്ന മണം ഉണ്ടായിരുന്നു",
    "സുഹൃത്ത് വിളിച്ചപ്പോൾ ആദ്യം എന്തെങ്കിലും അടിയന്തര കാര്യമാണെന്ന് കരുതി",
    "ഇന്ന് കടയിൽ പോയപ്പോൾ വേണ്ട സാധനം മാത്രം വാങ്ങാമെന്നായിരുന്നു പ്ലാൻ",
    "കോളേജിലെ പഴയ ഗ്രൂപ്പിൽ ഇന്ന് പെട്ടെന്ന് എല്ലാവരും സംസാരിക്കാൻ തുടങ്ങി",
    "റൂമിൽ ഇരുന്ന് ജോലി ചെയ്യുമ്പോൾ ഇന്റർനെറ്റ് വീണ്ടും പതുക്കെയായി",
    "രാവിലെ നടക്കാൻ പോയപ്പോൾ വഴിയരികിലെ പൂക്കൾ ഒന്ന് ശ്രദ്ധയിൽപ്പെട്ടു",
    "ഇന്ന് വീട്ടിലെ ചെറിയ ജോലികൾ തീർക്കാമെന്ന് കരുതി നേരത്തെ എഴുന്നേറ്റു",
    "അടുത്ത വീട്ടിലെ ചേട്ടൻ രാവിലെ തന്നെ ഒരു സഹായം ചോദിക്കാൻ വന്നു",
    "ബാങ്കിൽ പോകണമെന്ന് വിചാരിച്ചെങ്കിലും ടോക്കൺ കണ്ടപ്പോൾ ഒന്ന് മടി തോന്നി",
    "പുതിയ ഷൂ ഇട്ടിറങ്ങിയ ദിവസം തന്നെ റോഡിൽ വെള്ളക്കെട്ട് കണ്ടു",
    "കുട്ടി ഹോംവർക്ക് ചെയ്യാൻ ഇരിക്കുമ്പോൾ ആദ്യം പത്ത് കാരണങ്ങൾ പറഞ്ഞു",
    "ഇന്ന് മീറ്റിംഗ് തുടങ്ങുന്നതിന് മുമ്പ് എല്ലാവരും സാധാരണ പോലെ തമാശ പറഞ്ഞു",
    "കടൽത്തീരത്ത് കുറച്ച് നേരം ഇരിക്കണമെന്ന് രാവിലെ മുതൽ മനസ്സിലുണ്ടായിരുന്നു",
    "വീട്ടിലെ പഴയ അലമാര തുറന്നപ്പോൾ മറന്നുപോയ കുറെ സാധനങ്ങൾ കിട്ടി",
    "പുതിയ റെസിപ്പി പരീക്ഷിക്കാമെന്ന് കരുതി അടുക്കളയിൽ കയറിയതാണ്",
    "ഇന്ന് ട്രെയിനിൽ മുന്നിൽ ഇരുന്ന ആളോട് വഴിയെക്കുറിച്ച് ചോദിക്കേണ്ടിവന്നു",
    "അവധിദിവസം ആണെങ്കിലും രാവിലെ തന്നെ ആരോ വാതിൽ മുട്ടി",
    "സിനിമ കാണാൻ പോയപ്പോൾ ടിക്കറ്റ് കൗണ്ടറിൽ ചെറിയ ക്യൂ ഉണ്ടായിരുന്നു",
    "പഴയൊരു ഫോട്ടോ ഇന്ന് പെട്ടെന്ന് ഗാലറിയിൽ മുകളിൽ വന്നു",
    "ചെറിയൊരു തലവേദന ഉണ്ടായിരുന്നെങ്കിലും പുറത്തേക്ക് പോകേണ്ടിവന്നു",
    "ഇന്ന് ഷോപ്പിൽ കാർഡ് മെഷീൻ പ്രവർത്തിക്കാതെ കുറച്ച് കാത്തിരിക്കേണ്ടിവന്നു",
    "വീട്ടിൽ എല്ലാവരും ഒത്തു കൂടിയപ്പോൾ ആദ്യം സംസാരിച്ചത് ഭക്ഷണത്തെക്കുറിച്ചായിരുന്നു",
    "വഴിയിൽ പഴയ സ്കൂൾ സുഹൃത്തിനെ കണ്ടപ്പോൾ രണ്ട് മിനിറ്റ് നിന്നു സംസാരിച്ചു",
    "മഴ തുടങ്ങുന്നതിന് മുമ്പ് വസ്ത്രങ്ങൾ എടുത്തുവെയ്ക്കണം എന്ന് ഓർമ്മ വന്നു",
    "രാത്രി ചായ കുടിക്കാൻ ഇരുന്നപ്പോൾ ദിവസത്തെ ക്ഷീണം ഒന്ന് കുറഞ്ഞു",
    "രാവിലെ ഓട്ടോ കിട്ടാൻ കുറച്ച് സമയം എടുത്തെങ്കിലും ഡ്രൈവർ നല്ല ആളായിരുന്നു",
    "ഓൺലൈൻ ഓർഡർ വന്നപ്പോൾ പാക്കറ്റ് തുറക്കാൻ തന്നെ ചെറിയ ആവേശമുണ്ടായിരുന്നു",
    "ഇന്ന് അമ്മമ്മ പഴയ കഥ പറഞ്ഞപ്പോൾ എല്ലാവരും ശാന്തമായി കേട്ടിരുന്നു",
    "കമ്പ്യൂട്ടർ അപ്ഡേറ്റ് തുടങ്ങിയത് തെറ്റായ സമയത്തായിരുന്നു",
    "വൈകുന്നേരത്തെ ചെറിയ നടപ്പിന് ഇറങ്ങിയപ്പോൾ വഴിയിൽ നല്ല കാറ്റുണ്ടായിരുന്നു",
    "ഒരു കാര്യം പറയണമെന്ന് കരുതി വിളിച്ചതാണ്, പക്ഷേ നീ ആദ്യം തന്നെ മറ്റൊരു വിഷയം തുടങ്ങി",
    "ഇന്ന് വീട്ടിൽ കറന്റ് പോയപ്പോൾ എല്ലാവരും ഹാളിൽ കൂടി ഇരിക്കേണ്ടിവന്നു",
    "പുതിയ ആളുകളെ കാണുമ്പോൾ ആദ്യം എനിക്ക് കുറച്ച് മടിയുണ്ടാകും",
    "ഡോക്ടറെ കാണാൻ പോയപ്പോൾ കാത്തിരിപ്പ് ഞാൻ കരുതിയതിനെക്കാൾ കുറവായിരുന്നു",
    "പുസ്തകം വാങ്ങാൻ പോയിട്ട് അവസാനം വേറെയും രണ്ട് സാധനങ്ങൾ എടുത്തു",
]

CONV_MIDDLES = [
    "അത് കണ്ടപ്പോൾ തന്നെ ഇന്നത്തെ ദിവസം കുറച്ച് സുഖമായി തുടങ്ങുന്നു എന്ന് തോന്നി",
    "എല്ലാവരും സ്വന്തം കാര്യം പറയുമ്പോൾ ഞാൻ ശാന്തമായി കേട്ട് നിന്നു",
    "ആദ്യം ചെറിയ അസ്വസ്ഥത തോന്നിയെങ്കിലും പിന്നെ അതിൽ തന്നെ ഒരു രസം കിട്ടി",
    "നീ ഉണ്ടായിരുന്നെങ്കിൽ ഇതിനെക്കുറിച്ച് ഉടനെ ഒരു തമാശ പറഞ്ഞേനേ",
    "വലിയ സംഭവം ഒന്നുമല്ല, പക്ഷേ ആ നിമിഷം മനസ്സിൽ നന്നായി പതിഞ്ഞു",
    "ചിലപ്പോൾ ഇങ്ങനെയുള്ള ചെറിയ കാര്യങ്ങൾ തന്നെ ദിവസം മാറ്റിമറിക്കും",
    "അവിടെ നിന്നപ്പോൾ എല്ലാവരും ഒരുപോലെ തിരക്കിലാണെന്നു മനസ്സിലായി",
    "കുറച്ച് കാത്തിരിക്കേണ്ടി വന്നെങ്കിലും ആരും അത്ര ബുദ്ധിമുട്ടിയില്ല",
    "ആദ്യം പ്ലാൻ ചെയ്തതുപോലെ ഒന്നും നടന്നില്ല, പക്ഷേ അവസാനം ശരിയായി",
    "എന്തോ പറയാൻ തുടങ്ങിയിട്ട് നമ്മൾ രണ്ടുപേരും മറ്റൊരു വിഷയത്തിലേക്ക് പോയി",
    "അന്ന് ഉണ്ടായിരുന്ന ഒരു ഓർമ്മ പെട്ടെന്ന് വീണ്ടും മനസ്സിലേക്ക് വന്നു",
    "ഒന്ന് ശ്വാസം വിട്ടു ഇരുന്നപ്പോൾ കാര്യങ്ങൾ അത്ര സീരിയസല്ലെന്ന് തോന്നി",
    "ഞാൻ കരുതിയതിനേക്കാൾ ആളുകൾ സഹായിക്കാൻ തയ്യാറായിരുന്നു",
    "ചില ചോദ്യങ്ങൾക്ക് ഉടനെ മറുപടി കിട്ടാത്തതും ശരിയാണ് എന്ന് തോന്നി",
    "പതുക്കെ സംസാരിച്ചപ്പോൾ വിഷയം തന്നെ ലളിതമായി മാറി",
    "ഇങ്ങനെ സാധാരണ ദിവസങ്ങളിലാണ് വീട്ടിലെ ശബ്ദം പോലും ആശ്വാസമായി തോന്നുന്നത്",
    "കുറച്ച് നേരം ഒന്നും പറയാതെ ഇരുന്നാലും ചിലപ്പോൾ മനസ്സിന് സമാധാനം കിട്ടും",
    "അവസാനം ചിരിക്കാനൊരു കാരണമെങ്കിലും കിട്ടിയാൽ ദിവസം മോശമല്ല",
    "പുറത്ത് വലിയ തിരക്കായിരുന്നെങ്കിലും അകത്ത് മനസ്സ് ഒന്ന് ശാന്തമായിരുന്നു",
    "ആ ചെറിയ മാറ്റം തന്നെ എനിക്ക് നല്ല ആശ്വാസമായി തോന്നി",
]

CONV_ENDINGS = [
    "അതിനാൽ വൈകുന്നേരം നിന്നോട് ഇത് പറയണം എന്ന് വിചാരിച്ചു.",
    "ഇനി ഇങ്ങനെ വന്നാൽ അത്ര ടെൻഷൻ എടുക്കണ്ടെന്ന് തോന്നുന്നു.",
    "പറഞ്ഞാൽ ചെറിയ കാര്യമാണെങ്കിലും അന്നേരം അതിന് നല്ല വില ഉണ്ടായിരുന്നു.",
    "നാളെ ഇതേ കാര്യം വീണ്ടും കുറച്ച് നന്നായി നോക്കാം.",
    "ഇത് കേട്ടാൽ നിനക്കും അതേ പോലെ തോന്നുമോ എന്ന് അറിയില്ല.",
    "വീട്ടിലെത്തിയപ്പോൾ മാത്രം മുഴുവൻ ക്ഷീണം ഒന്ന് മനസ്സിലായി.",
    "അതുകൊണ്ട് ഇന്ന് ബാക്കി സമയം അല്പം ശാന്തമായി ചെലവഴിക്കാം.",
    "പിന്നെ എല്ലാം ഓർത്തപ്പോൾ ചിരി തന്നെയാണ് വന്നത്.",
    "ചില കാര്യങ്ങൾ പ്ലാൻ ചെയ്യാതെ നടന്നാൽ മാത്രമേ അതിന്റെ രസം കിട്ടൂ.",
    "അങ്ങനെ നോക്കുമ്പോൾ ഇന്നത്തെ ദിവസം വെറുതെയായില്ല.",
    "ഇതുപോലൊരു ചെറിയ ഇടവേള എല്ലാവർക്കും വേണം എന്ന് തോന്നുന്നു.",
    "ഇനി നീ പറയൂ, നിന്റെ ദിവസം എങ്ങനെയായിരുന്നു?",
    "ഒരു ചായ കൂടി കിട്ടിയിരുന്നെങ്കിൽ സംഭവം പൂർണ്ണമായേനേ.",
    "അവസാനം എല്ലാവരും സന്തോഷത്തോടെ പിരിഞ്ഞത് മതി.",
    "ഇത് പിന്നെയും ഓർമ്മയിൽ വരുമെന്ന് തോന്നുന്നു.",
    "അടുത്ത തവണ നീയും കൂടെ വരണം എന്ന് തോന്നി.",
]

CONV_SIDE_NOTES = [
    "അന്നേരം അടുക്കളയിൽ നിന്ന് വരുന്ന മണം കൂടി എല്ലാം വീട്ടുപോലെ തോന്നിച്ചു",
    "ഫോൺ കൈയിൽ ഉണ്ടായിരുന്നിട്ടും ഉടനെ ആരെയും വിളിക്കാൻ തോന്നിയില്ല",
    "വഴിയിലൂടെ പോകുന്ന ആളുകളുടെ തിരക്ക് നോക്കിയാൽ തന്നെ സമയം മനസ്സിലാകും",
    "ചെറിയൊരു പാട്ട് പിന്നിൽ കേട്ടുകൊണ്ടിരുന്നത് മനസ്സിന് നല്ലതായിരുന്നു",
    "അവസാനം ആരും കരുതാത്ത ആളാണ് ശരിയായ വഴി പറഞ്ഞുതന്നത്",
    "കൂടെ നിന്ന ആളുകളുടെ മുഖം കണ്ടപ്പോൾ തന്നെ ടെൻഷൻ കുറച്ചു പോയി",
    "അന്നത്തെ വെളിച്ചം പോലും കുറച്ച് വ്യത്യസ്തമായി തോന്നി",
    "കൈയിൽ ഉണ്ടായിരുന്ന ബാഗ് ഭാരമായിരുന്നെങ്കിലും മനസ്സ് അത്ര ഭാരമായിരുന്നില്ല",
    "ആരും വലിയ ഉപദേശം പറഞ്ഞില്ല, പക്ഷേ ഒപ്പം നിന്നത് മതിയായി",
    "പഴയ ശബ്ദങ്ങൾ കേൾക്കുമ്പോൾ തന്നെ സ്ഥലത്തിന്റെ ഓർമ്മ മാറിപ്പോകും",
    "ഒന്ന് ചിരിച്ച് മറുപടി പറഞ്ഞപ്പോൾ അന്തരീക്ഷം തന്നെ മാറി",
    "വാച്ച് നോക്കിയപ്പോൾ ഞാൻ കരുതിയതിലും സമയം പോയിരുന്നു",
    "അടുത്തുള്ള കുട്ടികൾ കളിക്കുന്ന ശബ്ദം കേട്ടപ്പോൾ വിഷയം അല്പം ലഘുവായി",
    "മേശപ്പുറത്തുണ്ടായിരുന്ന ചായ തണുത്തെങ്കിലും സംസാരം തുടർന്നു",
    "കുറച്ച് സമയം പുറത്തേക്കിറങ്ങി നിന്നപ്പോൾ തല തെളിഞ്ഞു",
    "പറയാതെ വച്ച കാര്യങ്ങൾ ചിലപ്പോൾ മുഖത്ത് തന്നെ കാണാം",
    "ആ ചെറിയ തെറ്റാണ് പിന്നെ എല്ലാവർക്കും സംസാരവിഷയമായത്",
    "കതക് അടച്ചശേഷം മാത്രമാണ് കാര്യം മുഴുവൻ ഓർമ്മ വന്നത്",
    "പേഴ്സ് തിരഞ്ഞുനോക്കുമ്പോൾ പഴയ രസീത് വരെ കിട്ടി",
    "അടുത്ത ദിവസം ഇതിനെക്കുറിച്ച് വീണ്ടും പറയേണ്ടി വരും എന്ന് മനസ്സിലായിരുന്നു",
    "ശബ്ദം കുറച്ചാൽ പോലും ആ സന്തോഷം മറയ്ക്കാൻ പറ്റിയില്ല",
    "പുറത്തെ ചൂട് കുറയുമ്പോൾ ആളുകളുടെ മുഖവും ശാന്തമാകുന്നതുപോലെ തോന്നി",
    "ആദ്യം ചോദിക്കാൻ മടി തോന്നിയെങ്കിലും ചോദിച്ചതിന് ശേഷം ആശ്വാസമായി",
    "തെറ്റായ സമയത്ത് വന്ന കോൾ പോലും പിന്നെ ഉപകാരമായി മാറി",
    "ബാക്കി എല്ലാവരും തിരക്കിലായിരുന്നപ്പോൾ ചെറിയ ഇടവേള കിട്ടിയത് നന്നായി",
    "കാത്തിരിപ്പ് ബോറായിരുന്നെങ്കിലും അടുത്തുള്ള സംഭാഷണം രസകരമായിരുന്നു",
    "പുതിയ സ്ഥലമായതിനാൽ ഓരോ ബോർഡും ശ്രദ്ധിച്ച് വായിക്കേണ്ടിവന്നു",
    "ആ ദിവസം വേഗത്തിൽ തീർന്നുപോകാതെ കുറച്ച് നീണ്ടുനിന്ന പോലെ തോന്നി",
    "ഒരാൾ ശാന്തമായി കേൾക്കുമ്പോൾ തന്നെ സംസാരിക്കാൻ ധൈര്യം കിട്ടും",
    "മഴ നിർത്തിയെങ്കിലും നിലത്ത് വെള്ളത്തിന്റെ മണം ബാക്കിയുണ്ടായിരുന്നു",
    "അവസാനം എടുത്ത ചെറിയ തീരുമാനമാണ് ഏറ്റവും ശരിയായത് പോലെ തോന്നി",
    "എല്ലാം സാധാരണമായിരുന്നെങ്കിലും ആ നിമിഷത്തിന് സ്വന്തം ഭംഗിയുണ്ടായിരുന്നു",
    "പിന്നീട് ഓർക്കുമ്പോൾ അതൊരു നല്ല പാഠമായിരുന്നു",
    "ചില സമയത്ത് നിശ്ശബ്ദത തന്നെ നല്ല മറുപടിയാകാം",
    "ആദ്യം ശ്രദ്ധിക്കാത്തൊരു ചെറിയ കാര്യം പിന്നെ പ്രധാനമായി",
    "അവിടെ ഉണ്ടായിരുന്ന കസേര പോലും പഴയ ഓർമ്മപോലെ തോന്നി",
    "തിരക്കിനിടയിലും ഒരാളുടെ നല്ല വാക്ക് മനസ്സിൽ നിൽക്കും",
]

NEUTRAL_TOPICS = [
    "നാളെ രാവിലെ പത്ത് മണിക്ക് സംസാരിക്കാം",
    "നിങ്ങളുടെ ഫയൽ ഞാൻ പരിശോധിച്ചു",
    "അത് ഇപ്പോൾ തന്നെ മാറ്റേണ്ട ആവശ്യമില്ല",
    "വീട്ടിൽ നിന്ന് ഇറങ്ങുമ്പോൾ ചാർജർ എടുത്തുവെയ്ക്കണം",
    "ഈ വഴിയിലൂടെ പോയാൽ സ്റ്റേഷനിൽ വേഗം എത്താം",
    "ഇന്ന് വൈകുന്നേരം മഴയ്ക്ക് സാധ്യതയുണ്ട്",
    "അവരുടെ മറുപടി കിട്ടിയാൽ ഞാൻ നിങ്ങളെ അറിയിക്കും",
    "അടുത്ത ആഴ്ച സമയം മാറ്റിയിട്ടുണ്ട്",
    "പേയ്‌മെന്റ് പൂർത്തിയായ ശേഷം രസീത് ലഭിക്കും",
    "മീറ്റിംഗിന് മുമ്പ് രണ്ട് മിനിറ്റ് സംസാരിക്കണം",
    "ഇപ്പോൾ തിരക്ക് കുറവാണ്, പോകാൻ നല്ല സമയമാണ്",
    "ആ നമ്പറിലേക്ക് വീണ്ടും വിളിച്ച് നോക്കൂ",
    "പുതിയ പാസ്‌വേഡ് സുരക്ഷിതമായി സൂക്ഷിക്കണം",
    "ഡോക്ടറെ കാണാൻ പഴയ റിപ്പോർട്ട് കൊണ്ടുവരണം",
    "ക്ലാസ് തുടങ്ങുന്നതിന് മുമ്പ് രജിസ്ട്രേഷൻ പൂർത്തിയാക്കണം",
    "ബുക്കിംഗ് മാറ്റാൻ ഒരു ദിവസം മുമ്പ് അറിയിക്കണം",
    "ഓർഡർ ഇന്ന് രാത്രി വരെ എത്താൻ സാധ്യതയുണ്ട്",
    "ഈ ഫോം പൂരിപ്പിച്ചതിന് ശേഷം സമർപ്പിക്കുക",
    "നിങ്ങൾ പറഞ്ഞ വിലാസം ഞാൻ കുറിച്ചുവെച്ചു",
    "ആപ്പ് തുറക്കാത്ത പക്ഷം ഫോൺ വീണ്ടും ഓൺ ചെയ്ത് നോക്കൂ",
    "ഇപ്പോൾ ലഭ്യമായ സീറ്റുകൾ കുറവാണ്",
    "കുട്ടികളെ സമയത്ത് കൊണ്ടുവരാൻ ശ്രദ്ധിക്കണം",
    "ഈ രേഖയുടെ കോപ്പി മാത്രം മതി",
    "അവിടെ പാർക്കിംഗ് സൗകര്യം പിന്നിലെ ഗേറ്റിലാണ്",
    "മരുന്ന് ഭക്ഷണത്തിന് ശേഷം കഴിക്കണം",
    "വൈകുന്നേരത്തെ പരിപാടി അകത്തെ ഹാളിലാണ്",
    "ചെറിയ ഇടവേളയ്ക്കു ശേഷം സെഷൻ തുടരും",
    "നിങ്ങളുടെ പേര് ലിസ്റ്റിൽ ചേർത്തിട്ടുണ്ട്",
    "അടുത്ത ഘട്ടത്തിന്റെ വിവരങ്ങൾ സന്ദേശമായി വരും",
    "സർവീസ് പൂർത്തിയാകാൻ ഏകദേശം ഒരു മണിക്കൂർ വേണം",
    "ഇന്ന് കട അഞ്ചര വരെ തുറന്നിരിക്കും",
    "മാറ്റം വേണമെങ്കിൽ ഇപ്പോൾ തന്നെ പറയാം",
    "വീഡിയോ അയച്ചാൽ ഞാൻ കേട്ട് നോക്കാം",
    "പുതിയ തീയതി സ്ഥിരീകരിച്ചാൽ അറിയിക്കുക",
    "വിലയിൽ ചെറിയ മാറ്റം വന്നിട്ടുണ്ട്",
    "അവിടെ എത്തുമ്പോൾ റിസപ്ഷനിൽ പേര് പറയുക",
    "കൂടുതൽ സഹായം വേണമെങ്കിൽ വീണ്ടും വിളിക്കാം",
    "ഇത് സാധാരണ നടപടിക്രമത്തിന്റെ ഭാഗമാണ്",
    "നാളെ മുതൽ സമയക്രമം പഴയപോലെ ആയിരിക്കും",
    "സന്ദേശത്തിലെ ലിങ്ക് തുറന്ന് വിശദാംശങ്ങൾ നോക്കാം",
]

NEUTRAL_DETAILS = [
    "സമയം ഒന്ന് ഉറപ്പാക്കിയാൽ ബാക്കി കാര്യങ്ങൾ എളുപ്പമാകും.",
    "ഇപ്പോൾ പ്രത്യേകമായി ആശങ്കപ്പെടേണ്ട കാര്യമൊന്നുമില്ല.",
    "കൂടുതൽ വിവരം കിട്ടുന്ന ഉടൻ ഞാൻ പറഞ്ഞുതരാം.",
    "ആദ്യമായി ഇത് ചെയ്തു നോക്കൂ, പിന്നെയും പ്രശ്നമുണ്ടെങ്കിൽ അറിയിക്കൂ.",
    "വേഗം വേണ്ട, ശാന്തമായി പരിശോധിച്ച് തീരുമാനിക്കാം.",
    "ആവശ്യമുള്ള രേഖകൾ കൈയിൽ ഉണ്ടെങ്കിൽ പ്രക്രിയ വേഗത്തിലാകും.",
    "നിങ്ങൾക്ക് സൗകര്യമുള്ള സമയം പറഞ്ഞാൽ അതനുസരിച്ച് മാറ്റാം.",
    "സംശയം ഉണ്ടെങ്കിൽ വീണ്ടും ചോദിക്കാൻ മടിക്കേണ്ട.",
    "സാധ്യമായാൽ കുറച്ച് നേരത്തെ എത്തുന്നത് നല്ലതാണ്.",
    "ഇത് കഴിഞ്ഞാൽ ബാക്കി ഘട്ടം വളരെ ലളിതമാണ്.",
]

EMOTION_BASES = {
    "laughter": [
        "അത് കേട്ടപ്പോൾ എനിക്ക് ചിരി അടക്കാൻ പറ്റിയില്ല",
        "നീ പറഞ്ഞ ആ മറുപടി ഇപ്പോഴും ഓർത്താൽ ചിരി വരുന്നു",
        "അവൻ അത്ര ഗൗരവമായി നിന്നിട്ട് അവസാനം ചെയ്തത് കണ്ടോ",
        "അമ്മ പറഞ്ഞ ആ ചെറിയ കമന്റ് മുഴുവൻ വീട്ടിനെയും ചിരിപ്പിച്ചു",
        "കൂട്ടുകാരൻ തെറ്റായ ഗ്രൂപ്പിലേക്ക് മെസേജ് അയച്ച കാര്യം കേട്ടോ",
        "ആ കുട്ടി അത്ര ആത്മവിശ്വാസത്തോടെ പറഞ്ഞതും പിന്നെ തന്നെ ചിരിച്ചതും രസമായിരുന്നു",
        "വീഡിയോയിൽ നീ വീഴാതെ പിടിച്ച് നിന്ന രീതിയാണ് ഏറ്റവും രസകരം",
        "കേക്കിലെ പേരെഴുത്ത് കണ്ടപ്പോൾ എല്ലാവർക്കും ഒരുമിച്ച് ചിരി വന്നു",
        "അവസാന നിമിഷം നീ പറഞ്ഞ ഡയലോഗ് മുഴുവൻ സീൻ മാറ്റി",
        "ഞാൻ സീരിയസായി പറയാൻ തുടങ്ങിയതാണ്, പക്ഷേ നിന്റെ മുഖം കണ്ടപ്പോൾ പോയി",
    ],
    "giggle": [
        "അങ്ങനെ പറയല്ലേ, കേൾക്കുമ്പോൾ തന്നെ ചെറിയ ചിരി വരുന്നു",
        "ഇത് ആരോടും പറയരുത്, പക്ഷേ ആ സംഭവം ഓർത്താൽ ഇപ്പോഴും ചിരിയുണ്ട്",
        "നീ അത്ര നിഷ്കളങ്കമായി ചോദിച്ചതാണ് എനിക്ക് രസമായത്",
        "അവൾ മുഖം മറച്ച് ചിരിച്ചത് കണ്ടപ്പോൾ എനിക്കും പിടിച്ചു നിൽക്കാൻ പറ്റിയില്ല",
        "ആ ചെറിയ തമാശ മുഴുവൻ ദിവസവും മനസ്സിൽ തന്നെ കിടന്നു",
        "നിന്റെ ശബ്ദത്തിൽ തന്നെ തമാശ തുടങ്ങുന്ന പോലെ തോന്നി",
        "ഞാൻ ഗൗരവമായി ഇരിക്കാൻ നോക്കി, പക്ഷേ അത് നടക്കില്ലായിരുന്നു",
        "അവിടെ എല്ലാവരും ശാന്തമായി ഇരിക്കുമ്പോൾ നീ മാത്രം കണ്ണുകൊണ്ട് സൂചന കൊടുത്തു",
    ],
    "whisper": [
        "ഇത് ആരോടും പറയരുത്, ഞാൻ നിന്നോട് മാത്രം പറയുന്ന കാര്യമാണിത്",
        "അവിടെ എല്ലാവരും ഇരിക്കുന്നുണ്ട്, അതുകൊണ്ട് ഇത് പതുക്കെ പറയാം",
        "ഒരു ചെറിയ രഹസ്യം ഉണ്ട്, ശബ്ദം കുറച്ച് കേൾക്ക്",
        "ഇപ്പോൾ പുറത്തേക്ക് വരാൻ പറ്റില്ല, പക്ഷേ കാര്യം പ്രധാനമാണ്",
        "അവർ കേൾക്കുന്നതിന് മുമ്പ് ഞാൻ വേഗം പറഞ്ഞുതരാം",
        "ഈ പ്ലാൻ ഇപ്പോൾ നമ്മൾ രണ്ടുപേരുടെ ഇടയിൽ മാത്രം ഇരിക്കട്ടെ",
        "വാതിൽക്കൽ ആരോ നിൽക്കുന്നുണ്ട്, അതുകൊണ്ട് ഞാൻ പതുക്കെ സംസാരിക്കുന്നു",
        "കുഞ്ഞ് ഉറങ്ങുകയാണ്, അതുകൊണ്ട് ശബ്ദം ഉയർത്താതെ കേൾക്കണം",
        "മീറ്റിംഗ് കഴിഞ്ഞാൽ വിശദമായി പറയാം, ഇപ്പോൾ ഇത്ര മാത്രം ഓർക്കൂ",
        "അവളുടെ സർപ്രൈസ് പൊളിയാതിരിക്കാൻ ഞാൻ വളരെ പതുക്കെ പറയുന്നു",
    ],
    "cry": [
        "അവൻ പറഞ്ഞത് കേട്ടപ്പോൾ മനസ്സിന് വളരെ വേദനയായി",
        "ആ പഴയ ഫോട്ടോ കണ്ടപ്പോൾ ഓർമ്മകൾ എല്ലാം ഒരുമിച്ച് വന്നു",
        "ഇത്രയും നാൾ കാത്തിരുന്ന കാര്യം ഇങ്ങനെ മാറുമെന്ന് കരുതിയില്ല",
        "അവൾ പോകുമ്പോൾ ഒന്നും പറയാൻ എനിക്ക് കഴിഞ്ഞില്ല",
        "വീട്ടിലെ ശൂന്യത ഇന്ന് വളരെ ശക്തമായി തോന്നുന്നു",
        "നന്ദി പറയണമെന്ന് വിചാരിച്ചു, പക്ഷേ ശബ്ദം തന്നെ വിറച്ചു",
        "ആ വാർത്ത കേട്ട ശേഷം കുറച്ച് നേരം ഞാൻ മിണ്ടാതെ ഇരുന്നു",
        "കൈയിൽ ഉണ്ടായിരുന്ന ചെറിയ കത്ത് വായിച്ചപ്പോൾ കണ്ണ് നിറഞ്ഞു",
        "നമ്മൾ ഒരുമിച്ച് ചെയ്ത കാര്യങ്ങൾ ഓർത്തപ്പോൾ മനസ്സ് പിടിച്ചുനിൽക്കാൻ പറ്റിയില്ല",
        "അവസാനമായി കണ്ട ദിവസം ഇത്രയും പെട്ടെന്ന് ഓർമ്മ വരുമെന്ന് കരുതിയില്ല",
    ],
    "sigh_frustration_tired": [
        "ഇന്ന് എത്ര ശ്രമിച്ചിട്ടും കാര്യം ശരിയായി തീർന്നില്ല",
        "ഒരേ കാര്യം വീണ്ടും വീണ്ടും വിശദീകരിക്കേണ്ടി വരുമ്പോൾ ശരിക്കും ക്ഷീണം തോന്നുന്നു",
        "രാവിലെ മുതൽ ഓടിനടന്നിട്ടും ചെയ്യേണ്ടത് ബാക്കി തന്നെ",
        "ഫോം സമർപ്പിക്കാൻ നോക്കുമ്പോഴെല്ലാം പുതിയൊരു പിശക് വരുന്നു",
        "ഇത്ര ചെറിയ കാര്യം തീർക്കാൻ ഇത്ര സമയം പോകുമെന്ന് കരുതിയില്ല",
        "എല്ലാവരും അവസാന നിമിഷം ചോദിക്കുമ്പോൾ കൈകാര്യം ചെയ്യാൻ ബുദ്ധിമുട്ടാണ്",
        "ഫോൺ ചാർജ് തീർന്നതും ബസ് വൈകിയതും ഒരേ സമയം വന്നു",
        "ഇന്ന് കുറച്ച് വിശ്രമിക്കണം എന്ന് വിചാരിച്ച ദിവസം തന്നെ അധിക ജോലി വന്നു",
        "പറയുന്നത് ആരും കേൾക്കാതെ പോയാൽ വീണ്ടും തുടങ്ങേണ്ടി വരും",
        "പദ്ധതി നല്ലതായിരുന്നു, പക്ഷേ നടപ്പാക്കുമ്പോൾ എല്ലാം കുഴഞ്ഞുപോയി",
    ],
    "sigh_nervous_uncertain": [
        "എനിക്ക് ഇതിൽ പൂർണ്ണമായി ഉറപ്പില്ല, പക്ഷേ പറയാതെ ഇരിക്കാൻ പറ്റില്ല",
        "ഇത് ശരിയാണോ എന്ന് അറിയില്ല, എന്നാലും ആദ്യം നിന്നോട് പറയണം തോന്നി",
        "അവരോട് എങ്ങനെ തുടങ്ങണം എന്നത് ഇപ്പോഴും മനസ്സിലാകുന്നില്ല",
        "മറുപടി എന്തായിരിക്കും എന്ന് ചിന്തിച്ചാൽ തന്നെ ചെറിയ ഭയം ഉണ്ട്",
        "ഞാൻ തെറ്റായി മനസ്സിലാക്കിയതാണോ എന്നൊരു സംശയം ഇപ്പോഴും ബാക്കി നിൽക്കുന്നു",
        "ഇപ്പോൾ പറയാമോ അല്ലെങ്കിൽ കുറച്ച് കാത്തിരിക്കാമോ എന്ന് തീരുമാനിക്കാൻ പറ്റുന്നില്ല",
        "ആ ഫലം വരുന്നത് വരെ മനസ്സ് ഒന്ന് ശാന്തമാകുന്നില്ല",
        "എനിക്ക് ശ്രമിക്കണം, പക്ഷേ ആദ്യ പടി എടുക്കാൻ മടി തോന്നുന്നു",
    ],
    "positive_excited": [
        "ഇത് ശരിക്കും നല്ല വാർത്തയാണ്",
        "ഇന്ന് കേട്ട കാര്യം കൊണ്ട് മുഴുവൻ ദിവസം തന്നെ പ്രകാശമായി തോന്നുന്നു",
        "നമ്മൾ ആലോചിച്ചതുപോലെ കാര്യം നടന്നാൽ ഇത് വലിയ മുന്നേറ്റമാകും",
        "ഇത്രയും നാളത്തെ പരിശ്രമത്തിന് ഒടുവിൽ നല്ല ഫലം കിട്ടി",
        "അടുത്ത ഘട്ടം തുടങ്ങാൻ ഇപ്പോൾ നല്ല ഉത്സാഹമുണ്ട്",
        "ഇത് വീട്ടിൽ പറഞ്ഞാൽ എല്ലാവർക്കും സന്തോഷമാകും",
        "ചെറിയ തുടക്കമാണെങ്കിലും ഇതിന് നല്ല സാധ്യതയുണ്ട്",
        "നീ പറഞ്ഞത് ശരിയായിരുന്നു, കാത്തിരിപ്പ് വെറുതെയായില്ല",
        "ഇനി ഇത് നന്നായി മുന്നോട്ട് കൊണ്ടുപോകണം എന്ന് തോന്നുന്നു",
        "ഇത്ര പെട്ടെന്ന് ഇത്ര നല്ല പ്രതികരണം കിട്ടുമെന്ന് കരുതിയില്ല",
    ],
    "curious_confused": [
        "അത് എങ്ങനെ സംഭവിച്ചു എന്ന് എനിക്ക് ശരിക്കും അറിയണം",
        "ഞാൻ കേട്ടത് ശരിയാണോ എന്ന് കുറച്ച് സംശയമുണ്ട്",
        "ഇതിലെ ചെറിയ വ്യത്യാസം എനിക്ക് പിടികിട്ടിയില്ല",
        "നീ പറഞ്ഞ ഉദാഹരണം ഒരിക്കൽ കൂടി പറഞ്ഞുതരുമോ",
        "അവിടെ നിന്ന് ഇവിടേക്ക് കാര്യം എങ്ങനെ എത്തിയെന്ന് മനസ്സിലായില്ല",
        "ഇത് സാധാരണ പ്രശ്നമാണോ അല്ലെങ്കിൽ വേറെ എന്തെങ്കിലും കാരണമുണ്ടോ",
        "നമ്മൾ ആദ്യം തീരുമാനിച്ചതിൽ നിന്ന് എന്താണ് മാറിയത്",
        "ആ വാക്കിന്റെ അർത്ഥം ഇവിടെ വേറെയാണോ എന്ന് എനിക്ക് സംശയമുണ്ട്",
    ],
    "cough": [
        "ക്ഷമിക്കണം, തൊണ്ട കുറച്ച് വരണ്ടുപോയി",
        "ഒന്ന് കാത്തിരിക്ക്, വെള്ളം കുടിച്ചിട്ട് ഞാൻ തുടരും",
        "ചെറിയ ചുമയാണ്, പേടിക്കേണ്ട",
        "വാക്ക് നടുവിൽ നിന്നുപോയി, ഞാൻ വീണ്ടും തുടങ്ങാം",
    ],
}

EMOTION_DETAILS = [
    "പക്ഷേ കാര്യം മുഴുവൻ കേട്ടാൽ നിനക്കും മനസ്സിലാകും.",
    "ഇപ്പോൾ പറഞ്ഞില്ലെങ്കിൽ പിന്നെ പറയാൻ ബുദ്ധിമുട്ടാകും.",
    "അന്നേരം എല്ലാവരും ശാന്തമായി നിന്നത് തന്നെ വിചിത്രമായി തോന്നി.",
    "ഞാൻ കരുതിയതിനെക്കാൾ അത് മനസ്സിൽ കൂടുതൽ പിടിച്ചു.",
    "കുറച്ച് സമയം എടുത്താലും ഇത് ശാന്തമായി സംസാരിക്കണം.",
    "നിന്റെ മറുപടി കേട്ടാൽ എനിക്ക് തീരുമാനിക്കാൻ എളുപ്പമാകും.",
    "അവിടെ ഉണ്ടായിരുന്നവർക്ക് പോലും അത് മറക്കാൻ പറ്റില്ല.",
    "ഇപ്പോൾ ഓർത്താലും ആ നിമിഷം നേരെ മുന്നിൽ വരുന്നു.",
    "ഇത് ചെറിയ കാര്യമല്ലെന്ന് അപ്പോൾ തന്നെയാണ് മനസ്സിലായത്.",
    "അവസാനം എന്തായാലും സത്യം പറയുന്നതാണ് നല്ലത്.",
    "ഇനി അടുത്തത് എന്താണെന്ന് നമുക്ക് നോക്കാം.",
    "ഒരു മിനിറ്റ് ശാന്തമായി ഇരുന്നാൽ ഞാൻ ബാക്കി പറയാം.",
]

EMOTION_CONTEXTS = [
    "അന്നത്തെ മുഖഭാവമാണ് സംഭവം മറക്കാനാവാത്തതാക്കിയത്",
    "കുറച്ച് നേരം കഴിഞ്ഞിട്ടും ആ അനുഭവം മനസ്സിൽ തന്നെ നിന്നു",
    "അവിടെ ഉണ്ടായിരുന്ന ശബ്ദം പോലും ആ സമയത്തെ പോലെ ഓർമ്മയുണ്ട്",
    "നിന്നോട് പറയുമ്പോഴാണ് അതിന്റെ ഭാരം ശരിക്കും മനസ്സിലാകുന്നത്",
    "ആ നിമിഷത്തിൽ എന്ത് പറയണം എന്ന് എനിക്ക് പിടികിട്ടിയില്ല",
    "ചുറ്റും ആളുകൾ ഉണ്ടായിരുന്നെങ്കിലും അത് വളരെ വ്യക്തിപരമായി തോന്നി",
    "പിന്നെ വീട്ടിലെത്തിയ ശേഷവും അതിനെക്കുറിച്ച് തന്നെ ചിന്തിച്ചു",
    "ഒന്ന് ശാന്തമായി ഇരുന്നാൽ മാത്രമേ ബാക്കി പറയാൻ പറ്റൂ",
    "അന്നേരം ചെറിയൊരു വാക്ക് പോലും വലിയതായി തോന്നി",
    "ഇത് കേട്ടാൽ നീ എന്ത് പറയും എന്ന് ഞാൻ ആലോചിച്ചു",
    "അവസാനം ആരോടും പറയാതെ ഇരിക്കാൻ കഴിഞ്ഞില്ല",
    "അത് നടന്ന സ്ഥലം പോലും ഇപ്പോൾ ഓർമ്മയിൽ തെളിഞ്ഞു നിൽക്കുന്നു",
    "പുറത്ത് നിന്ന് സാധാരണ പോലെ തോന്നിയെങ്കിലും ഉള്ളിൽ അങ്ങനെ ആയിരുന്നില്ല",
    "ഇത്രയും പെട്ടെന്ന് മനസ്സ് മാറുമെന്ന് ഞാൻ കരുതിയില്ല",
    "നമ്മൾ നേരത്തെ സംസാരിച്ച കാര്യം ഇതുമായി കൂടി ചേർന്നു",
    "ഒരുപാട് വാക്കുകൾ വേണ്ട, ആ സമയം തന്നെ എല്ലാം പറഞ്ഞു",
    "ഇനി ഇങ്ങനെ വന്നാൽ ഞാൻ കുറച്ച് വ്യത്യസ്തമായി കൈകാര്യം ചെയ്യും",
]


def parse_voice_list(value: Optional[str], defaults: List[str]) -> List[str]:
    if not value:
        return defaults
    voices = [item.strip() for item in value.split(",") if item.strip()]
    return voices or defaults


def clean_file_part(value: str) -> str:
    return value.replace("_", "-").replace(" ", "-")


def count_by(items: Iterable[Dict[str, str]], *keys: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        label = "/".join(str(item.get(key, "")) for key in keys)
        counts[label] = counts.get(label, 0) + 1
    return counts


def style_for_sample(category: str, emotion_type: str) -> str:
    if category == "emotion":
        return STYLE_BY_EMOTION[emotion_type]
    return STYLE_BY_CATEGORY[category]


def conversation_text(index: int) -> str:
    template = index % 5
    opener = CONV_OPENERS[index % len(CONV_OPENERS)]
    middle = CONV_MIDDLES[(index * 7 + 3) % len(CONV_MIDDLES)]
    ending = CONV_ENDINGS[(index * 11 + 5) % len(CONV_ENDINGS)]
    extra = CONV_MIDDLES[(index * 13 + 9) % len(CONV_MIDDLES)]
    if extra == middle:
        extra = CONV_MIDDLES[(index * 13 + 10) % len(CONV_MIDDLES)]
    side_note = CONV_SIDE_NOTES[(index * 17 + 2) % len(CONV_SIDE_NOTES)]
    if template == 0:
        text = f"{opener}. {middle}. {side_note}. {ending}"
    elif template == 1:
        text = f"{opener}. {extra}. {side_note}. {ending}"
    elif template == 2:
        text = f"{opener}. {middle}, പിന്നെ {extra[0].lower() + extra[1:]}. {side_note}. {ending}"
    elif template == 3:
        text = f"{opener}. {side_note}. {ending} {middle}."
    else:
        text = f"{opener}. {middle}. {extra}. {side_note}. {ending}"
    return normalize_indic_text(text)


def neutral_replay_text(index: int) -> str:
    topic = NEUTRAL_TOPICS[index % len(NEUTRAL_TOPICS)]
    detail = NEUTRAL_DETAILS[(index * 3 + index // len(NEUTRAL_TOPICS)) % len(NEUTRAL_DETAILS)]
    if index % 4 == 0:
        text = f"{topic}. {detail}"
    elif index % 4 == 1:
        text = f"{topic}; {detail}"
    elif index % 4 == 2:
        text = f"{topic}. ആവശ്യമുണ്ടെങ്കിൽ വീണ്ടും പറയാം. {detail}"
    else:
        text = f"{topic}. ആദ്യം ഇത് നോക്കൂ. {detail}"
    return normalize_indic_text(text)


def emotion_text(emotion_type: str, tag: str, index: int) -> str:
    base_items = EMOTION_BASES[emotion_type]
    base = base_items[index % len(base_items)]
    detail = EMOTION_DETAILS[(index * 5 + len(emotion_type)) % len(EMOTION_DETAILS)]
    context = EMOTION_CONTEXTS[(index * 7 + len(base_items)) % len(EMOTION_CONTEXTS)]
    if index % 3 == 0:
        spoken = f"{base}. {context}. {detail}"
    elif index % 3 == 1:
        spoken = f"{base}, അതുകൊണ്ട് കുറച്ച് സമയം എടുക്കാം. {context}. {detail}"
    else:
        spoken = f"{base}. ഞാൻ പറയുന്നത് ശാന്തമായി കേൾക്കൂ. {context}. {detail}"
    return normalize_indic_text(f"{tag} {spoken}" if tag else spoken)


def add_group(
    plan: List[Dict[str, str]],
    category: str,
    gender: str,
    count: int,
    voices: List[str],
    text_counter: Dict[str, int],
    emotion_type: str = "",
    tag: str = "",
):
    key = emotion_type or category
    for offset in range(count):
        local_index = text_counter[key]
        text_counter[key] += 1
        if category == "conversation":
            text = conversation_text(local_index)
        elif category == "neutral_replay":
            text = neutral_replay_text(local_index)
        else:
            text = emotion_text(emotion_type, tag, local_index)

        row_number = len(plan) + 1
        category_part = category if not emotion_type else f"{category}_{emotion_type}"
        file_id = f"synthetic_{row_number:04d}_{clean_file_part(category_part)}_{gender}"
        plan.append(
            {
                "id": file_id,
                "category": category,
                "emotion_type": emotion_type,
                "tag": tag,
                "gender": gender,
                "voice": voices[offset % len(voices)],
                "style": style_for_sample(category, emotion_type),
                "language_id": "ml",
                "text": text,
            }
        )


def build_plan(female_voices: List[str], male_voices: List[str]) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = []
    text_counter: Dict[str, int] = defaultdict(int)
    for gender, count in CONVERSATION_SPLIT.items():
        add_group(plan, "conversation", gender, count, female_voices if gender == "female" else male_voices, text_counter)
    for gender, count in NEUTRAL_REPLAY_SPLIT.items():
        add_group(plan, "neutral_replay", gender, count, female_voices if gender == "female" else male_voices, text_counter)
    for emotion_type, tag, female_count, male_count in EMOTION_SPLIT:
        add_group(plan, "emotion", "female", female_count, female_voices, text_counter, emotion_type, tag)
        add_group(plan, "emotion", "male", male_count, male_voices, text_counter, emotion_type, tag)
    validate_plan(plan)
    return plan


def malayalam_char_count(text: str) -> int:
    return len(MALAYALAM_RE.findall(text))


def validate_text(row: Dict[str, str], text: str, existing_texts: set) -> List[str]:
    issues = []
    text = normalize_indic_text(text)
    if not text:
        issues.append("empty")
    if "|" in text:
        issues.append("contains pipe delimiter")
    if text in existing_texts:
        issues.append("duplicate text")
    if malayalam_char_count(text) < 20:
        issues.append("too little Malayalam script")
    if len(text) < 45:
        issues.append("too short")
    if len(text) > 360:
        issues.append("too long")
    sentence_parts = [part.strip() for part in re.split(r"[.!?।]+", text) if part.strip()]
    if len(sentence_parts) != len(set(sentence_parts)):
        issues.append("repeats a sentence internally")

    found_tags = TAG_RE.findall(text)
    unsupported = [tag for tag in found_tags if tag not in SUPPORTED_TAGS]
    if unsupported:
        issues.append(f"unsupported tags: {unsupported}")

    required_tag = row.get("tag", "")
    if required_tag:
        if not text.startswith(required_tag):
            issues.append(f"must start with {required_tag}")
        if text.count(required_tag) != 1:
            issues.append(f"{required_tag} must occur exactly once")
    elif found_tags:
        issues.append("unexpected bracket tag")
    return issues


def validate_plan(plan: Sequence[Dict[str, str]]):
    if len(plan) != 800:
        raise AssertionError(f"Expected 800 rows, got {len(plan)}")
    category_counts = count_by(plan, "category")
    if category_counts != {"conversation": 400, "neutral_replay": 200, "emotion": 200}:
        raise AssertionError(f"Unexpected category split: {category_counts}")
    gender_counts = count_by(plan, "gender")
    if gender_counts != {"female": 400, "male": 400}:
        raise AssertionError(f"Unexpected gender split: {gender_counts}")

    texts = set()
    issues = {}
    for row in plan:
        row_issues = validate_text(row, row["text"], texts)
        if row_issues:
            issues[row["id"]] = row_issues
        texts.add(row["text"])
    if issues:
        preview = dict(list(issues.items())[:10])
        raise AssertionError(f"Text validation failed: {preview}")


def selected_range(total: int, start_index: int, limit: Optional[int]) -> List[int]:
    start = max(start_index, 0)
    end = total if limit is None else min(total, start + max(limit, 0))
    return list(range(start, end))


def write_dataset_files(output_dir: Path, plan: Sequence[Dict[str, str]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    manifest_path = output_dir / "manifest.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_handle, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_handle:
        writer = csv.writer(metadata_handle, delimiter="|", lineterminator="\n")
        for row in plan:
            transcript = normalize_indic_text(row["text"])
            full_row = dict(row)
            full_row["text"] = transcript
            full_row["tts_input"] = make_tts_input(full_row["style"], transcript)
            writer.writerow([full_row["id"], transcript, transcript, full_row["language_id"]])
            manifest_handle.write(json.dumps(full_row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> List[Dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_manifest(path: Path):
    rows = load_manifest(path)
    validate_plan(rows)
    print(f"Validated manifest rows: {len(rows)}")
    print(f"Unique texts: {len({row['text'] for row in rows})}")
    print(json.dumps(count_by(rows, "category"), ensure_ascii=False, indent=2))
    print(json.dumps(count_by(rows, "gender"), ensure_ascii=False, indent=2))


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


def generate_audio(
    output_dir: Path,
    api_key: str,
    tts_model: str,
    response_format: str,
    start_index: int,
    limit: Optional[int],
    skip_existing: bool,
    sleep_seconds: float,
):
    manifest_path = output_dir / "manifest.jsonl"
    validate_manifest(manifest_path)
    rows = load_manifest(manifest_path)
    selected_indexes = selected_range(len(rows), start_index, limit)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    selected = [(index, rows[index]) for index in selected_indexes]
    for position, (index, row) in enumerate(selected, start=1):
        wav_path = wav_dir / f"{row['id']}.wav"
        if skip_existing and wav_path.exists() and wav_path.stat().st_size > 0:
            print(f"[{position}/{len(selected)}] skip existing {row['id']}")
            continue
        print(
            f"[{position}/{len(selected)}] index={index} id={row['id']} "
            f"gender={row['gender']} voice={row['voice']} category={row['category']} emotion={row['emotion_type'] or '-'}"
        )
        audio_bytes = request_speech(api_key, tts_model, row["voice"], row["tts_input"], response_format)
        save_audio_bytes(wav_path, audio_bytes, response_format)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    print(f"Done. WAVs: {wav_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create and generate the 800-sample Malayalam synthetic continuation dataset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=["plan", "audio", "all", "validate"], default="plan")
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--female-voices", default=",".join(DEFAULT_FEMALE_VOICES))
    parser.add_argument("--male-voices", default=",".join(DEFAULT_MALE_VOICES))
    parser.add_argument("--response-format", choices=["pcm", "mp3"], default="pcm")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based row index for audio generation.")
    parser.add_argument("--limit", type=int, default=None, help="Generate audio for only this many rows from start-index.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each generated audio sample.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    female_voices = parse_voice_list(args.female_voices, DEFAULT_FEMALE_VOICES)
    male_voices = parse_voice_list(args.male_voices, DEFAULT_MALE_VOICES)
    plan = build_plan(female_voices, male_voices)

    print("Target split:")
    print(json.dumps(count_by(plan, "category"), ensure_ascii=False, indent=2))
    print(json.dumps(count_by(plan, "gender"), ensure_ascii=False, indent=2))
    print(f"Unique texts: {len({row['text'] for row in plan})}")

    if args.stage in {"plan", "all"}:
        write_dataset_files(output_dir, plan)
        print(f"Text manifest written: {output_dir / 'manifest.jsonl'}")
        print(f"Training metadata written: {output_dir / 'metadata.csv'}")

    if args.stage == "validate":
        validate_manifest(output_dir / "manifest.jsonl")
        return

    if args.stage in {"audio", "all"}:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("Set OPENROUTER_API_KEY before running audio generation.")
        generate_audio(
            output_dir,
            api_key,
            args.tts_model,
            args.response_format,
            args.start_index,
            args.limit,
            args.skip_existing,
            args.sleep,
        )


if __name__ == "__main__":
    main()
