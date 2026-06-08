import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.generate_openrouter_pilot_dataset import make_tts_input
from IndicFinetuning.indic_text import normalize_indic_text


DEFAULT_OUTPUT = "./IndicFinetuning/datasets/OpenRouterManualReview50"
SUPPORTED_TAGS = {"[laughter]", "[giggle]", "[sigh]", "[cry]", "[whisper]", "[cough]"}
TAG_RE = re.compile(r"\[[^\]]+\]")
WORD_RE = re.compile(r"[\u0D00-\u0D7F]+")


STYLE_BY_CATEGORY = {
    "conversation": "Speak in natural everyday Malayalam, casual and human, not like a newsreader.",
    "neutral_replay": "Speak in plain neutral Malayalam with clear pronunciation, steady pacing, and no drama.",
    "emotion": "Speak Malayalam in the requested emotional style while keeping the words clean and understandable.",
}


MANUAL_ROWS: List[Dict[str, str]] = [
    {
        "id": "manual_0001_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Callirrhoe",
        "emotion_type": "",
        "tag": "",
        "text": "ഇന്ന് രാവിലെ അമ്മ ചായ കൊണ്ടുവന്നപ്പോൾ ഞാൻ ഇപ്പോഴും കിടക്കയിൽ തന്നെ ആയിരുന്നു. അവൾ ഒന്നും പറയാതെ ജനൽ തുറന്നു, മുറിയിലെ കാറ്റ് മാറിയതോടെ എഴുന്നേൽക്കാൻ മനസ്സ് വന്നു.",
    },
    {
        "id": "manual_0002_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Puck",
        "emotion_type": "",
        "tag": "",
        "text": "ബസിൽ ഇന്ന് ടിക്കറ്റ് എടുക്കാൻ കാത്തുനിൽക്കുമ്പോൾ മുന്നിലെ കുട്ടി പണം എണ്ണിക്കൊണ്ട് കുഴങ്ങി. കണ്ടക്ടർ ചിരിച്ച് കാത്തുനിന്നതുകൊണ്ട് ആരും അസ്വസ്ഥരായില്ല.",
    },
    {
        "id": "manual_0003_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Aoede",
        "emotion_type": "",
        "tag": "",
        "text": "ഉച്ചയ്ക്ക് ഓഫീസിലെ ഭക്ഷണം ഇഷ്ടമായില്ല, അതിനാൽ ഞാൻ പുറത്തുനിന്ന് ഒരു ചെറിയ പൊറോട്ട വാങ്ങി. കൂടെ കിട്ടിയ കറിയാണ് ദിവസത്തെ ഏറ്റവും നല്ല ഭാഗം ആയി തോന്നിയത്.",
    },
    {
        "id": "manual_0004_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Charon",
        "emotion_type": "",
        "tag": "",
        "text": "കടയിൽ പോയപ്പോൾ പഞ്ചസാര വാങ്ങാനാണ് വിചാരിച്ചത്, പക്ഷേ വീട്ടിലെത്തിയപ്പോൾ ബാഗിൽ സോപ്പും ബിസ്കറ്റും മാത്രം. അപ്പോൾ മാത്രമാണ് ഞാൻ ലിസ്റ്റ് നോക്കാതിരുന്നതു മനസ്സിലായത്.",
    },
    {
        "id": "manual_0005_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Kore",
        "emotion_type": "",
        "tag": "",
        "text": "വൈകുന്നേരം മഴ നിർത്തിയതിന് ശേഷം മുറ്റത്ത് നിന്നപ്പോൾ മണ്ണിന്റെ മണം നല്ലതായിരുന്നു. അച്ഛൻ ചെടികൾ നോക്കിക്കൊണ്ട് ഇന്നത്തെ വാർത്തകൾ പതുക്കെ പറഞ്ഞു.",
    },
    {
        "id": "manual_0006_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Fenrir",
        "emotion_type": "",
        "tag": "",
        "text": "ലാപ്ടോപ്പ് ചാർജ് ഇല്ലാതെ ഓഫ് ആയപ്പോൾ ആദ്യം എനിക്ക് കോപം വന്നു. പിന്നെ ചായ കുടിച്ച് ഇരുന്നപ്പോൾ ജോലി വീണ്ടും തുടങ്ങാൻ പറ്റുന്നത്ര ശാന്തമായി.",
    },
    {
        "id": "manual_0007_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Despina",
        "emotion_type": "",
        "tag": "",
        "text": "പഴയ സ്കൂൾ ബാഗ് അലമാരയിൽ നിന്ന് കിട്ടി. അതിന്റെ ചെറിയ പോക്കറ്റിൽ ഞാൻ മറന്നുവച്ചിരുന്ന നീല പേന കണ്ടപ്പോൾ മുഴുവൻ ക്ലാസ് മുറിയും ഓർമ്മ വന്നു.",
    },
    {
        "id": "manual_0008_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Orus",
        "emotion_type": "",
        "tag": "",
        "text": "ഇന്ന് ജിം പോകാമെന്ന് രാവിലെ പറഞ്ഞ ഞാൻ വൈകുന്നേരം നേരെ വീട്ടിലേക്ക് വന്നു. ഷൂ കെട്ടിയിരിക്കുമ്പോൾ തന്നെ ഉറക്കം ജയിച്ചുപോയി.",
    },
    {
        "id": "manual_0009_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Laomedeia",
        "emotion_type": "",
        "tag": "",
        "text": "വീട്ടിലെ കുഞ്ഞ് പുതിയ വാക്ക് പഠിച്ചിട്ട് എല്ലാവരോടും അതേ വാക്ക് ആവർത്തിച്ചു. ആദ്യം രസമായിരുന്നു, പിന്നെ അത് മുഴുവൻ വീട്ടിന്റെ പാട്ടായി മാറി.",
    },
    {
        "id": "manual_0010_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Iapetus",
        "emotion_type": "",
        "tag": "",
        "text": "ഓട്ടോ ഡ്രൈവർ ഇന്നത്തെ ട്രാഫിക്കിനെക്കുറിച്ച് സംസാരിച്ചുകൊണ്ടേയിരുന്നു. ഞാൻ മറുപടി കുറച്ചേ പറഞ്ഞുള്ളൂ, പക്ഷേ അവന്റെ കഥകൾ യാത്ര ചെറുതാക്കി.",
    },
    {
        "id": "manual_0011_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Leda",
        "emotion_type": "",
        "tag": "",
        "text": "അടുക്കളയിൽ ഇന്ന് ഉപ്പ് കുറച്ച് കൂടുതലായി പോയി. വീട്ടുകാർ ഒന്നും പറഞ്ഞില്ലെങ്കിലും എല്ലാവരും വെള്ളം അധികം കുടിക്കുന്നത് കണ്ടപ്പോൾ കാര്യം മനസ്സിലായി.",
    },
    {
        "id": "manual_0012_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Algenib",
        "emotion_type": "",
        "tag": "",
        "text": "രാവിലെ ഫോൺ സൈലന്റിൽ ആയതിനാൽ മൂന്ന് മിസ് കോൾ ഞാൻ കണ്ടില്ല. പിന്നെ തിരിച്ച് വിളിച്ചപ്പോൾ കാര്യം അത്ര അടിയന്തരമല്ലെന്ന് അറിഞ്ഞു.",
    },
    {
        "id": "manual_0013_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Callirrhoe",
        "emotion_type": "",
        "tag": "",
        "text": "വഴിയിൽ പഴയ അയൽക്കാരിയെ കണ്ടപ്പോൾ അവൾ എന്നെ ഉടനെ തിരിച്ചറിഞ്ഞു. വർഷങ്ങൾ കഴിഞ്ഞിട്ടും പേരുപറഞ്ഞത് കേട്ട് ഞാൻ സന്തോഷിച്ചു.",
    },
    {
        "id": "manual_0014_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Puck",
        "emotion_type": "",
        "tag": "",
        "text": "മീറ്റിംഗ് തുടങ്ങുന്നതിന് മുമ്പ് ഞാൻ കുറിപ്പുകൾ നോക്കിക്കൊണ്ടിരുന്നു. പക്ഷേ സംസാരിക്കാൻ തുടങ്ങിയപ്പോൾ പേപ്പറിൽ എഴുതിയതിനെക്കാൾ മനസ്സിൽ വന്നത് നന്നായി പ്രവർത്തിച്ചു.",
    },
    {
        "id": "manual_0015_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Aoede",
        "emotion_type": "",
        "tag": "",
        "text": "ഇന്ന് വൈകുന്നേരം ഫോൺ വച്ച് കുറച്ച് നേരം ഒന്നും ചെയ്യാതെ ഇരുന്നു. ആ ശാന്തത ആദ്യം വിചിത്രമായിരുന്നു, പിന്നെ അതാണ് ആവശ്യമെന്ന് തോന്നി.",
    },
    {
        "id": "manual_0016_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Charon",
        "emotion_type": "",
        "tag": "",
        "text": "പുതിയ സിനിമ കാണാൻ കൂട്ടുകാർ വിളിച്ചു, പക്ഷേ ഞാൻ കഥ പോലും അറിയാതെ പോയി. അവസാനം അവരുടെ പ്രതികരണങ്ങൾ കാണുന്നതാണ് സിനിമയേക്കാൾ രസകരമായത്.",
    },
    {
        "id": "manual_0017_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Kore",
        "emotion_type": "",
        "tag": "",
        "text": "ബേക്കറിയിൽ നിന്ന് വാങ്ങിയ കേക്ക് വീട്ടിലെത്തും മുമ്പ് അല്പം കുഴഞ്ഞുപോയി. എന്നാലും എല്ലാവരും ചേർന്ന് കഴിച്ചപ്പോൾ അതിന് വേറൊരു രസം ഉണ്ടായിരുന്നു.",
    },
    {
        "id": "manual_0018_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Fenrir",
        "emotion_type": "",
        "tag": "",
        "text": "വീട്ടിലെ ഫാൻ ശബ്ദം ചെയ്തതുകൊണ്ട് ഞാൻ മുഴുവൻ രാത്രി ഉറക്കം ശരിയാക്കാനായില്ല. രാവിലെ അത് ഓഫ് ചെയ്തപ്പോൾ മുറി പെട്ടെന്ന് ശാന്തമായി.",
    },
    {
        "id": "manual_0019_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Despina",
        "emotion_type": "",
        "tag": "",
        "text": "ഇന്ന് ബാഗിൽ കുടയുണ്ടായിരുന്നു, പക്ഷേ മഴ വന്നത് ഞാൻ അകത്ത് കയറിയതിന് ശേഷമാണ്. അതിനാൽ തയ്യാറെടുപ്പ് വെറുതെയായില്ലെന്ന് തോന്നി.",
    },
    {
        "id": "manual_0020_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Orus",
        "emotion_type": "",
        "tag": "",
        "text": "കടൽക്കരയിൽ നിന്നപ്പോൾ തിരമാലയുടെ ശബ്ദം മാത്രം കേൾക്കാമായിരുന്നു. സാധാരണ സംസാരിക്കാത്ത സുഹൃത്ത് അന്ന് വളരെ തുറന്ന് സംസാരിച്ചു.",
    },
    {
        "id": "manual_0021_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Laomedeia",
        "emotion_type": "",
        "tag": "",
        "text": "വീട്ടിലെ പൂച്ച ഇന്ന് എന്റെ പുസ്തകത്തിന്മേൽ കിടന്നതിനാൽ വായന നിർത്തേണ്ടി വന്നു. അതിനെ മാറ്റാൻ നോക്കിയപ്പോൾ അത് കൂടുതൽ സുഖമായി കിടന്നു.",
    },
    {
        "id": "manual_0022_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Iapetus",
        "emotion_type": "",
        "tag": "",
        "text": "ഇന്ന് രാവിലെ ഷർട്ടിൽ ബട്ടൺ പൊട്ടിയതുകൊണ്ട് അവസാന നിമിഷം വസ്ത്രം മാറ്റി. ചെറിയ കാര്യമാണെങ്കിലും സമയം മുഴുവൻ അതിൽ പോയി.",
    },
    {
        "id": "manual_0023_conversation_female",
        "category": "conversation",
        "gender": "female",
        "voice": "Leda",
        "emotion_type": "",
        "tag": "",
        "text": "വീഡിയോ കോൾ ചെയ്യുമ്പോൾ അമ്മമ്മയ്ക്ക് ക്യാമറ എവിടെയെന്ന് പിടികിട്ടിയില്ല. സ്ക്രീനിൽ അവളുടെ നെറ്റി മാത്രം കണ്ടതോടെ ഞങ്ങൾ എല്ലാവരും ചിരിച്ചു.",
    },
    {
        "id": "manual_0024_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Algenib",
        "emotion_type": "",
        "tag": "",
        "text": "പുതിയ വഴിയിലൂടെ ഓഫീസിലേക്ക് പോയപ്പോൾ ആദ്യം സംശയമായി. പക്ഷേ ചുറ്റുമുള്ള കടകൾ കണ്ടപ്പോൾ ആ പ്രദേശം പതുക്കെ പരിചിതമായി.",
    },
    {
        "id": "manual_0025_conversation_male",
        "category": "conversation",
        "gender": "male",
        "voice": "Algenib",
        "emotion_type": "",
        "tag": "",
        "text": "ചെറിയൊരു തർക്കം ഉണ്ടായിരുന്നെങ്കിലും രാത്രി എല്ലാവരും കൂടെ ഭക്ഷണം കഴിച്ചു. ചിലപ്പോൾ ഭക്ഷണ മേശയിലാണ് വീട്ടിലെ പ്രശ്നങ്ങൾ ശാന്തമാകുന്നത്.",
    },
    {
        "id": "manual_0026_neutral_female",
        "category": "neutral_replay",
        "gender": "female",
        "voice": "Aoede",
        "emotion_type": "",
        "tag": "",
        "text": "നാളെ പത്ത് മണിക്ക് വിളിച്ചാൽ മതി. ഞാൻ അതിന് മുമ്പ് രേഖകൾ പരിശോധിച്ച് വെക്കാം.",
    },
    {
        "id": "manual_0027_neutral_male",
        "category": "neutral_replay",
        "gender": "male",
        "voice": "Puck",
        "emotion_type": "",
        "tag": "",
        "text": "നിങ്ങൾ അയച്ച വിലാസം ശരിയാണ്. അവിടെ എത്തുമ്പോൾ റിസപ്ഷനിൽ പേര് പറഞ്ഞാൽ അവർ വഴി കാണിക്കും.",
    },
    {
        "id": "manual_0028_neutral_female",
        "category": "neutral_replay",
        "gender": "female",
        "voice": "Kore",
        "emotion_type": "",
        "tag": "",
        "text": "ഇപ്പോൾ മാറ്റം വരുത്തേണ്ടതില്ല. ആദ്യം നിലവിലുള്ള പതിപ്പ് ഉപയോഗിച്ച് നോക്കൂ.",
    },
    {
        "id": "manual_0029_neutral_male",
        "category": "neutral_replay",
        "gender": "male",
        "voice": "Charon",
        "emotion_type": "",
        "tag": "",
        "text": "മരുന്ന് ഭക്ഷണത്തിന് ശേഷം കഴിക്കണം. എന്തെങ്കിലും അസ്വസ്ഥത തോന്നിയാൽ ഉടനെ ഡോക്ടറെ അറിയിക്കുക.",
    },
    {
        "id": "manual_0030_neutral_female",
        "category": "neutral_replay",
        "gender": "female",
        "voice": "Despina",
        "emotion_type": "",
        "tag": "",
        "text": "അടുത്ത ആഴ്ച സമയക്രമം മാറും. പുതുക്കിയ പട്ടിക വന്നാൽ ഞാൻ ഗ്രൂപ്പിൽ അയക്കാം.",
    },
    {
        "id": "manual_0031_neutral_male",
        "category": "neutral_replay",
        "gender": "male",
        "voice": "Fenrir",
        "emotion_type": "",
        "tag": "",
        "text": "പേയ്‌മെന്റ് പൂർത്തിയായിട്ടുണ്ട്. രസീത് നിങ്ങളുടെ ഇമെയിലിലേക്ക് കുറച്ച് സമയത്തിനകം വരും.",
    },
    {
        "id": "manual_0032_neutral_female",
        "category": "neutral_replay",
        "gender": "female",
        "voice": "Laomedeia",
        "emotion_type": "",
        "tag": "",
        "text": "ക്ലാസ് തുടങ്ങുന്നതിന് മുമ്പ് ഹാജർ രേഖപ്പെടുത്തണം. വൈകിയാൽ ഓഫീസിൽ അറിയിച്ച ശേഷം അകത്ത് കയറുക.",
    },
    {
        "id": "manual_0033_neutral_male",
        "category": "neutral_replay",
        "gender": "male",
        "voice": "Orus",
        "emotion_type": "",
        "tag": "",
        "text": "ഫയൽ തുറക്കാത്ത പക്ഷം വീണ്ടും ഡൗൺലോഡ് ചെയ്ത് നോക്കൂ. അതിനുശേഷവും പ്രശ്നമുണ്ടെങ്കിൽ സ്ക്രീൻഷോട്ട് അയക്കുക.",
    },
    {
        "id": "manual_0034_neutral_female",
        "category": "neutral_replay",
        "gender": "female",
        "voice": "Leda",
        "emotion_type": "",
        "tag": "",
        "text": "ഇന്ന് വൈകുന്നേരത്തെ പരിപാടി അകത്തെ ഹാളിലാണ്. പ്രവേശന കവാടത്തിൽ ടിക്കറ്റ് കാണിച്ചാൽ മതി.",
    },
    {
        "id": "manual_0035_neutral_male",
        "category": "neutral_replay",
        "gender": "male",
        "voice": "Iapetus",
        "emotion_type": "",
        "tag": "",
        "text": "ബുക്കിംഗ് മാറ്റണമെങ്കിൽ ഇന്നുതന്നെ അറിയിക്കുക. നാളെ കഴിഞ്ഞാൽ പുതിയ നിരക്ക് ബാധകമാകും.",
    },
    {
        "id": "manual_0036_neutral_female",
        "category": "neutral_replay",
        "gender": "female",
        "voice": "Callirrhoe",
        "emotion_type": "",
        "tag": "",
        "text": "നിങ്ങളുടെ അപേക്ഷ ലഭിച്ചിട്ടുണ്ട്. പരിശോധിച്ചതിന് ശേഷം അടുത്ത ഘട്ടം സന്ദേശമായി അറിയിക്കും.",
    },
    {
        "id": "manual_0037_neutral_male",
        "category": "neutral_replay",
        "gender": "male",
        "voice": "Algenib",
        "emotion_type": "",
        "tag": "",
        "text": "വൈകുന്നേരം മഴ വരാൻ സാധ്യതയുണ്ട്. പുറത്തേക്ക് പോകുന്നെങ്കിൽ ഒരു കുട കൈയിൽ വെക്കുന്നത് നല്ലതാണ്.",
    },
    {
        "id": "manual_0038_emotion_female_laughter",
        "category": "emotion",
        "gender": "female",
        "voice": "Aoede",
        "emotion_type": "laughter",
        "tag": "[laughter]",
        "text": "[laughter] അവൾ അത്ര ഗൗരവത്തോടെ രഹസ്യം പറയാൻ വന്നതാണ്, പക്ഷേ അവസാനം സ്വന്തം ഫോൺ തന്നെ എവിടെ വെച്ചെന്ന് മറന്നു.",
    },
    {
        "id": "manual_0039_emotion_male_laughter",
        "category": "emotion",
        "gender": "male",
        "voice": "Puck",
        "emotion_type": "laughter",
        "tag": "[laughter]",
        "text": "[laughter] ഞാൻ വഴി അറിയാമെന്ന് ആത്മവിശ്വാസത്തോടെ പറഞ്ഞു, പിന്നെ മാപ്പ് തുറന്നപ്പോൾ നമ്മൾ മറുവശത്തേക്ക് പോയെന്ന് മനസ്സിലായി.",
    },
    {
        "id": "manual_0040_emotion_female_giggle",
        "category": "emotion",
        "gender": "female",
        "voice": "Kore",
        "emotion_type": "giggle",
        "tag": "[giggle]",
        "text": "[giggle] നീ പറഞ്ഞ compliment കേട്ടപ്പോൾ എനിക്ക് മറുപടി കിട്ടിയില്ല. അതുകൊണ്ട് ഞാൻ ചിരിച്ച് വിഷയം മാറ്റി.",
    },
    {
        "id": "manual_0041_emotion_male_giggle",
        "category": "emotion",
        "gender": "male",
        "voice": "Charon",
        "emotion_type": "giggle",
        "tag": "[giggle]",
        "text": "[giggle] ആ കുട്ടി എന്നെ സാർ എന്ന് വിളിച്ചപ്പോൾ ഞാൻ പെട്ടെന്ന് പ്രായം കൂടിയ ആളായി തോന്നി.",
    },
    {
        "id": "manual_0042_emotion_female_whisper",
        "category": "emotion",
        "gender": "female",
        "voice": "Despina",
        "emotion_type": "whisper",
        "tag": "[whisper]",
        "text": "[whisper] അവൾക്കായി വാങ്ങിയ സമ്മാനം അലമാരയുടെ മുകളിലാണ്. ഇപ്പോൾ പറയരുത്, സർപ്രൈസ് പൊളിയും.",
    },
    {
        "id": "manual_0043_emotion_male_whisper",
        "category": "emotion",
        "gender": "male",
        "voice": "Fenrir",
        "emotion_type": "whisper",
        "tag": "[whisper]",
        "text": "[whisper] കുഞ്ഞ് ഉറങ്ങുകയാണ്, അതിനാൽ വാതിൽ പതുക്കെ അടയ്ക്കണം. ബാഗ് ഇവിടെ വെച്ചാൽ മതി.",
    },
    {
        "id": "manual_0044_emotion_female_cry",
        "category": "emotion",
        "gender": "female",
        "voice": "Laomedeia",
        "emotion_type": "cry",
        "tag": "[cry]",
        "text": "[cry] ആ പഴയ ശബ്ദസന്ദേശം വീണ്ടും കേട്ടപ്പോൾ എനിക്ക് സംസാരിക്കാൻ കഴിഞ്ഞില്ല. അവൻ ചിരിച്ച രീതിയാണ് ഏറ്റവും കൂടുതൽ ഓർമ്മ വന്നത്.",
    },
    {
        "id": "manual_0045_emotion_male_cry",
        "category": "emotion",
        "gender": "male",
        "voice": "Orus",
        "emotion_type": "cry",
        "tag": "[cry]",
        "text": "[cry] വീട്ടിലേക്ക് തിരിച്ച് വരുമ്പോൾ ഒഴിഞ്ഞ കസേര കണ്ടത് സഹിക്കാൻ ബുദ്ധിമുട്ടായി. എല്ലാവരും മിണ്ടാതെ നിന്നു.",
    },
    {
        "id": "manual_0046_emotion_female_sigh_tired",
        "category": "emotion",
        "gender": "female",
        "voice": "Leda",
        "emotion_type": "sigh_frustration_tired",
        "tag": "[sigh]",
        "text": "[sigh] രാവിലെ മുതൽ ഈ ഫോം ശരിയാക്കാൻ നോക്കുകയാണ്. ഓരോ തവണയും വേറൊരു പിശക് കാണിക്കുന്നത് ക്ഷീണമാക്കുന്നു.",
    },
    {
        "id": "manual_0047_emotion_male_sigh_tired",
        "category": "emotion",
        "gender": "male",
        "voice": "Iapetus",
        "emotion_type": "sigh_frustration_tired",
        "tag": "[sigh]",
        "text": "[sigh] ഇന്ന് ഒന്ന് നേരത്തെ വീട്ടിൽ എത്തണമെന്നായിരുന്നു ആഗ്രഹം. പക്ഷേ അവസാന നിമിഷം വീണ്ടും ജോലി കൂടി.",
    },
    {
        "id": "manual_0048_emotion_female_sigh_nervous",
        "category": "emotion",
        "gender": "female",
        "voice": "Callirrhoe",
        "emotion_type": "sigh_nervous_uncertain",
        "tag": "[sigh]",
        "text": "[sigh] ഇത് പറയുന്നത് ശരിയാണോ എന്നറിയില്ല. പക്ഷേ ഞാൻ മിണ്ടാതെ ഇരുന്നാൽ കാര്യങ്ങൾ കൂടുതൽ കുഴയുമെന്ന് തോന്നുന്നു.",
    },
    {
        "id": "manual_0049_emotion_male_positive",
        "category": "emotion",
        "gender": "male",
        "voice": "Algenib",
        "emotion_type": "positive_excited",
        "tag": "",
        "text": "നമ്മൾ അയച്ച പ്രൊപ്പോസൽ അവർ അംഗീകരിച്ചു. ഇനി അടുത്ത ഘട്ടം തുടങ്ങാൻ പറ്റുമെന്ന് കേട്ടപ്പോൾ എനിക്ക് ശരിക്കും ആവേശമായി.",
    },
    {
        "id": "manual_0050_emotion_female_curious",
        "category": "emotion",
        "gender": "female",
        "voice": "Aoede",
        "emotion_type": "curious_confused",
        "tag": "",
        "text": "അവൻ ആദ്യം സമ്മതിച്ച കാര്യം പിന്നെ മാറ്റിയത് എന്തുകൊണ്ടാണ്? ഇടയിൽ ആരെങ്കിലും മറ്റൊരു വിവരം പറഞ്ഞിട്ടുണ്ടാകുമോ?",
    },
]


MANUAL_ROWS.extend(
    [
        {
            "id": "manual_0051_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Kore",
            "emotion_type": "",
            "tag": "",
            "text": "ജിമ്മിൽ ഇന്ന് ആദ്യമായി ട്രെയിനർ എന്റെ ഫോം ശരിയാക്കി. ചെറിയ മാറ്റം മാത്രമായിരുന്നു, പക്ഷേ ഭാരം ഉയർത്തുമ്പോൾ മുഴുവൻ വ്യത്യാസവും ഉടനെ മനസ്സിലായി.",
        },
        {
            "id": "manual_0052_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Puck",
            "emotion_type": "",
            "tag": "",
            "text": "ബാങ്കിൽ ടോക്കൺ എടുത്തപ്പോൾ എന്റെ മുന്നിൽ പന്ത്രണ്ട് പേർ ഉണ്ടായിരുന്നു. പക്ഷേ കൗണ്ടറിലെ ഉദ്യോഗസ്ഥൻ വേഗത്തിൽ കാര്യങ്ങൾ തീർത്തതുകൊണ്ട് ഞാൻ കരുതിയതിനെക്കാൾ പെട്ടെന്ന് ഇറങ്ങി.",
        },
        {
            "id": "manual_0053_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Aoede",
            "emotion_type": "",
            "tag": "",
            "text": "പാർക്കിൽ നടക്കുമ്പോൾ ഒരു ചെറിയ പെൺകുട്ടി സൈക്കിൾ പഠിക്കാൻ ശ്രമിക്കുന്നത് കണ്ടു. ഓരോ തവണ വീഴുമ്പോഴും അവൾ വീണ്ടും എഴുന്നേൽക്കുന്നത് കാണാൻ മനോഹരമായിരുന്നു.",
        },
        {
            "id": "manual_0054_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Charon",
            "emotion_type": "",
            "tag": "",
            "text": "സ്കൂളിലെ രക്ഷിതാക്കളുടെ മീറ്റിംഗിന് പോയപ്പോൾ പഴയ ബെഞ്ചുകളുടെ മണം തന്നെ ആദ്യം ശ്രദ്ധയിൽപ്പെട്ടു. അധ്യാപിക കുട്ടിയെക്കുറിച്ച് പറഞ്ഞ ചെറിയ പുരോഗതി കേട്ട് ആശ്വാസമായി.",
        },
        {
            "id": "manual_0055_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Despina",
            "emotion_type": "",
            "tag": "",
            "text": "ആശുപത്രിയിലെ കാത്തിരിപ്പ് മുറിയിൽ എല്ലാവരും സ്വന്തം ചിന്തകളിൽ ആയിരുന്നു. ഒരു നഴ്സ് ചിരിച്ച് വെള്ളം കൊണ്ടുവന്നപ്പോൾ ആ ഇടം അല്പം മൃദുവായി തോന്നി.",
        },
        {
            "id": "manual_0056_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Fenrir",
            "emotion_type": "",
            "tag": "",
            "text": "ട്രെയിൻ വൈകിയതുകൊണ്ട് പ്ലാറ്റ്ഫോമിൽ കൂടുതൽ സമയം നിൽക്കേണ്ടിവന്നു. അടുത്ത് നിന്നിരുന്ന വയസ്സനായ ആളുമായി സംസാരിച്ചപ്പോൾ യാത്രയുടെ ബോറടിപ്പ് കുറഞ്ഞു.",
        },
        {
            "id": "manual_0057_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Laomedeia",
            "emotion_type": "",
            "tag": "",
            "text": "അപ്പാർട്ട്മെന്റിലെ ലിഫ്റ്റ് വീണ്ടും പ്രവർത്തിക്കാതിരുന്നതിനാൽ നാലാം നില വരെ പടികൾ കയറി. മുകളിലെത്തിയപ്പോൾ വ്യായാമം കഴിഞ്ഞതുപോലെ ശ്വാസം മുട്ടി.",
        },
        {
            "id": "manual_0058_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Orus",
            "emotion_type": "",
            "tag": "",
            "text": "മത്സ്യ മാർക്കറ്റിൽ രാവിലെ പോയാൽ വില കുറവായിരിക്കും എന്ന് അച്ഛൻ പറഞ്ഞു. പക്ഷേ അവിടെ എത്തുമ്പോഴേക്കും എനിക്ക് ആദ്യം തോന്നിയത് ശബ്ദത്തിന്റെ തിരക്കായിരുന്നു.",
        },
        {
            "id": "manual_0059_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Leda",
            "emotion_type": "",
            "tag": "",
            "text": "ലൈബ്രറിയിൽ പുസ്തകം തിരികെ നൽകാൻ പോയപ്പോൾ പിഴ അടയ്ക്കേണ്ടിവരും എന്ന് കരുതി. ലൈബ്രേറിയൻ തീയതി നീട്ടിയിരുന്നെന്ന് പറഞ്ഞപ്പോൾ വലിയ ആശ്വാസമായി.",
        },
        {
            "id": "manual_0060_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Iapetus",
            "emotion_type": "",
            "tag": "",
            "text": "വണ്ടിയുടെ സർവീസിന് പോയപ്പോൾ മെക്കാനിക്ക് പഴയ ഭാഗം കാണിച്ചുകൊണ്ട് പ്രശ്നം വിശദീകരിച്ചു. അതുവരെ ഞാൻ കേട്ട ശബ്ദം എന്താണെന്ന് മനസ്സിലായിരുന്നില്ല.",
        },
        {
            "id": "manual_0061_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Callirrhoe",
            "emotion_type": "",
            "tag": "",
            "text": "ഡാൻസ് ക്ലാസിൽ പുതിയ ചുവട് പഠിക്കുമ്പോൾ എല്ലാവരും ഒരുമിച്ച് തെറ്റിച്ചു. ടീച്ചർ പോലും ചിരിച്ചതോടെ പേടി മാറി വീണ്ടും ശ്രമിക്കാൻ തോന്നി.",
        },
        {
            "id": "manual_0062_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Algenib",
            "emotion_type": "",
            "tag": "",
            "text": "പഞ്ചായത്ത് ഓഫീസിൽ സർട്ടിഫിക്കറ്റ് വാങ്ങാൻ പോയത് നീണ്ടുപോകുമെന്ന് കരുതി. പക്ഷേ ശരിയായ രേഖകൾ കൈയിൽ ഉണ്ടായതുകൊണ്ട് കാര്യങ്ങൾ സുലഭമായി തീർന്നു.",
        },
        {
            "id": "manual_0063_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Kore",
            "emotion_type": "",
            "tag": "",
            "text": "ബ്യൂട്ടി പാർലറിൽ കാത്തിരിക്കുമ്പോൾ മുന്നിലെ സ്ത്രീ തന്റെ മകളുടെ വിവാഹ ഒരുക്കങ്ങൾ പറഞ്ഞു. അന്യരോട് പോലും ചിലർ എത്ര തുറന്ന് സംസാരിക്കും എന്ന് തോന്നി.",
        },
        {
            "id": "manual_0064_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Puck",
            "emotion_type": "",
            "tag": "",
            "text": "ഫുട്ബോൾ കളിക്കാൻ പോയപ്പോൾ മൈതാനം മുഴുവൻ ചെളിയായിരുന്നു. പന്ത് നിയന്ത്രിക്കാൻ ബുദ്ധിമുട്ടായെങ്കിലും എല്ലാവരും അതിനെയാണ് കളിയുടെ രസമായി എടുത്തത്.",
        },
        {
            "id": "manual_0065_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Aoede",
            "emotion_type": "",
            "tag": "",
            "text": "പച്ചക്കറി കടയിൽ ഇന്ന് തക്കാളിക്ക് വില കൂടുതലായിരുന്നു. കടക്കാരൻ ചിരിച്ചുകൊണ്ട് രണ്ടെണ്ണം കൂടി ചേർത്തപ്പോൾ ചെറുതായി സന്തോഷമായി.",
        },
        {
            "id": "manual_0066_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Charon",
            "emotion_type": "",
            "tag": "",
            "text": "കുട്ടിയുടെ സ്കൂൾ പ്രോജക്റ്റിന് കാർഡ്ബോർഡ് മുറിക്കാനാണ് ഞാൻ ഇരുന്നത്. അവസാനം ഒട്ടും നേരായില്ലെങ്കിലും അവൻ അതിനെ റോക്കറ്റിന്റെ ഡിസൈൻ എന്ന് വിളിച്ചു.",
        },
        {
            "id": "manual_0067_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Despina",
            "emotion_type": "",
            "tag": "",
            "text": "മെട്രോയിൽ ഒരു സീറ്റ് ഒഴിഞ്ഞപ്പോൾ മുന്നിലെ അമ്മച്ചിക്ക് ഞാൻ കൈ കാണിച്ചു. അവൾ ഇരുന്ന ശേഷം പറഞ്ഞ നന്ദി മുഴുവൻ യാത്രയെയും നല്ലതാക്കി.",
        },
        {
            "id": "manual_0068_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Fenrir",
            "emotion_type": "",
            "tag": "",
            "text": "ചെറിയ കടയിൽ കാപ്പി കുടിക്കാൻ നിർത്തിയപ്പോൾ ഉടമ പഴയ റേഡിയോ ഓണാക്കി. പാട്ട് പഴയതായിരുന്നു, പക്ഷേ രാവിലെ അതിനൊത്തതായിരുന്നു.",
        },
        {
            "id": "manual_0069_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Laomedeia",
            "emotion_type": "",
            "tag": "",
            "text": "കമ്പ്യൂട്ടർ ക്ലാസിൽ കുട്ടികൾ കീബോർഡ് നോക്കാതെ ടൈപ്പ് ചെയ്യാൻ പഠിച്ചു. ഓരോ തെറ്റിനും അവർ തന്നെ ചിരിച്ചതുകൊണ്ട് ക്ലാസ് വളരെ ലളിതമായി പോയി.",
        },
        {
            "id": "manual_0070_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Orus",
            "emotion_type": "",
            "tag": "",
            "text": "വീട്ടിലെ വെള്ളം തീർന്നപ്പോൾ ടാങ്കർ വിളിക്കേണ്ടിവന്നു. ഡ്രൈവർ എത്തുന്നതുവരെ എല്ലാവരും കുപ്പികളിൽ ബാക്കിയുള്ള വെള്ളം സൂക്ഷിച്ചു ഉപയോഗിച്ചു.",
        },
        {
            "id": "manual_0071_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Leda",
            "emotion_type": "",
            "tag": "",
            "text": "സംഗീത ക്ലാസിൽ ഞാൻ പാടുമ്പോൾ ശ്രുതി അല്പം തെറ്റി. ടീച്ചർ ശാന്തമായി വീണ്ടും കേൾക്കാൻ പറഞ്ഞതുകൊണ്ട് നാണക്കേട് തോന്നിയില്ല.",
        },
        {
            "id": "manual_0072_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Iapetus",
            "emotion_type": "",
            "tag": "",
            "text": "സൂപ്പർമാർക്കറ്റിൽ ബിൽ അടയ്ക്കുമ്പോൾ കൂപ്പൺ സ്കാൻ ആകാതെ നിന്നു. പിന്നിലെ ആളുകൾ ക്ഷമയോടെ കാത്തിരുന്നതുകൊണ്ട് എനിക്ക് ആശ്വാസമായി.",
        },
        {
            "id": "manual_0073_conversation_female",
            "category": "conversation",
            "gender": "female",
            "voice": "Callirrhoe",
            "emotion_type": "",
            "tag": "",
            "text": "തൈക്കടയിൽ നിന്ന് അളവ് എടുക്കുമ്പോൾ അമ്മ തന്റെ പഴയ സാരിയെക്കുറിച്ച് സംസാരിച്ചു. പുതിയ വസ്ത്രം തയ്‌ക്കുന്നതിനെക്കാൾ ഓർമ്മകളാണ് അവൾക്ക് പ്രധാനമായത്.",
        },
        {
            "id": "manual_0074_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Algenib",
            "emotion_type": "",
            "tag": "",
            "text": "പുസ്തകമേളയിൽ പോകുമ്പോൾ ഒരു പുസ്തകം മാത്രം വാങ്ങാമെന്നായിരുന്നു തീരുമാനം. പുറത്തിറങ്ങിയപ്പോൾ ബാഗിൽ അഞ്ച് പുസ്തകങ്ങൾ ഉണ്ടായിരുന്നു.",
        },
        {
            "id": "manual_0075_conversation_male",
            "category": "conversation",
            "gender": "male",
            "voice": "Algenib",
            "emotion_type": "",
            "tag": "",
            "text": "വീട്ടിലെ ചെടിക്ക് ആദ്യമായി പൂ വന്നപ്പോൾ അമ്മ അതിന്റെ ഫോട്ടോ എടുത്തു. ചെറിയ പൂവാണെങ്കിലും അവളുടെ സന്തോഷം മുഴുവൻ വീട്ടിലേക്കും പടർന്നു.",
        },
        {
            "id": "manual_0076_neutral_female",
            "category": "neutral_replay",
            "gender": "female",
            "voice": "Aoede",
            "emotion_type": "",
            "tag": "",
            "text": "ജിമ്മിലെ പുതിയ സമയം രാവിലെ ആറര മുതൽ ആണ്. ആദ്യ ദിവസം വരുമ്പോൾ അംഗത്വ കാർഡ് കൂടെ കൊണ്ടുവരണം.",
        },
        {
            "id": "manual_0077_neutral_male",
            "category": "neutral_replay",
            "gender": "male",
            "voice": "Puck",
            "emotion_type": "",
            "tag": "",
            "text": "ബാങ്ക് അക്കൗണ്ട് പുതുക്കാൻ തിരിച്ചറിയൽ രേഖയും വിലാസ തെളിവും ആവശ്യമാണ്. കൗണ്ടറിൽ പോകുന്നതിന് മുമ്പ് ഫോട്ടോ കോപ്പി എടുത്തുവെക്കുക.",
        },
        {
            "id": "manual_0078_neutral_female",
            "category": "neutral_replay",
            "gender": "female",
            "voice": "Despina",
            "emotion_type": "",
            "tag": "",
            "text": "പാർക്കിലെ കുട്ടികളുടെ മേഖല വൈകുന്നേരം ഏഴ് മണിക്ക് അടയ്ക്കും. കളിപ്പാട്ടങ്ങൾ ഉപയോഗിച്ച ശേഷം അതേ സ്ഥലത്ത് വെയ്ക്കുക.",
        },
        {
            "id": "manual_0079_neutral_male",
            "category": "neutral_replay",
            "gender": "male",
            "voice": "Charon",
            "emotion_type": "",
            "tag": "",
            "text": "സ്കൂൾ ബസ് നാളെ പതിനഞ്ച് മിനിറ്റ് നേരത്തെ വരും. കുട്ടികളെ തയ്യാറാക്കി ഗേറ്റിന് സമീപം നിർത്തുക.",
        },
        {
            "id": "manual_0080_neutral_female",
            "category": "neutral_replay",
            "gender": "female",
            "voice": "Laomedeia",
            "emotion_type": "",
            "tag": "",
            "text": "ഡോക്ടറുടെ അപ്പോയിന്റ്മെന്റ് മൂന്ന് മണിയിലേക്ക് മാറ്റിയിട്ടുണ്ട്. പഴയ റിപ്പോർട്ടുകളും മരുന്ന് പട്ടികയും കൈയിൽ വെക്കണം.",
        },
        {
            "id": "manual_0081_neutral_male",
            "category": "neutral_replay",
            "gender": "male",
            "voice": "Fenrir",
            "emotion_type": "",
            "tag": "",
            "text": "ട്രെയിൻ ടിക്കറ്റ് സ്ഥിരീകരിച്ചു. യാത്രയ്ക്ക് മുമ്പ് പ്ലാറ്റ്ഫോം നമ്പർ വീണ്ടും പരിശോധിക്കുന്നത് നല്ലതാണ്.",
        },
        {
            "id": "manual_0082_neutral_female",
            "category": "neutral_replay",
            "gender": "female",
            "voice": "Leda",
            "emotion_type": "",
            "tag": "",
            "text": "അപ്പാർട്ട്മെന്റ് മീറ്റിംഗ് ഞായറാഴ്ച രാവിലെ ആയിരിക്കും. പാർക്കിംഗ് വിഷയവും വെള്ളവിതരണവും അജണ്ടയിൽ ഉൾപ്പെടുത്തിയിട്ടുണ്ട്.",
        },
        {
            "id": "manual_0083_neutral_male",
            "category": "neutral_replay",
            "gender": "male",
            "voice": "Orus",
            "emotion_type": "",
            "tag": "",
            "text": "വണ്ടിയുടെ സർവീസ് പൂർത്തിയായാൽ സന്ദേശം വരും. അതുവരെ റിസപ്ഷനിൽ കാത്തിരിക്കുകയോ പുറത്തേക്ക് പോകുകയോ ചെയ്യാം.",
        },
        {
            "id": "manual_0084_neutral_female",
            "category": "neutral_replay",
            "gender": "female",
            "voice": "Callirrhoe",
            "emotion_type": "",
            "tag": "",
            "text": "ലൈബ്രറി കാർഡ് പുതുക്കാൻ പഴയ കാർഡ് മതി. പിഴ ഉണ്ടെങ്കിൽ കൗണ്ടറിൽ അടച്ച ശേഷം പുസ്തകം എടുക്കാം.",
        },
        {
            "id": "manual_0085_neutral_male",
            "category": "neutral_replay",
            "gender": "male",
            "voice": "Iapetus",
            "emotion_type": "",
            "tag": "",
            "text": "മത്സ്യ മാർക്കറ്റിലെ പുതിയ നിരക്ക് രാവിലെ മാത്രമേ ലഭിക്കൂ. ഉച്ചയ്ക്ക് ശേഷം വില മാറാൻ സാധ്യതയുണ്ട്.",
        },
        {
            "id": "manual_0086_neutral_female",
            "category": "neutral_replay",
            "gender": "female",
            "voice": "Kore",
            "emotion_type": "",
            "tag": "",
            "text": "ഡാൻസ് ക്ലാസിന് ലളിതമായ വസ്ത്രം ധരിക്കുക. വെള്ളക്കുപ്പി കൊണ്ടുവരുന്നത് കുട്ടികൾക്ക് ഉപകാരപ്പെടും.",
        },
        {
            "id": "manual_0087_neutral_male",
            "category": "neutral_replay",
            "gender": "male",
            "voice": "Algenib",
            "emotion_type": "",
            "tag": "",
            "text": "പഞ്ചായത്ത് ഓഫീസിൽ അപേക്ഷ രാവിലെ സമർപ്പിച്ചാൽ അതേ ദിവസം രസീത് ലഭിക്കും. ഒറിജിനൽ രേഖകൾ പരിശോധിച്ച ശേഷം തിരികെ നൽകും.",
        },
        {
            "id": "manual_0088_emotion_female_laughter",
            "category": "emotion",
            "gender": "female",
            "voice": "Aoede",
            "emotion_type": "laughter",
            "tag": "[laughter]",
            "text": "[laughter] ജിമ്മിൽ കണ്ണാടിക്ക് മുന്നിൽ പോസ് ചെയ്തവൻ പിന്നിൽ ട്രെയിനർ നിൽക്കുന്നത് കണ്ടില്ല. മുഖം മാറിയത് കണ്ടപ്പോൾ മുഴുവൻ റൂമും ചിരിച്ചു.",
        },
        {
            "id": "manual_0089_emotion_male_giggle",
            "category": "emotion",
            "gender": "male",
            "voice": "Puck",
            "emotion_type": "giggle",
            "tag": "[giggle]",
            "text": "[giggle] ബാങ്കിലെ ക്യൂവിൽ ഞാൻ ടോക്കൺ നഷ്ടപ്പെട്ടു എന്ന് കരുതി. അത് മുഴുവൻ സമയം എന്റെ കൈയിലായിരുന്നു.",
        },
        {
            "id": "manual_0090_emotion_female_whisper",
            "category": "emotion",
            "gender": "female",
            "voice": "Despina",
            "emotion_type": "whisper",
            "tag": "[whisper]",
            "text": "[whisper] ടീച്ചർ ബോർഡിൽ എഴുതുമ്പോൾ നീ നോട്ട് എടുത്തോ? ഞാൻ അവസാന വരി മാത്രം നഷ്ടപ്പെടുത്തി.",
        },
        {
            "id": "manual_0091_emotion_male_cry",
            "category": "emotion",
            "gender": "male",
            "voice": "Charon",
            "emotion_type": "cry",
            "tag": "[cry]",
            "text": "[cry] പാർക്കിലെ പഴയ ബെഞ്ച് മാറ്റിയെന്ന് കണ്ടപ്പോൾ അച്ഛനൊപ്പം ഇരുന്ന സന്ധ്യകൾ എല്ലാം ഓർമ്മയായി.",
        },
        {
            "id": "manual_0092_emotion_female_sigh_tired",
            "category": "emotion",
            "gender": "female",
            "voice": "Laomedeia",
            "emotion_type": "sigh_frustration_tired",
            "tag": "[sigh]",
            "text": "[sigh] സ്കൂൾ പ്രോജക്റ്റ് അവസാന രാത്രിയിലേക്ക് മാറ്റിയതാണ് പിഴവ്. ഇപ്പോൾ പേപ്പറും ഗവും എല്ലാം മേശയിൽ കലർന്നിരിക്കുന്നു.",
        },
        {
            "id": "manual_0093_emotion_male_sigh_nervous",
            "category": "emotion",
            "gender": "male",
            "voice": "Fenrir",
            "emotion_type": "sigh_nervous_uncertain",
            "tag": "[sigh]",
            "text": "[sigh] ബാങ്ക് മാനേജരോട് ഇത് എങ്ങനെ പറയണം എന്ന് അറിയില്ല. പക്ഷേ ലോൺ വിഷയത്തിൽ ഇന്നുതന്നെ വ്യക്തത വേണം.",
        },
        {
            "id": "manual_0094_emotion_female_positive",
            "category": "emotion",
            "gender": "female",
            "voice": "Leda",
            "emotion_type": "positive_excited",
            "tag": "",
            "text": "പാട്ട് മത്സരത്തിന്റെ ആദ്യ റൗണ്ട് ഞാൻ കടന്നുപോയി. പേര് വിളിച്ചപ്പോൾ കൈകൾ വിറച്ചെങ്കിലും ഉള്ളിൽ വലിയ സന്തോഷം ഉണ്ടായിരുന്നു.",
        },
        {
            "id": "manual_0095_emotion_male_curious",
            "category": "emotion",
            "gender": "male",
            "voice": "Orus",
            "emotion_type": "curious_confused",
            "tag": "",
            "text": "ജിമ്മിലെ മെഷീൻ ഒരേ വ്യായാമത്തിന് മൂന്ന് രീതിയിൽ ഉപയോഗിക്കുന്നത് എന്തുകൊണ്ടാണ്? ട്രെയിനർ പറഞ്ഞ വ്യത്യാസം വീണ്ടും കേൾക്കണം.",
        },
        {
            "id": "manual_0096_emotion_female_cough",
            "category": "emotion",
            "gender": "female",
            "voice": "Callirrhoe",
            "emotion_type": "cough",
            "tag": "[cough]",
            "text": "[cough] വെള്ളം കുറച്ച് കുടിക്കട്ടെ. പിന്നെ രജിസ്ട്രേഷൻ ഫോം എവിടെ സമർപ്പിക്കണം എന്ന് ഞാൻ പറഞ്ഞുതരാം.",
        },
        {
            "id": "manual_0097_emotion_male_laughter",
            "category": "emotion",
            "gender": "male",
            "voice": "Iapetus",
            "emotion_type": "laughter",
            "tag": "[laughter]",
            "text": "[laughter] ഫുട്ബോൾ കളിക്കാനെത്തിയ ഞാൻ പന്തിന് പകരം ചെരുപ്പാണ് ആദ്യം പറത്തി. എല്ലാവരും കളി നിർത്തി നോക്കി.",
        },
        {
            "id": "manual_0098_emotion_female_giggle",
            "category": "emotion",
            "gender": "female",
            "voice": "Kore",
            "emotion_type": "giggle",
            "tag": "[giggle]",
            "text": "[giggle] ലൈബ്രറിയിൽ ശബ്ദമില്ലാതെ ഇരിക്കാൻ നോക്കിയപ്പോൾ എന്റെ കസേര തന്നെയാണ് ഏറ്റവും വലിയ ശബ്ദം ചെയ്തത്.",
        },
        {
            "id": "manual_0099_emotion_male_whisper",
            "category": "emotion",
            "gender": "male",
            "voice": "Algenib",
            "emotion_type": "whisper",
            "tag": "[whisper]",
            "text": "[whisper] സർപ്രൈസ് കേക്ക് അടുക്കളയിൽ പിന്നിലെ ഷെൽഫിലാണ്. അവൾ വരുന്നതിന് മുമ്പ് ലൈറ്റ് ഓഫ് ചെയ്യണം.",
        },
        {
            "id": "manual_0100_emotion_female_cry",
            "category": "emotion",
            "gender": "female",
            "voice": "Aoede",
            "emotion_type": "cry",
            "tag": "[cry]",
            "text": "[cry] സ്കൂൾ അവസാന ദിവസം കൂട്ടുകാരി നൽകിയ ചെറിയ കത്ത് ഇന്നും ബാഗിൽ സൂക്ഷിച്ചിരിക്കുന്നു. അത് വായിക്കുമ്പോൾ കണ്ണ് നിറയും.",
        },
    ]
)


def count_by(rows: Iterable[Dict[str, str]], key: str) -> Counter:
    return Counter(row[key] for row in rows)


def chunk_counts(rows: Iterable[Dict[str, str]], n: int) -> Counter:
    counts = Counter()
    for row in rows:
        words = WORD_RE.findall(row["text"])
        for index in range(len(words) - n + 1):
            counts[" ".join(words[index : index + n])] += 1
    return counts


def validate_rows(rows: List[Dict[str, str]]):
    ids = [row["id"] for row in rows]
    texts = [normalize_indic_text(row["text"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate ids found")
    if len(texts) != len(set(texts)):
        raise AssertionError("Duplicate full texts found")

    issues = {}
    for row in rows:
        text = normalize_indic_text(row["text"])
        row_issues = []
        found_tags = TAG_RE.findall(text)
        if "|" in text:
            row_issues.append("contains pipe delimiter")
        if len(WORD_RE.findall(text)) < 8:
            row_issues.append("too short")
        unsupported = [tag for tag in found_tags if tag not in SUPPORTED_TAGS]
        if unsupported:
            row_issues.append(f"unsupported tags: {unsupported}")
        if row["tag"]:
            if not text.startswith(row["tag"]):
                row_issues.append(f"text must start with {row['tag']}")
            if found_tags.count(row["tag"]) != 1:
                row_issues.append(f"{row['tag']} must occur exactly once")
        elif found_tags:
            row_issues.append("unexpected bracket tag")
        if row_issues:
            issues[row["id"]] = row_issues

    for n in (5, 6, 7):
        repeated = {chunk: count for chunk, count in chunk_counts(rows, n).items() if count > 2}
        if repeated:
            issues[f"repeated_{n}_word_chunks"] = dict(list(repeated.items())[:10])

    if issues:
        raise AssertionError(json.dumps(issues, ensure_ascii=False, indent=2))


def write_dataset(output: Path, rows: List[Dict[str, str]]):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metadata.csv").open("w", encoding="utf-8", newline="") as metadata_handle, (
        output / "manifest.jsonl"
    ).open("w", encoding="utf-8") as manifest_handle:
        writer = csv.writer(metadata_handle, delimiter="|", lineterminator="\n")
        for row in rows:
            text = normalize_indic_text(row["text"])
            full_row = dict(row)
            full_row["text"] = text
            full_row["language_id"] = "ml"
            full_row["style"] = STYLE_BY_CATEGORY[row["category"]]
            full_row["tts_input"] = make_tts_input(full_row["style"], text)
            writer.writerow([row["id"], text, text, "ml"])
            manifest_handle.write(json.dumps(full_row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Write the manually authored Malayalam review dataset.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    validate_rows(MANUAL_ROWS)
    output = Path(args.output)
    write_dataset(output, MANUAL_ROWS)
    print(f"Rows: {len(MANUAL_ROWS)}")
    print(f"Gender: {dict(count_by(MANUAL_ROWS, 'gender'))}")
    print(f"Category: {dict(count_by(MANUAL_ROWS, 'category'))}")
    print(f"Tags from text: {dict(Counter(tag for row in MANUAL_ROWS for tag in TAG_RE.findall(row['text'])))}")
    print(f"Metadata: {output / 'metadata.csv'}")
    print(f"Manifest: {output / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
