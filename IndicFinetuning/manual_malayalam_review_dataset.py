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
