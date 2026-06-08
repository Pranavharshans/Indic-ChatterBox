import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.config_indic import IndicTrainConfig
from IndicFinetuning.inference_indic import load_lora_engine
from IndicFinetuning.indic_text import apply_language_tag, normalize_indic_text
from src.utils import trim_silence_with_vad


SCRIPTS = {
    "malayalam": (
        "ഇന്ന് വൈകുന്നേരം മഴ പെയ്യാൻ തുടങ്ങുമ്പോൾ, ഞാൻ വീട്ടിന്റെ മുന്നിൽ കുറച്ച് നേരം നിന്നു. "
        "കാറ്റ് പതുക്കെ വീശി, വഴിയിലൂടെ പോകുന്ന ആളുകളുടെ ശബ്ദം അകലെ കേട്ടു. "
        "ആ നിമിഷം മനസ്സിൽ ഒരു ശാന്തത വന്നു, ജീവിതം എത്ര തിരക്കായാലും ഇങ്ങനെ ചെറിയ സന്തോഷങ്ങൾ നമ്മളെ പിടിച്ചു നിർത്തും. "
        "പിന്നെ ഒരു ചായ എടുത്ത് ഇരിക്കുമ്പോൾ, ദിവസം മുഴുവൻ ഉണ്ടായിരുന്ന ക്ഷീണം പതുക്കെ മാറിപ്പോയി."
    ),
    "malappuram": (
        "ഇജ്ജ് ഒന്ന് കേട്ടോ, ഇന്നലെ ടൗണിൽ പോയപ്പോ ഒരു പൊളി സംഭവം ഉണ്ടായി. "
        "ബസ് കാത്ത് നിക്കുമ്പോ എന്റെ പഴയ കൂട്ടുകാരൻ വന്നു പറഞ്ഞു, എടാ മോനെ, അനക്ക് ഇപ്പൊ സമയം ഒന്നും ഇല്ലേന്ന്. "
        "ഞാൻ പറഞ്ഞു, കാക്കാ, പണി ഒക്കെ ആയിട്ട് തല തിരിയാൻ സമയം കിട്ടുന്നില്ല, പക്ഷേ നാട്ടിലെ ആളുകളെ കണ്ടാൽ മനസ്സിന് വേറെ സന്തോഷം തന്നെയാണ്. "
        "അപ്പൊ അവൻ ചിരിച്ചിട്ട് പറഞ്ഞു, എന്നാ ഒരു ചായ കുടിച്ച് പോവാ, ബാക്കി കഥ ഒക്കെ അവിടെ ഇരുന്ന് പറയാം."
    ),
}


PRESETS = {
    "default": {
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
    },
    "expressive": {
        "exaggeration": 0.75,
        "cfg_weight": 0.3,
        "temperature": 0.85,
        "repetition_penalty": 1.2,
    },
    "accent_safe": {
        "exaggeration": 0.5,
        "cfg_weight": 0.05,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
    },
}


def generate(engine, text, prompt_path, language_id, params):
    formatted_text = normalize_indic_text(text)
    formatted_text = apply_language_tag(formatted_text, language_id, enabled=True)
    wav_tensor = engine.generate(text=formatted_text, audio_prompt_path=prompt_path, **params)
    if isinstance(wav_tensor, tuple):
        wav_tensor = wav_tensor[0]
    wav_np = wav_tensor.squeeze().detach().cpu().numpy()
    return engine.sr, trim_silence_with_vad(wav_np, engine.sr)


def main():
    cfg = IndicTrainConfig()
    prompt_path = os.environ.get("INDIC_PROMPT_PATH", cfg.inference_prompt_path)
    output_dir = Path(os.environ.get("INDIC_SAMPLE_DIR", "./IndicFinetuning/outputs/voice_samples"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(prompt_path).exists():
        raise FileNotFoundError(f"Prompt WAV not found: {prompt_path}")

    print(f"Loading adapter: {os.environ.get('INDIC_ADAPTER_PATH', os.path.join(cfg.output_dir, 'indic_adapter'))}")
    print(f"Prompt: {prompt_path}")
    print(f"Output dir: {output_dir}")

    engine = load_lora_engine("cuda" if torch.cuda.is_available() else "cpu")

    for script_name, script_text in SCRIPTS.items():
        for preset_name, params in PRESETS.items():
            sample_rate, audio = generate(engine, script_text, prompt_path, cfg.inference_language, params)
            output_path = output_dir / f"{script_name}_{preset_name}.wav"
            sf.write(output_path, audio, sample_rate)
            duration = len(audio) / sample_rate if sample_rate else 0.0
            print(f"Wrote {output_path} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
