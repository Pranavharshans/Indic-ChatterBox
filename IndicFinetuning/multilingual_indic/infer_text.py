import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_text import apply_language_tag, normalize_indic_text
from IndicFinetuning.multilingual_indic.generate_eval_samples import load_config, load_lora_engine
from src.utils import setup_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Generate custom text with a multilingual Indic adapter.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--config-class", default="MultilingualIndicConfig")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--min-p", type=float, default=0.05)
    args = parser.parse_args()

    logger = setup_logger("Multilingual-Indic-Custom-Inference")
    config = load_config(args.config_file, args.config_class)
    if args.language not in config.target_languages:
        raise ValueError(f"Language {args.language!r} is not enabled in this adapter: {config.target_languages}")

    adapter_path = Path(args.adapter_path)
    prompt_wav = Path(args.prompt_wav)
    output_path = Path(args.output)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Prompt WAV not found: {prompt_wav}")

    if config.is_turbo:
        info = sf.info(str(prompt_wav))
        if info.frames / info.samplerate <= 5.0:
            raise ValueError("Turbo reference audio must be longer than five seconds.")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {'Turbo' if config.is_turbo else 'standard'} adapter on {device}: {adapter_path}")
    engine = load_lora_engine(config, str(adapter_path), device)

    clean_text = normalize_indic_text(args.text, config.normalize_unicode)
    formatted_text = apply_language_tag(clean_text, args.language, config.add_language_tag)
    generation_args = {
        "text": formatted_text,
        "audio_prompt_path": str(prompt_wav),
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
    }
    if config.is_turbo:
        generation_args["top_k"] = args.top_k
    else:
        generation_args.update(
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            min_p=args.min_p,
        )

    wav_tensor = engine.generate(**generation_args)
    if isinstance(wav_tensor, tuple):
        wav_tensor = wav_tensor[0]
    wav = wav_tensor.squeeze().detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wav, engine.sr)

    manifest = {
        "mode": "turbo" if config.is_turbo else "standard",
        "adapter": str(adapter_path),
        "prompt_wav": str(prompt_wav),
        "language": args.language,
        "text": args.text,
        "output": str(output_path),
        "sample_rate": engine.sr,
        "seed": args.seed,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
        "top_k": args.top_k if config.is_turbo else None,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved audio: {output_path}")


if __name__ == "__main__":
    main()
