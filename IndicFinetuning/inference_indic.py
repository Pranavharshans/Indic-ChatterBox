import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.config_indic import IndicTrainConfig
from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from IndicFinetuning.indic_text import apply_language_tag, normalize_indic_text
from src.chatterbox_.models.t3.t3 import T3
from src.model import resize_and_load_t3_weights
from src.utils import setup_logger, trim_silence_with_vad


logger = setup_logger("Indic-Chatterbox-Inference")
cfg = IndicTrainConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_PATH = os.environ.get("INDIC_ADAPTER_PATH", os.path.join(cfg.output_dir, "indic_adapter"))
OUTPUT_FILE = os.environ.get("INDIC_OUTPUT_FILE", "./IndicFinetuning/outputs/indic_output.wav")
SKIP_VAD = os.environ.get("INDIC_SKIP_VAD", "0") == "1"

if not cfg.is_turbo:
    cfg.new_vocab_size = tokenizer_vocab_size(cfg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_text_for_generation(text: str, language_id: str) -> str:
    text = normalize_indic_text(text, cfg.normalize_unicode)
    return apply_language_tag(text, language_id, cfg.add_language_tag)


def load_lora_engine(device):
    from peft import PeftModel

    engine_class = get_engine_class(cfg.is_turbo)
    temp_engine = engine_class.from_local(cfg.model_dir, device="cpu")
    temp_engine = attach_indic_tokenizer(temp_engine, cfg)
    pretrained_state = temp_engine.t3.state_dict()
    original_config = temp_engine.t3.hp
    original_config.text_tokens_dict_size = cfg.new_vocab_size
    setattr(original_config, "use_cache", False)

    new_t3 = T3(hp=original_config)
    new_t3 = resize_and_load_t3_weights(new_t3, pretrained_state)
    if cfg.is_turbo and hasattr(new_t3.tfmr, "wte"):
        del new_t3.tfmr.wte

    del temp_engine
    del pretrained_state

    engine = engine_class.from_local(cfg.model_dir, device="cpu")
    engine = attach_indic_tokenizer(engine, cfg)
    engine.t3 = PeftModel.from_pretrained(new_t3, ADAPTER_PATH, is_trainable=False)
    engine.t3.to(device).eval()
    engine.s3gen.to(device).eval()
    engine.ve.to(device).eval()
    engine.device = device
    return engine


def load_full_engine(device):
    engine_class = get_engine_class(cfg.is_turbo)
    engine = engine_class.from_local(cfg.model_dir, device="cpu")
    engine = attach_indic_tokenizer(engine, cfg)
    t3_config = engine.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size
    new_t3 = T3(hp=t3_config)
    if cfg.is_turbo and hasattr(new_t3.tfmr, "wte"):
        del new_t3.tfmr.wte

    filename = "t3_turbo_indic_finetuned.safetensors" if cfg.is_turbo else "t3_indic_finetuned.safetensors"
    state_dict = load_file(os.path.join(cfg.output_dir, filename), device="cpu")
    new_t3.load_state_dict(state_dict, strict=True)
    engine.t3 = new_t3
    engine.t3.to(device).eval()
    engine.s3gen.to(device).eval()
    engine.ve.to(device).eval()
    engine.device = device
    return engine


def generate_sentence_audio(engine, text, prompt_path, language_id):
    formatted_text = format_text_for_generation(text, language_id)
    wav_tensor = engine.generate(
        text=formatted_text,
        audio_prompt_path=prompt_path,
        temperature=0.8,
        repetition_penalty=1.2,
    )
    if isinstance(wav_tensor, tuple):
        wav_tensor = wav_tensor[0]
    wav_np = wav_tensor.squeeze().cpu().numpy()
    if SKIP_VAD:
        return engine.sr, wav_np
    return engine.sr, trim_silence_with_vad(wav_np, engine.sr)


def main():
    logger.info(f"Indic inference running on: {DEVICE}")
    engine = load_lora_engine(DEVICE) if cfg.is_lora else load_full_engine(DEVICE)
    language_id = cfg.inference_language
    sentences = [sentence for sentence in re.split(r"(?<=[.?!।॥])\s+", cfg.inference_test_text.strip()) if sentence.strip()]

    set_seed(42)
    all_chunks = []
    sample_rate = 24000
    for sentence in sentences:
        sr, audio_chunk = generate_sentence_audio(engine, sentence, cfg.inference_prompt_path, language_id)
        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            all_chunks.append(np.zeros(int(sr * 0.2), dtype=np.float32))

    if not all_chunks:
        raise RuntimeError("No audio was generated.")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    sf.write(OUTPUT_FILE, np.concatenate(all_chunks), sample_rate)
    logger.info(f"Result saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
