from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from src.chatterbox_.models.tokenizers import MTLTokenizer
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS


def get_engine_class(is_turbo: bool):
    return ChatterboxTurboTTS if is_turbo else ChatterboxTTS


def resolve_tokenizer_path(config) -> Path:
    configured = Path(config.tokenizer_path)
    if configured.exists():
        return configured
    return Path(config.model_dir) / "tokenizer.json"


def attach_indic_tokenizer(engine, config):
    if config.is_turbo:
        return engine
    tokenizer_path = resolve_tokenizer_path(config)
    engine.tokenizer = MTLTokenizer(str(tokenizer_path))
    return engine


def tokenizer_vocab_size(config) -> int:
    if config.is_turbo:
        tokenizer = AutoTokenizer.from_pretrained(config.model_dir, local_files_only=True)
        return len(tokenizer)
    tokenizer_path = resolve_tokenizer_path(config)
    return len(Tokenizer.from_file(str(tokenizer_path)).get_vocab())
