from dataclasses import dataclass, field
from typing import List, Optional

from IndicFinetuning.config_indic import IndicTrainConfig


@dataclass
class MultilingualIndicConfig(IndicTrainConfig):
    csv_path: str = "./IndicFinetuning/multilingual_indic/strong_run/dataset/metadata.csv"
    wav_dir: str = "./IndicFinetuning/multilingual_indic/strong_run/dataset/wavs"
    preprocessed_dir: str = "./IndicFinetuning/multilingual_indic/strong_run/dataset/preprocess"
    output_dir: str = "./IndicFinetuning/outputs/multilingual_indic_strong_run"
    tokenizer_path: str = "./IndicFinetuning/tokenizer/tokenizer_indic_12lang.json"

    target_languages: List[str] = field(default_factory=lambda: ["hi", "ta", "te", "ml", "kn", "bn", "mr", "gu", "pa", "ur", "or", "as"])
    default_language: str = "hi"
    metadata_language_column: Optional[int] = 3
    add_language_tag: bool = True

    is_turbo: bool = False
    is_lora: bool = True
    preprocess: bool = True
    ljspeech: bool = True
    json_format: bool = False

    # This is auto-corrected from the tokenizer at runtime if needed.
    new_vocab_size: int = 4096

    lora_r: int = 128
    lora_alpha: int = 256

    # Strong run: 20h/language, 12 languages, ~240h total.
    batch_size: int = 24
    grad_accum: int = 1
    learning_rate: float = 6e-5
    num_epochs: int = 2
    save_steps: int = 1500
    logging_steps: int = 50
    save_total_limit: int = 6
    dataloader_num_workers: int = 8

    max_text_len: int = 320
    max_speech_len: int = 850
    prompt_duration: float = 3.0

    resume_from_checkpoint: Optional[str] = None
    continue_adapter_path: Optional[str] = None
