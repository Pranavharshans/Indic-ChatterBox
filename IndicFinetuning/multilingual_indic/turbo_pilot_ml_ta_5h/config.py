from dataclasses import dataclass, field
from typing import List, Optional

from IndicFinetuning.config_indic import IndicTrainConfig


@dataclass
class MultilingualIndicConfig(IndicTrainConfig):
    model_dir: str = "./pretrained_models_turbo"
    csv_path: str = "./IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/dataset/metadata.csv"
    wav_dir: str = "./IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/dataset/wavs"
    preprocessed_dir: str = "./IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/dataset/preprocess"
    output_dir: str = "./IndicFinetuning/outputs/turbo_pilot_ml_ta_5h"
    tokenizer_path: str = "./pretrained_models_turbo"

    target_languages: List[str] = field(default_factory=lambda: ["ml", "ta"])
    default_language: str = "ml"
    metadata_language_column: Optional[int] = 3
    add_language_tag: bool = True

    is_turbo: bool = True
    is_lora: bool = True
    preprocess: bool = True
    ljspeech: bool = True
    json_format: bool = False

    # Replaced automatically with the tokenizer length at runtime.
    new_vocab_size: int = 50276

    lora_r: int = 64
    lora_alpha: int = 128

    # Five unique hours per language. Two epochs let the pilot expose both
    # first-pass learning and any second-pass quality regression.
    batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 2
    save_steps: int = 250
    logging_steps: int = 25
    save_total_limit: int = 8
    save_safetensors: bool = False
    dataloader_num_workers: int = 8

    eval_on_save: bool = True
    eval_sample_steps: int = 250
    eval_samples_per_language: int = 4
    eval_output_dir: str = "./IndicFinetuning/outputs/turbo_pilot_ml_ta_5h/eval_samples"
    eval_temperature: float = 0.8
    eval_repetition_penalty: float = 1.2
    eval_prompt_min_duration: float = 5.1
    eval_prompt_max_duration: float = 10.0

    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = 3.0

    resume_from_checkpoint: Optional[str] = None
    continue_adapter_path: Optional[str] = None
