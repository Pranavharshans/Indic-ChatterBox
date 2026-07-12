from dataclasses import dataclass, field
from typing import List, Optional

from IndicFinetuning.config_indic import IndicTrainConfig


@dataclass
class MultilingualIndicConfig(IndicTrainConfig):
    csv_path: str = "./IndicFinetuning/multilingual_indic/core_8lang_45h/dataset/metadata.csv"
    wav_dir: str = "./IndicFinetuning/multilingual_indic/core_8lang_45h/dataset/wavs"
    preprocessed_dir: str = "./IndicFinetuning/multilingual_indic/core_8lang_45h/dataset/preprocess"
    output_dir: str = "./IndicFinetuning/outputs/multilingual_indic_core_8lang_45h"
    tokenizer_path: str = "./IndicFinetuning/tokenizer/tokenizer_indic_core_8lang.json"

    target_languages: List[str] = field(
        default_factory=lambda: ["hi", "bn", "mr", "gu", "ta", "te", "ml", "kn"]
    )
    default_language: str = "hi"
    metadata_language_column: Optional[int] = 3
    add_language_tag: bool = True

    is_turbo: bool = False
    is_lora: bool = True
    preprocess: bool = True
    ljspeech: bool = True
    json_format: bool = False

    # Corrected from the generated tokenizer at runtime if necessary.
    new_vocab_size: int = 4096

    lora_r: int = 128
    lora_alpha: int = 256

    # 45 unique hours per language, balanced 22.5h female/22.5h male.
    batch_size: int = 24
    grad_accum: int = 1
    learning_rate: float = 4e-5
    num_epochs: int = 1
    save_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: int = 6
    save_safetensors: bool = False
    dataloader_num_workers: int = 8

    eval_on_save: bool = True
    eval_sample_steps: int = 1000
    eval_samples_per_language: int = 4
    eval_output_dir: str = "./IndicFinetuning/outputs/multilingual_indic_core_8lang_45h/eval_samples"
    eval_temperature: float = 0.8
    eval_repetition_penalty: float = 1.2
    eval_prompt_min_duration: float = 5.0
    eval_prompt_max_duration: float = 10.0

    max_text_len: int = 320
    max_speech_len: int = 850
    prompt_duration: float = 3.0

    resume_from_checkpoint: Optional[str] = None
    continue_adapter_path: Optional[str] = None
