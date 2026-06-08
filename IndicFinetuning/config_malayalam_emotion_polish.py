from dataclasses import dataclass, field
import os
from typing import Optional

from IndicFinetuning.config_indic import IndicTrainConfig


@dataclass
class MalayalamEmotionPolishConfig(IndicTrainConfig):
    csv_path: str = "./IndicFinetuning/datasets/OpenRouterManualReview800/metadata.csv"
    wav_dir: str = "./IndicFinetuning/datasets/OpenRouterManualReview800/wavs"
    preprocessed_dir: str = "./IndicFinetuning/datasets/OpenRouterManualReview800/preprocess"
    output_dir: str = "./IndicFinetuning/outputs/malayalam_emotion_polish"

    is_turbo: bool = False
    is_lora: bool = True
    target_languages: list[str] = field(default_factory=lambda: ["ml"])
    default_language: str = "ml"
    inference_language: str = "ml"

    # Small continuation run on top of the selected 17k Malayalam adapter.
    batch_size: int = 8
    grad_accum: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 4
    save_steps: int = 100
    save_total_limit: int = 4
    dataloader_num_workers: int = 4

    continue_adapter_path: Optional[str] = None

    def __post_init__(self):
        if self.continue_adapter_path is None:
            self.continue_adapter_path = os.environ.get(
                "INDIC_CONTINUE_ADAPTER_PATH",
                "./IndicFinetuning/outputs/adapter_ckpt_17000",
            )
