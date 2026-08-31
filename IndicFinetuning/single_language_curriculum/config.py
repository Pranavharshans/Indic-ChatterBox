from dataclasses import dataclass, field
from typing import List, Optional

from IndicFinetuning.config_indic import IndicTrainConfig


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    manifest: str
    max_steps: int
    learning_rate: float


@dataclass
class MalayalamCurriculumConfig(IndicTrainConfig):
    output_dir: str = "./IndicFinetuning/outputs/malayalam_curriculum_pilot"
    preprocessed_dir: str = "./IndicFinetuning/single_language_curriculum/work/preprocessed"
    plan_dir: str = "./IndicFinetuning/single_language_curriculum/work/plan"
    tokenizer_path: str = "./IndicFinetuning/tokenizer/tokenizer_indic.json"

    target_languages: List[str] = field(default_factory=lambda: ["ml"])
    default_language: str = "ml"
    add_language_tag: bool = True
    is_turbo: bool = False
    is_lora: bool = True
    preprocess: bool = True

    batch_size: int = 16
    grad_accum: int = 1
    dataloader_num_workers: int = 8
    logging_steps: int = 25
    eval_steps: int = 1000
    save_steps: int = 500
    save_total_limit: int = 3

    lora_r: int = 128
    lora_alpha: int = 256
    max_text_len: int = 320
    max_speech_len: int = 850
    prompt_duration: float = 3.0

    stages: List[CurriculumStage] = field(
        default_factory=lambda: [
            CurriculumStage(
                "stage1_ivr",
                "./IndicFinetuning/single_language_curriculum/work/plan/stages/stage1.jsonl",
                6000,
                6e-5,
            ),
            CurriculumStage(
                "stage2_50rasa_50ivr",
                "./IndicFinetuning/single_language_curriculum/work/plan/stages/stage2.jsonl",
                6000,
                3e-5,
            ),
            CurriculumStage(
                "stage3_80rasa_20ivr",
                "./IndicFinetuning/single_language_curriculum/work/plan/stages/stage3.jsonl",
                3000,
                1e-5,
            ),
        ]
    )
    validation_manifest: str = (
        "./IndicFinetuning/single_language_curriculum/work/plan/splits/validation_combined.jsonl"
    )
    continue_adapter_path: Optional[str] = None
