from dataclasses import dataclass, field
from typing import List, Optional

from IndicFinetuning.config_indic import IndicTrainConfig


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    manifest: str
    learning_rate: float
    epochs: float = 1.0


@dataclass
class MalayalamCurriculumConfig(IndicTrainConfig):
    output_dir: str = "./IndicFinetuning/outputs/malayalam_curriculum_1epoch"
    preprocessed_dir: str = "./IndicFinetuning/single_language_curriculum/work/preprocessed"
    plan_dir: str = "./IndicFinetuning/single_language_curriculum/work/plan"
    tokenizer_path: str = "./IndicFinetuning/tokenizer/tokenizer_indic.json"

    target_languages: List[str] = field(default_factory=lambda: ["ml"])
    default_language: str = "ml"
    add_language_tag: bool = True
    is_turbo: bool = False
    is_lora: bool = True
    preprocess: bool = True

    # Conservative effective batch 16 for a single RTX 5090 32 GB.
    batch_size: int = 8
    grad_accum: int = 2
    dataloader_num_workers: int = 8
    logging_steps: int = 25
    eval_steps: int = 1000
    save_steps: int = 1000
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
                6e-5,
            ),
            CurriculumStage(
                "stage2_50rasa_50ivr",
                "./IndicFinetuning/single_language_curriculum/work/plan/stages/stage2.jsonl",
                3e-5,
            ),
            CurriculumStage(
                "stage3_80rasa_20ivr",
                "./IndicFinetuning/single_language_curriculum/work/plan/stages/stage3.jsonl",
                1e-5,
            ),
        ]
    )
    validation_manifest: str = (
        "./IndicFinetuning/single_language_curriculum/work/plan/splits/validation_combined.jsonl"
    )
    eval_prompt_manifest: str = (
        "./IndicFinetuning/single_language_curriculum/work/plan/splits/rasa_validation.jsonl"
    )
    audio_samples_on_steps: bool = True
    audio_sample_steps: int = 1000
    audio_samples_per_checkpoint: int = 2
    audio_samples_on_stage_end: bool = True
    audio_sample_output_dir: str = (
        "./IndicFinetuning/outputs/malayalam_curriculum_1epoch/audio_samples"
    )
    audio_sample_temperature: float = 0.8
    audio_sample_repetition_penalty: float = 1.2
    audio_sample_seed: int = 42
    continue_adapter_path: Optional[str] = None
