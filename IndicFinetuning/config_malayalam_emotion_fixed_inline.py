from dataclasses import dataclass

from IndicFinetuning.config_malayalam_emotion_polish import MalayalamEmotionPolishConfig


@dataclass
class MalayalamEmotionFixedInlineConfig(MalayalamEmotionPolishConfig):
    csv_path: str = "./IndicFinetuning/datasets/OpenRouterScript800FixedInline/metadata.csv"
    wav_dir: str = "./IndicFinetuning/datasets/OpenRouterScript800FixedInline/wavs"
    preprocessed_dir: str = "./IndicFinetuning/datasets/OpenRouterScript800FixedInline/preprocess"
    output_dir: str = "./IndicFinetuning/outputs/malayalam_emotion_fixed_inline"

    # Slightly lower LR for the corrected tag-control continuation.
    learning_rate: float = 1e-5
    num_epochs: int = 4
    save_steps: int = 100
