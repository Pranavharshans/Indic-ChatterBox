import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.config_malayalam_emotion_fixed_inline import MalayalamEmotionFixedInlineConfig
from IndicFinetuning.train_malayalam_emotion_polish import run_training
from src.utils import setup_logger


logger = setup_logger("Malayalam-Emotion-Fixed-Inline")


def main():
    run_training(
        MalayalamEmotionFixedInlineConfig(),
        run_logger=logger,
        run_title="Malayalam Emotion Fixed Inline Continuation",
    )


if __name__ == "__main__":
    main()
