import os
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.config_malayalam_emotion_polish import MalayalamEmotionPolishConfig
from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from IndicFinetuning.preprocess_indic import preprocess_dataset_indic
from src.chatterbox_.models.t3.t3 import T3
from src.dataset import ChatterboxDataset, data_collator_standart
from src.model import ChatterboxTrainerWrapper, resize_and_load_t3_weights
from src.utils import check_pretrained_models, setup_logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = setup_logger("Malayalam-Emotion-Polish")


def run_training(cfg, run_logger=logger, run_title="Malayalam Emotion Polish Continuation"):
    run_logger.info(f"--- Starting {run_title} ---")
    run_logger.info(f"Dataset: {cfg.csv_path}")
    run_logger.info(f"Continue adapter: {cfg.continue_adapter_path}")

    if not check_pretrained_models(mode="chatterbox"):
        sys.exit(1)

    inferred_vocab_size = tokenizer_vocab_size(cfg)
    if inferred_vocab_size != cfg.new_vocab_size:
        run_logger.warning(f"Configured new_vocab_size={cfg.new_vocab_size}, tokenizer has {inferred_vocab_size}. Using tokenizer size.")
        cfg.new_vocab_size = inferred_vocab_size

    engine_class = get_engine_class(False)

    run_logger.info("Loading original model to extract T3 weights...")
    tts_engine_original = engine_class.from_local(cfg.model_dir, device="cpu")
    tts_engine_original = attach_indic_tokenizer(tts_engine_original, cfg)
    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    run_logger.info(f"Creating resized T3 model with vocab size: {cfg.new_vocab_size}")
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
    setattr(new_t3_config, "use_cache", False)
    new_t3_model = T3(hp=new_t3_config)
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)

    del tts_engine_original
    del pretrained_t3_state_dict

    tts_engine_new = engine_class.from_local(cfg.model_dir, device="cpu")
    tts_engine_new = attach_indic_tokenizer(tts_engine_new, cfg)
    tts_engine_new.t3 = new_t3_model

    for param in tts_engine_new.ve.parameters():
        param.requires_grad = False
    for param in tts_engine_new.s3gen.parameters():
        param.requires_grad = False
    for param in tts_engine_new.t3.parameters():
        param.requires_grad = False

    from peft import PeftModel

    if not cfg.continue_adapter_path or not Path(cfg.continue_adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found: {cfg.continue_adapter_path}")

    run_logger.info("Loading 17k Malayalam adapter as trainable continuation adapter...")
    tts_engine_new.t3 = PeftModel.from_pretrained(tts_engine_new.t3, cfg.continue_adapter_path, is_trainable=True)
    tts_engine_new.t3.print_trainable_parameters()

    if cfg.preprocess:
        preprocess_dataset_indic(cfg, tts_engine_new)

    train_ds = ChatterboxDataset(cfg)
    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=20,
        logging_strategy="steps",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=False,
        bf16=True,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=True,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator_standart,
    )
    trainer.train()

    save_path = os.path.join(cfg.output_dir, "indic_adapter")
    tts_engine_new.t3.save_pretrained(save_path)
    run_logger.info(f"Malayalam emotion polish adapter saved to: {save_path}")


def main():
    run_training(MalayalamEmotionPolishConfig())


if __name__ == "__main__":
    main()
