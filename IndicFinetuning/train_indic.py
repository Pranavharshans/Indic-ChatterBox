import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.config_indic import IndicTrainConfig
from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from IndicFinetuning.preprocess_indic import preprocess_dataset_indic
from src.dataset import ChatterboxDataset, data_collator_standart, data_collator_turbo
from src.model import ChatterboxTrainerWrapper, resize_and_load_t3_weights
from src.chatterbox_.models.t3.t3 import T3
from src.utils import check_pretrained_models, setup_logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = setup_logger("Indic-Chatterbox-Finetune")


def main():
    cfg = IndicTrainConfig()
    logger.info("--- Starting Indic Chatterbox Finetuning ---")
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")
    logger.info(f"Languages: {', '.join(cfg.target_languages)}")

    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        sys.exit(1)

    if not cfg.is_turbo:
        inferred_vocab_size = tokenizer_vocab_size(cfg)
        if inferred_vocab_size != cfg.new_vocab_size:
            logger.warning(f"Configured new_vocab_size={cfg.new_vocab_size}, tokenizer has {inferred_vocab_size}. Using tokenizer size.")
            cfg.new_vocab_size = inferred_vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine_class = get_engine_class(cfg.is_turbo)

    logger.info("Loading original model to extract T3 weights...")
    tts_engine_original = engine_class.from_local(cfg.model_dir, device="cpu")
    tts_engine_original = attach_indic_tokenizer(tts_engine_original, cfg)
    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
    setattr(new_t3_config, "use_cache", False)
    new_t3_model = T3(hp=new_t3_config)
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)

    if cfg.is_turbo and hasattr(new_t3_model.tfmr, "wte"):
        del new_t3_model.tfmr.wte

    del tts_engine_original
    del pretrained_t3_state_dict

    tts_engine_new = engine_class.from_local(cfg.model_dir, device="cpu")
    tts_engine_new = attach_indic_tokenizer(tts_engine_new, cfg)
    tts_engine_new.t3 = new_t3_model

    for param in tts_engine_new.ve.parameters():
        param.requires_grad = False
    for param in tts_engine_new.s3gen.parameters():
        param.requires_grad = False

    if cfg.is_lora:
        from peft import LoraConfig, get_peft_model

        for param in tts_engine_new.t3.parameters():
            param.requires_grad = False

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.turbo_lora_target_modules if cfg.is_turbo else cfg.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=cfg.lora_modules_to_save,
        )
        tts_engine_new.t3 = get_peft_model(tts_engine_new.t3, peft_config)
        tts_engine_new.t3.print_trainable_parameters()
    else:
        tts_engine_new.t3.train()
        for param in tts_engine_new.t3.parameters():
            param.requires_grad = True

    if cfg.preprocess:
        preprocess_dataset_indic(cfg, tts_engine_new)

    train_ds = ChatterboxDataset(cfg)
    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)
    selected_collator = data_collator_turbo if cfg.is_turbo else data_collator_standart

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="epoch",
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
        data_collator=selected_collator,
    )
    trainer.train()

    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.is_lora:
        save_path = os.path.join(cfg.output_dir, "indic_adapter")
        tts_engine_new.t3.save_pretrained(save_path)
        logger.info(f"Indic LoRA adapter saved to: {save_path}")
    else:
        filename = "t3_turbo_indic_finetuned.safetensors" if cfg.is_turbo else "t3_indic_finetuned.safetensors"
        final_model_path = os.path.join(cfg.output_dir, filename)
        save_file(tts_engine_new.t3.state_dict(), final_model_path)
        logger.info(f"Indic full model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
