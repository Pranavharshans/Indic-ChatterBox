from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from IndicFinetuning.preprocess_indic import preprocess_rows_indic
from IndicFinetuning.single_language_curriculum.curriculum import read_jsonl
from src.chatterbox_.models.t3.t3 import T3
from src.dataset import CurriculumManifestDataset, data_collator_standart, data_collator_turbo
from src.model import ChatterboxTrainerWrapper, resize_and_load_t3_weights
from src.utils import check_pretrained_models, setup_logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = setup_logger("Malayalam-Curriculum-Finetune")


def load_config(config_file: str, class_name: str):
    path = Path(config_file).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)()


def unique_plan_rows(cfg):
    rows = {}
    manifests = [stage.manifest for stage in cfg.stages]
    manifests.append(cfg.validation_manifest)
    split_dir = Path(cfg.plan_dir) / "splits"
    manifests.extend(str(path) for path in split_dir.glob("*_test.jsonl"))
    for manifest in manifests:
        for row in read_jsonl(manifest):
            rows[(row.source, row.id)] = row.to_dict()
    return list(rows.values())


def prepare_model(cfg):
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(model_dir=cfg.model_dir, mode=mode_check):
        raise RuntimeError(f"Pretrained {mode_check} model files are missing from {cfg.model_dir}")

    inferred_vocab_size = tokenizer_vocab_size(cfg)
    if inferred_vocab_size != cfg.new_vocab_size:
        logger.warning(
            f"Configured new_vocab_size={cfg.new_vocab_size}, tokenizer has {inferred_vocab_size}; using tokenizer size"
        )
        cfg.new_vocab_size = inferred_vocab_size

    engine_class = get_engine_class(cfg.is_turbo)
    original = attach_indic_tokenizer(engine_class.from_local(cfg.model_dir, device="cpu"), cfg)
    original_state = original.t3.state_dict()
    t3_config = original.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size
    setattr(t3_config, "use_cache", False)
    new_t3 = resize_and_load_t3_weights(T3(hp=t3_config), original_state)
    if cfg.is_turbo and hasattr(new_t3.tfmr, "wte"):
        del new_t3.tfmr.wte
    del original
    del original_state

    engine = attach_indic_tokenizer(engine_class.from_local(cfg.model_dir, device="cpu"), cfg)
    engine.t3 = new_t3
    for component in (engine.ve, engine.s3gen):
        for parameter in component.parameters():
            parameter.requires_grad = False

    if cfg.is_lora:
        if cfg.continue_adapter_path:
            from peft import PeftModel

            engine.t3 = PeftModel.from_pretrained(
                engine.t3,
                cfg.continue_adapter_path,
                is_trainable=True,
            )
        else:
            from peft import LoraConfig, get_peft_model

            for parameter in engine.t3.parameters():
                parameter.requires_grad = False
            lora = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.turbo_lora_target_modules if cfg.is_turbo else cfg.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                modules_to_save=cfg.lora_modules_to_save,
            )
            engine.t3 = get_peft_model(engine.t3, lora)
        engine.t3.print_trainable_parameters()
    else:
        for parameter in engine.t3.parameters():
            parameter.requires_grad = True
    return engine


def training_arguments(cfg, stage):
    kwargs = dict(
        output_dir=str(Path(cfg.output_dir) / stage.name),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=stage.learning_rate,
        max_steps=stage.max_steps,
        lr_scheduler_type="constant",
        save_strategy="steps",
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        logging_strategy="steps",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=False,
        bf16=True,
        save_total_limit=cfg.save_total_limit,
        save_safetensors=False,
        gradient_checkpointing=True,
        dataloader_persistent_workers=cfg.dataloader_num_workers > 0,
        dataloader_pin_memory=True,
    )
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def save_stage_adapter(engine, output_dir: Path):
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    engine.t3.save_pretrained(adapter_dir)
    return adapter_dir


def tensors_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensors_to_cpu(item) for item in value)
    return value


def run_training(cfg):
    logger.info("Starting continuous Malayalam curriculum training")
    engine = prepare_model(cfg)
    if cfg.preprocess:
        rows = unique_plan_rows(cfg)
        preprocess_rows_indic(
            cfg,
            engine,
            rows,
            output_dir=cfg.preprocessed_dir,
            skip_existing=True,
        )
        engine.ve.to("cpu")
        engine.s3gen.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wrapper = ChatterboxTrainerWrapper(engine.t3)
    collator = data_collator_turbo if cfg.is_turbo else data_collator_standart
    validation = CurriculumManifestDataset(
        cfg,
        cfg.validation_manifest,
        conditioning_dropout=False,
    )
    optimizer_state = None
    journal = []

    for stage in cfg.stages:
        logger.info(
            f"Starting {stage.name}: steps={stage.max_steps}, lr={stage.learning_rate}, manifest={stage.manifest}"
        )
        train_dataset = CurriculumManifestDataset(cfg, stage.manifest)
        args = training_arguments(cfg, stage)
        trainer = Trainer(
            model=wrapper,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=validation,
            data_collator=collator,
        )
        trainer.create_optimizer()
        if optimizer_state is not None:
            trainer.optimizer.load_state_dict(optimizer_state)
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = stage.learning_rate
            parameter_group["initial_lr"] = stage.learning_rate
        trainer.create_scheduler(num_training_steps=stage.max_steps, optimizer=trainer.optimizer)
        result = trainer.train()
        optimizer_state = tensors_to_cpu(trainer.optimizer.state_dict())

        stage_dir = Path(args.output_dir)
        adapter_dir = save_stage_adapter(engine, stage_dir)
        torch.save(optimizer_state, stage_dir / "optimizer_stage_end.pt")
        stage_metrics = result.metrics
        journal.append(
            {
                "stage": stage.name,
                "manifest": stage.manifest,
                "max_steps": stage.max_steps,
                "learning_rate": stage.learning_rate,
                "adapter": str(adapter_dir),
                "metrics": stage_metrics,
            }
        )
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / "curriculum_state.json").write_text(
            json.dumps(journal, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_dir = Path(cfg.output_dir) / "final_adapter"
    engine.t3.save_pretrained(final_dir)
    logger.info(f"Curriculum complete. Final adapter: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train one Indic language with a three-stage curriculum.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--config-class", default="MalayalamCurriculumConfig")
    args = parser.parse_args()
    run_training(load_config(args.config_file, args.config_class))


if __name__ == "__main__":
    main()
