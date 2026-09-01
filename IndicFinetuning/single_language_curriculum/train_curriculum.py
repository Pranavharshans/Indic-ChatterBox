from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from IndicFinetuning.indic_text import apply_language_tag, normalize_indic_text
from IndicFinetuning.multilingual_indic.generate_eval_samples import PROMPTS
from IndicFinetuning.preprocess_indic import preprocess_rows_indic
from IndicFinetuning.single_language_curriculum.curriculum import cumulative_interval_step, read_jsonl
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
        num_train_epochs=stage.epochs,
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


def set_generation_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def find_eval_prompt(cfg) -> str:
    rows = read_jsonl(cfg.eval_prompt_manifest)
    candidates = [row for row in rows if Path(row.audio_path).exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No existing reference WAV was found in {cfg.eval_prompt_manifest}"
        )
    preferred = [row for row in candidates if 5.0 <= row.duration <= 10.0]
    return (preferred or candidates)[0].audio_path


def generate_audio_samples(engine, cfg, cumulative_step: int, stage_name: str, prompt_wav: str):
    output_dir = Path(cfg.audio_sample_output_dir) / f"step-{cumulative_step:06d}"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = engine.t3.training
    t3_device = next(engine.t3.parameters()).device
    s3gen_device = next(engine.s3gen.parameters()).device
    ve_device = next(engine.ve.parameters()).device
    original_engine_device = engine.device
    rng_state = capture_rng_state()
    records = []

    engine.t3.eval()
    engine.s3gen.to(t3_device).eval()
    engine.ve.to(t3_device).eval()
    engine.device = str(t3_device)
    try:
        prompts = PROMPTS["ml"][: cfg.audio_samples_per_checkpoint]
        set_generation_seed(cfg.audio_sample_seed)
        with torch.no_grad():
            for index, text in enumerate(prompts, start=1):
                formatted_text = apply_language_tag(
                    normalize_indic_text(text, cfg.normalize_unicode),
                    "ml",
                    cfg.add_language_tag,
                )
                wav = engine.generate(
                    text=formatted_text,
                    audio_prompt_path=prompt_wav,
                    temperature=cfg.audio_sample_temperature,
                    repetition_penalty=cfg.audio_sample_repetition_penalty,
                )
                if isinstance(wav, tuple):
                    wav = wav[0]
                output_path = output_dir / f"ml_{index:02d}.wav"
                sf.write(output_path, wav.squeeze().detach().cpu().numpy(), engine.sr)
                records.append(
                    {
                        "cumulative_step": cumulative_step,
                        "stage": stage_name,
                        "text": text,
                        "prompt_wav": prompt_wav,
                        "audio": str(output_path),
                        "seed": cfg.audio_sample_seed,
                    }
                )
    finally:
        if hasattr(engine, "conds"):
            engine.conds = None
        engine.t3.train(was_training)
        engine.s3gen.to(s3gen_device)
        engine.ve.to(ve_device)
        engine.device = original_engine_device
        restore_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Generated cumulative-step audio samples: {output_dir}")


class AudioSamplesCallback(TrainerCallback):
    def __init__(self, engine, cfg, stage_name: str, step_offset: int, prompt_wav: str):
        self.engine = engine
        self.cfg = cfg
        self.stage_name = stage_name
        self.step_offset = step_offset
        self.prompt_wav = prompt_wav
        self.completed_steps = set()

    def on_step_end(self, args, state, control, **kwargs):
        if not self.cfg.audio_samples_on_steps:
            return control
        cumulative_step = cumulative_interval_step(
            state.global_step,
            self.step_offset,
            self.cfg.audio_sample_steps,
        )
        if (
            cumulative_step is None
            or cumulative_step in self.completed_steps
        ):
            return control
        self.completed_steps.add(cumulative_step)
        try:
            generate_audio_samples(
                self.engine,
                self.cfg,
                cumulative_step,
                self.stage_name,
                self.prompt_wav,
            )
        except Exception as exc:
            logger.exception(f"Audio generation failed at cumulative step {cumulative_step}: {exc}")
        return control


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
    cumulative_step = 0
    prompt_wav = find_eval_prompt(cfg)

    for stage in cfg.stages:
        logger.info(
            f"Starting {stage.name}: epochs={stage.epochs}, lr={stage.learning_rate}, manifest={stage.manifest}"
        )
        train_dataset = CurriculumManifestDataset(cfg, stage.manifest)
        args = training_arguments(cfg, stage)
        trainer = Trainer(
            model=wrapper,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=validation,
            data_collator=collator,
            callbacks=[
                AudioSamplesCallback(
                    engine,
                    cfg,
                    stage.name,
                    cumulative_step,
                    prompt_wav,
                )
            ],
        )
        trainer.create_optimizer()
        if optimizer_state is not None:
            trainer.optimizer.load_state_dict(optimizer_state)
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = stage.learning_rate
            parameter_group["initial_lr"] = stage.learning_rate
        result = trainer.train()
        optimizer_state = tensors_to_cpu(trainer.optimizer.state_dict())
        stage_steps = int(trainer.state.global_step)
        cumulative_step += stage_steps

        stage_dir = Path(args.output_dir)
        adapter_dir = save_stage_adapter(engine, stage_dir)
        torch.save(optimizer_state, stage_dir / "optimizer_stage_end.pt")
        stage_metrics = result.metrics
        if cfg.audio_samples_on_stage_end:
            try:
                generate_audio_samples(
                    engine,
                    cfg,
                    cumulative_step,
                    stage.name,
                    prompt_wav,
                )
            except Exception as exc:
                logger.exception(f"Stage-end audio generation failed for {stage.name}: {exc}")
        journal.append(
            {
                "stage": stage.name,
                "manifest": stage.manifest,
                "epochs": stage.epochs,
                "optimizer_steps": stage_steps,
                "cumulative_optimizer_step": cumulative_step,
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
