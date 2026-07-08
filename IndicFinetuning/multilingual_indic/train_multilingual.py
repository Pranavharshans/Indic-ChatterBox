import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file
from transformers import Trainer, TrainerCallback, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from IndicFinetuning.indic_text import apply_language_tag, normalize_indic_text
from IndicFinetuning.multilingual_indic.generate_eval_samples import PROMPTS
from IndicFinetuning.preprocess_indic import preprocess_dataset_indic
from src.chatterbox_.models.t3.t3 import T3
from src.dataset import ChatterboxDataset, data_collator_standart, data_collator_turbo
from src.model import ChatterboxTrainerWrapper, resize_and_load_t3_weights
from src.utils import check_pretrained_models, setup_logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = setup_logger("Multilingual-Indic-Finetune")


def load_config(config_file: str, class_name: str):
    path = Path(config_file).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)()


def format_eval_text(cfg, text: str, language_id: str) -> str:
    text = normalize_indic_text(text, cfg.normalize_unicode)
    return apply_language_tag(text, language_id, cfg.add_language_tag)


def find_eval_prompt_wav(cfg, language_id: str) -> Path:
    explicit_prompt = getattr(cfg, "eval_prompt_wav", None)
    if explicit_prompt:
        path = Path(explicit_prompt)
        if path.exists():
            return path
        raise FileNotFoundError(f"Configured eval_prompt_wav not found: {path}")

    wav_dir = Path(cfg.wav_dir)
    candidates = sorted(wav_dir.glob(f"{language_id}_*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No prompt wav candidates found for language {language_id} in {wav_dir}")

    min_duration = getattr(cfg, "eval_prompt_min_duration", 5.0)
    max_duration = getattr(cfg, "eval_prompt_max_duration", 10.0)
    for candidate in candidates:
        try:
            info = sf.info(str(candidate))
            duration = info.frames / info.samplerate
        except Exception:
            continue
        if min_duration <= duration <= max_duration:
            return candidate

    return candidates[0]


def generate_eval_samples_for_step(engine, cfg, step: int):
    output_root = Path(getattr(cfg, "eval_output_dir", os.path.join(cfg.output_dir, "eval_samples"))) / f"op-{step}"
    output_root.mkdir(parents=True, exist_ok=True)

    languages = getattr(cfg, "eval_languages", cfg.target_languages)
    samples_per_language = getattr(cfg, "eval_samples_per_language", 4)
    temperature = getattr(cfg, "eval_temperature", 0.8)
    repetition_penalty = getattr(cfg, "eval_repetition_penalty", 1.2)

    manifest = []
    was_training = engine.t3.training
    t3_device = next(engine.t3.parameters()).device
    s3gen_device = next(engine.s3gen.parameters()).device
    ve_device = next(engine.ve.parameters()).device
    original_engine_device = engine.device

    engine.t3.eval()
    engine.s3gen.to(t3_device).eval()
    engine.ve.to(t3_device).eval()
    engine.device = str(t3_device)

    try:
        with torch.no_grad():
            for language_id in languages:
                if language_id not in PROMPTS:
                    logger.warning(f"No eval prompts configured for language: {language_id}")
                    continue

                prompt_wav = find_eval_prompt_wav(cfg, language_id)
                language_dir = output_root / language_id
                language_dir.mkdir(parents=True, exist_ok=True)
                prompt_lines = []

                for index, prompt in enumerate(PROMPTS[language_id][:samples_per_language], start=1):
                    output_file = language_dir / f"{language_id}_{index:02d}.wav"
                    formatted_text = format_eval_text(cfg, prompt, language_id)
                    try:
                        wav_tensor = engine.generate(
                            text=formatted_text,
                            audio_prompt_path=str(prompt_wav),
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                        if isinstance(wav_tensor, tuple):
                            wav_tensor = wav_tensor[0]
                        wav_np = wav_tensor.squeeze().detach().cpu().numpy()
                        sf.write(str(output_file), wav_np, engine.sr)
                        manifest.append(
                            {
                                "step": step,
                                "language": language_id,
                                "index": index,
                                "text": prompt,
                                "prompt_wav": str(prompt_wav),
                                "audio": str(output_file),
                            }
                        )
                        prompt_lines.append(f"{index}. {prompt}")
                    except Exception as exc:
                        logger.exception(f"Eval generation failed for {language_id} sample {index} at step {step}: {exc}")

                (language_dir / "prompts.txt").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    finally:
        engine.t3.train(was_training)
        engine.s3gen.to(s3gen_device)
        engine.ve.to(ve_device)
        engine.device = original_engine_device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Checkpoint eval samples saved to: {output_root}")


class EvalSamplesOnSaveCallback(TrainerCallback):
    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.completed_steps = set()

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        eval_steps = getattr(self.cfg, "eval_sample_steps", getattr(self.cfg, "save_steps", 1000))
        if step <= 0 or step in self.completed_steps or step % eval_steps != 0:
            return control

        self.completed_steps.add(step)
        logger.info(f"Generating multilingual eval samples for checkpoint step {step}")
        try:
            generate_eval_samples_for_step(self.engine, self.cfg, step)
        except Exception as exc:
            logger.exception(f"Checkpoint eval generation failed at step {step}: {exc}")
        return control


def run_training(cfg):
    logger.info("--- Starting Multilingual Indic Finetuning ---")
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")
    logger.info(f"Languages: {', '.join(cfg.target_languages)}")
    logger.info(f"Dataset: {cfg.csv_path}")

    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        sys.exit(1)

    if not cfg.is_turbo:
        inferred_vocab_size = tokenizer_vocab_size(cfg)
        if inferred_vocab_size != cfg.new_vocab_size:
            logger.warning(f"Configured new_vocab_size={cfg.new_vocab_size}, tokenizer has {inferred_vocab_size}. Using tokenizer size.")
            cfg.new_vocab_size = inferred_vocab_size

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

    continue_adapter_path = getattr(cfg, "continue_adapter_path", None)
    if cfg.is_lora:
        if continue_adapter_path:
            from peft import PeftModel

            if not Path(continue_adapter_path).exists():
                raise FileNotFoundError(f"Continuation adapter not found: {continue_adapter_path}")
            for param in tts_engine_new.t3.parameters():
                param.requires_grad = False
            logger.info(f"Continuing trainable adapter: {continue_adapter_path}")
            tts_engine_new.t3 = PeftModel.from_pretrained(tts_engine_new.t3, continue_adapter_path, is_trainable=True)
        else:
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
        logging_steps=getattr(cfg, "logging_steps", 50),
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

    callbacks = []
    if getattr(cfg, "eval_on_save", False):
        callbacks.append(EvalSamplesOnSaveCallback(tts_engine_new, cfg))

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=selected_collator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=getattr(cfg, "resume_from_checkpoint", None))

    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.is_lora:
        save_path = os.path.join(cfg.output_dir, "indic_adapter")
        tts_engine_new.t3.save_pretrained(save_path)
        logger.info(f"Multilingual Indic LoRA adapter saved to: {save_path}")
    else:
        filename = "t3_turbo_multilingual_indic_finetuned.safetensors" if cfg.is_turbo else "t3_multilingual_indic_finetuned.safetensors"
        final_model_path = os.path.join(cfg.output_dir, filename)
        save_file(tts_engine_new.t3.state_dict(), final_model_path)
        logger.info(f"Multilingual Indic full model saved to: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a configurable multilingual Indic Chatterbox adapter.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--config-class", default="MultilingualIndicConfig")
    args = parser.parse_args()
    run_training(load_config(args.config_file, args.config_class))


if __name__ == "__main__":
    main()
