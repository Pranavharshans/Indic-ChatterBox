import argparse
import importlib.util
import os
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.config_indic import IndicTrainConfig
from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from src.chatterbox_.models.t3.t3 import T3
from src.model import ChatterboxTrainerWrapper, resize_and_load_t3_weights
from src.utils import setup_logger


logger = setup_logger("Indic-Checkpoint-Exporter")


def load_config(config_file: str, class_name: str):
    path = Path(config_file).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)()


def load_checkpoint_state(checkpoint_dir: Path):
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        return load_file(str(safetensors_path), device="cpu")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")


def build_lora_model(cfg: IndicTrainConfig):
    from peft import LoraConfig, get_peft_model

    inferred_vocab_size = tokenizer_vocab_size(cfg)
    if inferred_vocab_size != cfg.new_vocab_size:
        logger.warning(f"Configured new_vocab_size={cfg.new_vocab_size}, tokenizer has {inferred_vocab_size}. Using tokenizer size.")
        cfg.new_vocab_size = inferred_vocab_size

    engine_class = get_engine_class(cfg.is_turbo)
    base_engine = engine_class.from_local(cfg.model_dir, device="cpu")
    base_engine = attach_indic_tokenizer(base_engine, cfg)

    pretrained_t3_state_dict = base_engine.t3.state_dict()
    new_t3_config = base_engine.t3.hp
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
    setattr(new_t3_config, "use_cache", False)

    new_t3_model = T3(hp=new_t3_config)
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)
    if cfg.is_turbo and hasattr(new_t3_model.tfmr, "wte"):
        del new_t3_model.tfmr.wte

    for param in new_t3_model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.turbo_lora_target_modules if cfg.is_turbo else cfg.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=cfg.lora_modules_to_save,
    )
    return get_peft_model(new_t3_model, peft_config)


def main():
    parser = argparse.ArgumentParser(description="Export a Trainer checkpoint into a PEFT adapter folder for Indic inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./IndicFinetuning/outputs/indic_adapter_test")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--config-class", default="MultilingualIndicConfig")
    args = parser.parse_args()

    cfg = load_config(args.config_file, args.config_class) if args.config_file else IndicTrainConfig()
    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t3_model = build_lora_model(cfg)
    wrapper = ChatterboxTrainerWrapper(t3_model)
    checkpoint_state = load_checkpoint_state(checkpoint_dir)

    missing, unexpected = wrapper.load_state_dict(checkpoint_state, strict=False)
    if len(unexpected) > 0:
        logger.warning(f"Unexpected checkpoint keys: {len(unexpected)}")
    if len(missing) > 0:
        logger.warning(f"Missing checkpoint keys: {len(missing)}")

    wrapper.t3.save_pretrained(str(output_dir))
    logger.info(f"Exported PEFT adapter to: {output_dir}")
    logger.info("Use it with: INDIC_ADAPTER_PATH=%s python IndicFinetuning/inference_indic.py", output_dir)


if __name__ == "__main__":
    main()
