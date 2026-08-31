import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.config_indic import IndicTrainConfig
from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class
from IndicFinetuning.indic_text import (
    apply_language_tag,
    normalize_indic_text,
    read_file_based_rows,
    read_json_rows,
    read_ljspeech_rows,
)
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger


logger = setup_logger(__name__)


def load_rows(config):
    if config.ljspeech:
        return read_ljspeech_rows(config.csv_path, config.default_language, config.metadata_language_column)
    if config.json_format:
        return read_json_rows(config.metadata_path, config.default_language)
    return read_file_based_rows(config.wav_dir, config.default_language)


def wav_path_for_row(config, file_id: str) -> Path:
    if isinstance(file_id, dict):
        row = file_id
        if row.get("audio_path"):
            return Path(row["audio_path"])
        file_id = row["id"]
    filename = file_id if file_id.endswith(".wav") else f"{file_id}.wav"
    return Path(config.wav_dir) / filename


def load_wav_without_torchcodec(wav_path: Path):
    audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    wav = torch.from_numpy(audio.T)
    return wav, int(sr)


def tokenize_text(config, tts_engine, text: str, language_id: str):
    clean_text = normalize_indic_text(text, config.normalize_unicode)
    if config.is_turbo:
        clean_text = apply_language_tag(clean_text, language_id, config.add_language_tag)
        token_output = tts_engine.tokenizer(clean_text, return_tensors="pt")
        raw_text_tokens = token_output.input_ids[0].cpu()
        if tts_engine.tokenizer.eos_token_id is not None:
            text_eos = torch.tensor([tts_engine.tokenizer.eos_token_id], dtype=raw_text_tokens.dtype)
            return torch.cat([raw_text_tokens, text_eos], dim=0)
        return raw_text_tokens
    return tts_engine.tokenizer.text_to_tokens(
        clean_text,
        language_id=language_id if config.add_language_tag else None,
    ).squeeze(0).cpu()


def preprocess_rows_indic(config, tts_engine, rows, output_dir=None, skip_existing=False):
    output_dir = Path(output_dir or config.preprocessed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_engine.ve.to(device).eval()
    tts_engine.s3gen.to(device).eval()

    logger.info(f"Processing Indic dataset. Total rows: {len(rows)}")
    success_count = 0
    speech_stop_id = getattr(tts_engine.t3.hp, "stop_speech_token", 6562)

    for row in tqdm(rows, desc="Indic preprocessing"):
        file_id = row["id"]
        language_id = row["language_id"]
        source = str(row.get("source", "")).strip()
        save_dir = output_dir / source if source else output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{Path(file_id).stem}.pt"
        if skip_existing and save_path.exists():
            success_count += 1
            continue
        try:
            wav_path = wav_path_for_row(config, row)
            if not wav_path.exists():
                logger.warning(f"Audio file not found, skipping: {wav_path}")
                continue

            wav, sr = load_wav_without_torchcodec(wav_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != S3_SR:
                wav = torchaudio.transforms.Resample(sr, S3_SR)(wav)
            wav = wav.to(device)

            with torch.no_grad():
                wav_np = wav.cpu().squeeze().numpy()
                spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
                speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()

                s_tokens, _ = tts_engine.s3gen.tokenizer(wav.unsqueeze(0))
                raw_speech_tokens = s_tokens.squeeze().cpu()
                stop_speech_tensor = torch.tensor([speech_stop_id], dtype=raw_speech_tokens.dtype)
                speech_tokens = torch.cat([raw_speech_tokens, stop_speech_tensor], dim=0)

                prompt_samples = int(config.prompt_duration * S3_SR)
                if wav.shape[1] < prompt_samples:
                    prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
                else:
                    prompt_wav = wav[:, :prompt_samples]

                p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0))
                prompt_tokens = p_tokens.squeeze().cpu()

            text_tokens = tokenize_text(config, tts_engine, row["text"], language_id)
            torch.save(
                {
                    "speech_tokens": speech_tokens,
                    "speaker_emb": speaker_emb,
                    "prompt_tokens": prompt_tokens,
                    "text_tokens": text_tokens,
                    "language_id": language_id,
                    "source": source,
                    "speaker_id": str(row.get("speaker_id", "")),
                },
                save_path,
            )
            success_count += 1
        except Exception as exc:
            logger.error(f"Error preprocessing {file_id}: {exc}")

    logger.info(f"Indic preprocessing completed. Success: {success_count}/{len(rows)}")
    return success_count


def preprocess_dataset_indic(config, tts_engine):
    rows = load_rows(config)
    return preprocess_rows_indic(config, tts_engine, rows)


if __name__ == "__main__":
    cfg = IndicTrainConfig()
    engine_class = get_engine_class(cfg.is_turbo)
    engine = engine_class.from_local(cfg.model_dir, device="cpu")
    engine = attach_indic_tokenizer(engine, cfg)
    preprocess_dataset_indic(cfg, engine)
