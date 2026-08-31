import os
import random
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import setup_logger


logger = setup_logger(__name__)



class ChatterboxDataset(Dataset):
    
    def __init__(self, config):
        self.cfg = config
        self.preprocessed_dir = config.preprocessed_dir
        
        if not os.path.exists(self.preprocessed_dir):
            raise FileNotFoundError(f"Preprocessing folder not found: {self.preprocessed_dir}.")
            
        self.files = [f for f in os.listdir(self.preprocessed_dir) if f.endswith(".pt")]
        
        if len(self.files) == 0:
            raise RuntimeError(f"There are no .pt files in the folder: {self.preprocessed_dir}")
            
        logger.info(f"Dataset loaded. Total sample: {len(self.files)}")

        self.sot_token = config.start_text_token 
        self.eot_token = config.stop_text_token


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        try:
            
            filename = self.files[idx]
            
            pt_path = os.path.join(self.preprocessed_dir, filename)
            
            data = torch.load(pt_path)
            
            
            text_tokens = data["text_tokens"]
            if text_tokens.size(0) > self.cfg.max_text_len - 2:
                text_tokens = text_tokens[:self.cfg.max_text_len - 2]
                
            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            speech_tokens = data["speech_tokens"]
            if speech_tokens.size(0) > self.cfg.max_speech_len:
                speech_tokens = speech_tokens[:self.cfg.max_speech_len]

            speaker_emb = data["speaker_emb"]
            prompt_tokens = data["prompt_tokens"]

            if random.random() < 0.20:
                speaker_emb = torch.zeros_like(speaker_emb)
                prompt_tokens = torch.zeros(1, dtype=torch.long)


            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens
            }


        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None


class CurriculumManifestDataset(Dataset):
    """Load preprocessed samples through a fixed source-aware JSONL manifest."""

    def __init__(self, config, manifest_path, conditioning_dropout=True):
        self.cfg = config
        self.preprocessed_dir = Path(config.preprocessed_dir)
        self.rows = []
        self.conditioning_dropout = conditioning_dropout
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("id") or not row.get("source"):
                    raise ValueError(f"Missing id/source in {manifest_path}:{line_number}")
                self.rows.append(row)
        if not self.rows:
            raise RuntimeError(f"Curriculum manifest is empty: {manifest_path}")
        missing = []
        for row in self.rows:
            sample_path = self.preprocessed_dir / row["source"] / f"{Path(row['id']).stem}.pt"
            if not sample_path.exists():
                missing.append(str(sample_path))
                if len(missing) == 5:
                    break
        if missing:
            raise FileNotFoundError(
                f"Preprocessed curriculum samples are missing for {manifest_path}: {missing}"
            )
        self.sot_token = config.start_text_token
        self.eot_token = config.stop_text_token
        logger.info(f"Curriculum manifest loaded: {manifest_path} ({len(self.rows)} samples)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pt_path = self.preprocessed_dir / row["source"] / f"{Path(row['id']).stem}.pt"
        try:
            data = torch.load(pt_path)
            text_tokens = data["text_tokens"][: self.cfg.max_text_len - 2]
            text_tokens = torch.cat(
                [
                    torch.tensor([self.sot_token], dtype=torch.long),
                    text_tokens,
                    torch.tensor([self.eot_token], dtype=torch.long),
                ]
            )
            speech_tokens = data["speech_tokens"][: self.cfg.max_speech_len]
            speaker_emb = data["speaker_emb"]
            prompt_tokens = data["prompt_tokens"]
            if self.conditioning_dropout and random.random() < 0.20:
                speaker_emb = torch.zeros_like(speaker_emb)
                prompt_tokens = torch.zeros(1, dtype=torch.long)
            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
            }
        except Exception as exc:
            logger.error(f"Error loading curriculum sample {pt_path}: {exc}")
            return None


def data_collator_standart(batch):

    batch = [item for item in batch if item is not None]
    if not batch: 
        return {}

    # Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)

    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    # Lengths
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)


    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens
    }
    
    


def data_collator_turbo(batch):

    batch = [item for item in batch if item is not None]
    if not batch: 
        return {}

    # 1. Text Tokens Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)

    # 2. Speech Tokens Padding
    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    # 3. Prompt Tokens Padding
    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_lens = torch.tensor([x["prompt_tokens"].shape[0] for x in batch], dtype=torch.long)

    # 4. Speaker Embedding
    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens,
        "prompt_lens": prompt_lens
    }
