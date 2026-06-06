from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IndicTrainConfig:
    # Base Chatterbox model files downloaded by setup.py.
    model_dir: str = "./pretrained_models"

    # Dataset layout. Mixed-language metadata should include a language column:
    # filename|raw_text|normalized_text|language_id
    csv_path: str = "./IndicFinetuning/datasets/MalayalamDataset/metadata.csv"
    metadata_path: str = "./IndicFinetuning/datasets/metadata.json"
    wav_dir: str = "./IndicFinetuning/datasets/MalayalamDataset/wavs"
    preprocessed_dir: str = "./IndicFinetuning/datasets/MalayalamDataset/preprocess"
    output_dir: str = "./IndicFinetuning/outputs"

    # Model selection.
    is_turbo: bool = False
    is_lora: bool = True

    # Toggle languages here. For single-language Malayalam training, keep ["ml"].
    target_languages: List[str] = field(default_factory=lambda: ["ml"])
    default_language: str = "ml"
    metadata_language_column: Optional[int] = 3
    add_language_tag: bool = True
    normalize_unicode: str = "NFC"

    # Dataset format.
    ljspeech: bool = True
    json_format: bool = False
    preprocess: bool = True

    # Inference smoke test.
    is_inference: bool = False
    inference_language: str = "ml"
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "ഇത് മലയാളത്തിൽ ഒരു പരീക്ഷണ വാക്യമാണ്."

    # Vocabulary. Update after building the Indic tokenizer.
    new_vocab_size: int = 2454

    # LoRA.
    lora_r: int = 128
    lora_alpha: int = 256
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "spkr_enc"])
    turbo_lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc", "spkr_enc"])
    lora_modules_to_save: List[str] = field(default_factory=lambda: ["text_emb", "text_head"])

    # Training.
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 10
    save_steps: int = 500
    save_total_limit: int = 5
    dataloader_num_workers: int = 8

    # Sequence constraints.
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = 3.0
