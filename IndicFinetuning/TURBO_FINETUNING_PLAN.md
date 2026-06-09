# Indic Chatterbox Turbo Finetuning Plan

This plan is for adding Malayalam/Indic support to the **Chatterbox Turbo** model without disturbing the working non-Turbo flow.

## Current Code State

Turbo support already exists in the repo:

- `src/chatterbox_/tts_turbo.py` loads `ResembleAI/chatterbox-turbo`.
- `setup.py` downloads Turbo assets when `is_turbo=True`.
- `train.py` and `IndicFinetuning/train_indic.py` both branch on `cfg.is_turbo`.
- `src.dataset.data_collator_turbo` is already available.
- `IndicFinetuning/preprocess_indic.py` tokenizes Turbo text through the Hugging Face tokenizer path.

Important difference from non-Turbo:

- Non-Turbo uses `MTLTokenizer` and `tokenizer_indic.json`.
- Turbo uses a Hugging Face tokenizer folder under `pretrained_models/`.
- `setup.py` currently merges the expanded grapheme vocab into the Turbo GPT-style tokenizer and saves it back into `pretrained_models/`.

## Main Risk

The main risk is not the training loop. It is **tokenizer and vocab consistency**.

For Turbo, these must match exactly:

1. The tokenizer files inside `pretrained_models/`.
2. `cfg.new_vocab_size`.
3. The resized Turbo T3 text embedding/head dimensions.
4. The adapter's saved `text_emb` and `text_head` modules.

If any of these drift, inference will fail with shape mismatch errors.

## Target Output

Keep Turbo artifacts separate from non-Turbo artifacts.

Recommended output folders:

```text
IndicFinetuning/outputs/turbo_malayalam_17k_adapter/
IndicFinetuning/outputs/turbo_malayalam_expressive_adapter/
```

Recommended HF repos later:

```text
Praha-Labs/PrahaTTS-ML-Turbo-Adapter
Praha-Labs/PrahaTTS-ML-Turbo-Expressive-Adapter
```

## Implementation Plan

### 1. Add a Dedicated Turbo Config

Create a separate config file instead of editing `config_indic.py`.

Suggested file:

```text
IndicFinetuning/config_turbo_malayalam.py
```

It should subclass `IndicTrainConfig` and set:

```python
is_turbo = True
is_lora = True
csv_path = "./IndicFinetuning/datasets/MalayalamDataset/metadata.csv"
wav_dir = "./IndicFinetuning/datasets/MalayalamDataset/wavs"
preprocessed_dir = "./IndicFinetuning/datasets/MalayalamDataset/preprocess_turbo"
output_dir = "./IndicFinetuning/outputs/turbo_malayalam"
target_languages = ["ml"]
default_language = "ml"
inference_language = "ml"
batch_size = 8
grad_accum = 1
learning_rate = 1e-5
num_epochs = 4
save_steps = 500
```

Do not reuse the non-Turbo preprocessed folder. Turbo prompt/token handling differs.

### 2. Add a Dedicated Turbo Training Entrypoint

Create:

```text
IndicFinetuning/train_turbo_malayalam.py
```

It can reuse most logic from `train_indic.py`, but instantiate the Turbo config.

Key requirements:

- `cfg.is_turbo=True`
- `mode_check="chatterbox_turbo"`
- use `data_collator_turbo`
- use `cfg.turbo_lora_target_modules`
- keep `modules_to_save=["text_emb", "text_head"]`
- save adapter to `cfg.output_dir/indic_adapter`

### 3. Setup Turbo Pretrained Files

On a clean VM:

```bash
cd /workspace/Indic-ChatterBox
source .venv/bin/activate
```

Before running `setup.py`, ensure the active config used by setup has Turbo mode enabled. The original `setup.py` reads `src.config.TrainConfig`, not `IndicTrainConfig`.

Two safe options:

- temporarily set `src/config.py` `is_turbo=True`, run setup, then revert; or
- add a separate Indic setup script later that reads the Turbo Indic config.

Expected Turbo files:

```text
pretrained_models/t3_turbo_v1.safetensors
pretrained_models/s3gen_meanflow.safetensors
pretrained_models/ve.safetensors
pretrained_models/conds.pt
pretrained_models/vocab.json
pretrained_models/merges.txt
pretrained_models/tokenizer_config.json
pretrained_models/special_tokens_map.json
pretrained_models/added_tokens.json
```

After setup, verify tokenizer size:

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./pretrained_models")
print(len(tok))
print(tok.encode("മലയാളം", add_special_tokens=False))
print(tok.encode("[ml]", add_special_tokens=False))
PY
```

Update the Turbo config `new_vocab_size` to this tokenizer length.

### 4. Confirm Turbo Tokenization Before Training

Run a small tokenization check:

```bash
python - <<'PY'
from IndicFinetuning.config_turbo_malayalam import TurboMalayalamConfig
from IndicFinetuning.indic_engine import get_engine_class
from IndicFinetuning.preprocess_indic import tokenize_text

cfg = TurboMalayalamConfig()
engine = get_engine_class(True).from_local(cfg.model_dir, device="cpu")
tokens = tokenize_text(cfg, engine, "[ml] ഇത് മലയാളം ടെസ്റ്റ് ആണ്.", "ml")
print(tokens.shape)
print(tokens[:20])
PY
```

This must work before any training.

### 5. First Smoke Training

Do not start full training first. Create a small subset:

```text
IndicFinetuning/datasets/MalayalamTurboSmoke/
├── metadata.csv
└── wavs/
```

Use 50-100 samples.

Run Turbo training for one short test:

```bash
python IndicFinetuning/train_turbo_malayalam.py
```

Expected checks:

- model loads `CHATTERBOX-TURBO`
- T3 is resized to Turbo tokenizer vocab size
- trainable params print correctly
- preprocessing completes
- first loss logs without shape mismatch
- adapter saves to `IndicFinetuning/outputs/turbo_malayalam/indic_adapter`

### 6. Turbo Inference Script

`IndicFinetuning/inference_indic.py` already branches on `cfg.is_turbo`, but it currently instantiates `IndicTrainConfig()` directly.

Add a dedicated inference script:

```text
IndicFinetuning/inference_turbo_malayalam.py
```

It should use `TurboMalayalamConfig` and load:

```text
IndicFinetuning/outputs/turbo_malayalam/indic_adapter
```

Smoke inference should generate one short Malayalam WAV before full training.

### 7. Full Malayalam Turbo Training

Once smoke training and inference pass:

```text
dataset: MalayalamDataset or a cleaned 3k+ dataset
base: Chatterbox Turbo
training: LoRA
lr: 1e-5
epochs: 3-5
batch: start at 8 on 24 GB GPU, increase only after memory check
```

Use separate output:

```text
IndicFinetuning/outputs/turbo_malayalam_full/indic_adapter
```

### 8. Expressive Turbo Training

Do not start with expressive tags on Turbo. First get plain Malayalam working.

After plain Turbo Malayalam passes:

```text
base adapter: turbo_malayalam_full/indic_adapter
dataset: future 3k+ expressive dataset
output: turbo_malayalam_expressive/indic_adapter
```

Prefer an emotion-heavy dataset only after verifying the generated audio has real audible tag events.

## Open Questions Before Implementation

- Does the merged Turbo tokenizer actually preserve all needed Indic graphemes and control tags?
- Does Turbo inference preserve non-verbal tags as well as non-Turbo?
- Is `modules_to_save=["text_emb", "text_head"]` sufficient for Turbo after deleting `tfmr.wte`?
- Does Turbo need lower LR than non-Turbo because of the smaller/faster architecture?

## Recommended Next Step

Implement only these first:

1. `config_turbo_malayalam.py`
2. `train_turbo_malayalam.py`
3. `inference_turbo_malayalam.py`

Then run a 50-100 sample smoke test before committing to expensive full Turbo training.
