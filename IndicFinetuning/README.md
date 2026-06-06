# Indic Chatterbox Fine-Tuning

This folder keeps Indic-language fine-tuning separate from the upstream Chatterbox training kit.

The default setup targets Malayalam (`ml`) on the non-turbo model. To switch languages, edit `target_languages`, `default_language`, and `inference_language` in `config_indic.py`.

## Folder Layout

```text
IndicFinetuning/
├── config_indic.py
├── train_indic.py
├── inference_indic.py
├── preprocess_indic.py
├── indic_languages.py
├── tokenizer/
│   ├── audit_tokenizer.py
│   └── build_indic_tokenizer.py
├── datasets/
│   └── MalayalamDataset/
│       ├── metadata.csv
│       └── wavs/
└── outputs/
```

## Metadata Format

Use one row per utterance:

```text
file_id|raw_text|normalized_text|language_id
```

Example:

```text
segment_001|ഇത് മലയാളം വാക്യമാണ്.|ഇത് മലയാളം വാക്യമാണ്.|ml
```

For mixed-language training, keep the same format and change the final column:

```text
segment_ml_001|...|മലയാളം വാചകം|ml
segment_ta_001|...|தமிழ் வாக்கியம்|ta
segment_te_001|...|తెలుగు వాక్యం|te
```

## Malayalam Non-Turbo Workflow

1. Download the base Chatterbox files from the repo root:

```bash
python setup.py
```

2. Build an Indic tokenizer:

```bash
python IndicFinetuning/tokenizer/build_indic_tokenizer.py \
  --base-tokenizer ./pretrained_models/tokenizer.json \
  --output-tokenizer ./IndicFinetuning/tokenizer/tokenizer_indic.json \
  --languages ml
```

3. Copy the printed final vocab size into `new_vocab_size` in `IndicFinetuning/config_indic.py`.

4. Drop data into:

```text
IndicFinetuning/datasets/MalayalamDataset/metadata.csv
IndicFinetuning/datasets/MalayalamDataset/wavs/
```

If your data is on Hugging Face, convert it first:

```bash
pip install datasets

python IndicFinetuning/prepare_hf_dataset.py \
  --dataset your-user/your-malayalam-dataset \
  --split train \
  --output ./IndicFinetuning/datasets/MalayalamDataset \
  --audio-column audio \
  --text-column text \
  --default-language ml
```

For a multilingual HF dataset with a language column:

```bash
python IndicFinetuning/prepare_hf_dataset.py \
  --dataset your-user/your-indic-dataset \
  --split train \
  --output ./IndicFinetuning/datasets/MalayalamDataset \
  --audio-column audio \
  --text-column text \
  --language-column language_id \
  --default-language ml
```

5. Audit tokenizer coverage:

```bash
python IndicFinetuning/tokenizer/audit_tokenizer.py \
  --tokenizer ./IndicFinetuning/tokenizer/tokenizer_indic.json \
  --languages ml \
  --metadata ./IndicFinetuning/datasets/MalayalamDataset/metadata.csv
```

6. Train:

```bash
python IndicFinetuning/train_indic.py
```

7. Run inference:

```bash
python IndicFinetuning/inference_indic.py
```

## Switching Languages

For Tamil:

```python
target_languages = ["ta"]
default_language = "ta"
inference_language = "ta"
```

Then rebuild the tokenizer:

```bash
python IndicFinetuning/tokenizer/build_indic_tokenizer.py --languages ta
```

For multiple languages:

```python
target_languages = ["ml", "ta", "te", "kn"]
default_language = "ml"
```

Then rebuild:

```bash
python IndicFinetuning/tokenizer/build_indic_tokenizer.py --languages ml ta te kn
```

Every metadata row should include the correct language code.
