# Multilingual Indic Finetuning

This folder keeps large multilingual Indic runs separate from the Malayalam and emotion-tag experiments.

Selected 12-language set:

```text
hi ta te ml kn bn mr gu pa ur or as
```

Run order:

1. `pilot`: 2 hours per language, 24 hours total. Use this to validate tokenizer, Rasa loading, preprocessing, and inference quality.
2. `first_real_run`: 10 hours per language, 120 hours total. Use this after the pilot sounds stable.
3. `strong_run`: 20 hours per language, 240 hours total. Keep this for later, after the first real run proves quality.
4. `core_8lang_45h`: 45 hours each for Hindi, Bengali, Marathi, Gujarati, Tamil, Telugu, Malayalam, and Kannada. This is the one-epoch, gender-balanced follow-up to RIMA-TTS v1.

Build the 12-language tokenizer once before any run:

```bash
python IndicFinetuning/tokenizer/build_indic_tokenizer.py \
  --base-tokenizer ./pretrained_models/tokenizer.json \
  --output-tokenizer ./IndicFinetuning/tokenizer/tokenizer_indic_12lang.json \
  --languages hi ta te ml kn bn mr gu pa ur or as
```

Prepare and train the pilot:

```bash
export HF_TOKEN="hf_your_token_here"
bash IndicFinetuning/multilingual_indic/pilot/prepare_dataset.sh
bash IndicFinetuning/multilingual_indic/pilot/train.sh
```

Prepare and train the first real run:

```bash
export HF_TOKEN="hf_your_token_here"
bash IndicFinetuning/multilingual_indic/first_real_run/prepare_dataset.sh
bash IndicFinetuning/multilingual_indic/first_real_run/train.sh
```

Prepare and train the core eight-language run:

```bash
export HF_TOKEN="hf_your_token_here"
bash IndicFinetuning/multilingual_indic/core_8lang_45h/prepare_dataset.sh
bash IndicFinetuning/multilingual_indic/core_8lang_45h/train.sh
```

Hardware guidance:

| Run | Dataset Size | Minimum | Recommended |
| --- | ---: | --- | --- |
| Pilot | ~24h | 1x RTX 3090/4090/A5000 24GB | 1x A100 40GB |
| First real run | ~120h | 1x A100 40GB | 1x A100 80GB or H100 |
| Strong run | ~240h | 1x A100 80GB | 2x A100/H100 if the trainer is adapted for multi-GPU |

The current trainer is single-process Hugging Face `Trainer`. For now, prefer one strong GPU over multiple weak GPUs. If a 24GB card OOMs, lower `batch_size` in the run config to `8` or `12`.
