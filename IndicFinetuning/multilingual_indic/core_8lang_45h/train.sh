#!/usr/bin/env bash
set -euo pipefail

python IndicFinetuning/tokenizer/build_indic_tokenizer.py \
  --base-tokenizer ./pretrained_models/tokenizer.json \
  --output-tokenizer ./IndicFinetuning/tokenizer/tokenizer_indic_core_8lang.json \
  --languages hi bn mr gu ta te ml kn

python IndicFinetuning/multilingual_indic/train_multilingual.py \
  --config-file ./IndicFinetuning/multilingual_indic/core_8lang_45h/config.py
