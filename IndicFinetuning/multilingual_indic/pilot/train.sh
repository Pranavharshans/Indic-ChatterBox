#!/usr/bin/env bash
set -euo pipefail

python IndicFinetuning/tokenizer/build_indic_tokenizer.py \
  --base-tokenizer ./pretrained_models/tokenizer.json \
  --output-tokenizer ./IndicFinetuning/tokenizer/tokenizer_indic_12lang.json \
  --languages hi ta te ml kn bn mr gu pa ur or as

python IndicFinetuning/multilingual_indic/train_multilingual.py \
  --config-file ./IndicFinetuning/multilingual_indic/pilot/config.py
