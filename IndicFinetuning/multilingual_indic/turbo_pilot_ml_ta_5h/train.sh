#!/usr/bin/env bash
set -euo pipefail

python IndicFinetuning/multilingual_indic/train_multilingual.py \
  --config-file ./IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/config.py
