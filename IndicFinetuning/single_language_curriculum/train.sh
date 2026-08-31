#!/usr/bin/env bash
set -euo pipefail

python IndicFinetuning/single_language_curriculum/train_curriculum.py \
  --config-file ./IndicFinetuning/single_language_curriculum/config.py
