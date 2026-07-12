#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN before running this script}"

python IndicFinetuning/multilingual_indic/prepare_rasa_multilingual.py \
  --dataset ai4bharat/Rasa \
  --output ./IndicFinetuning/multilingual_indic/core_8lang_45h/dataset \
  --languages hi bn mr gu ta te ml kn \
  --hours-per-language 45 \
  --hf-token "$HF_TOKEN" \
  --split train \
  --min-duration 1.0 \
  --max-duration 15.0 \
  --streaming \
  --shuffle-buffer 1024 \
  --progress-every 100
