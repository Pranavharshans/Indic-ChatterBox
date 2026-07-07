#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN before running this script}"

python IndicFinetuning/multilingual_indic/prepare_rasa_multilingual.py \
  --dataset ai4bharat/Rasa \
  --output ./IndicFinetuning/multilingual_indic/pilot/dataset \
  --languages hi ta te ml kn bn mr gu pa ur or as \
  --hours-per-language 2 \
  --hf-token "$HF_TOKEN" \
  --split train \
  --min-duration 1.0 \
  --max-duration 15.0 \
  --streaming \
  --shuffle-buffer 256 \
  --progress-every 100
