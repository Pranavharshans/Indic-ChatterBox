#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN before running this script}"

bash IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/setup_models.sh
bash IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/prepare_dataset.sh
bash IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/train.sh
