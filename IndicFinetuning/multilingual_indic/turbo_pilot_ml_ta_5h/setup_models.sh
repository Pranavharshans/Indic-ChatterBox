#!/usr/bin/env bash
set -euo pipefail

python setup.py \
  --mode turbo \
  --dest-dir ./pretrained_models_turbo \
  --languages ml ta

python - <<'PY'
from transformers import AutoTokenizer

path = "./pretrained_models_turbo"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
vocab = tokenizer.get_vocab()

for token in ("[ml]", "[ta]"):
    if token not in vocab:
        raise RuntimeError(f"Required Turbo token missing: {token}")

print("Turbo tokenizer size:", len(tokenizer))
print("Malayalam tag ID:", vocab["[ml]"])
print("Tamil tag ID:", vocab["[ta]"])
PY
