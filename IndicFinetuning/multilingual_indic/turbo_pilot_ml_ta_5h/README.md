# Turbo Malayalam/Tamil Pilot

This isolated pilot checks whether Chatterbox Turbo can learn Malayalam and
Tamil before committing to a large multilingual Turbo run.

## Scope

| Setting | Value |
| --- | ---: |
| Languages | Malayalam (`ml`), Tamil (`ta`) |
| Unique data | 5 hours per language |
| Gender target | 2.5h female + 2.5h male per language |
| Epochs | 2 |
| Batch / accumulation | 8 / 2 |
| Effective batch | 16 |
| Learning rate | 5e-5 |
| LoRA rank / alpha | 64 / 128 |
| Checkpoint and eval interval | 250 steps |

Turbo models and tokenizer files are stored in `pretrained_models_turbo`, so
the standard Chatterbox files under `pretrained_models` remain untouched.
Preprocessed data and outputs are also separate from every non-Turbo run.

Run everything from the repository root:

```bash
export HF_TOKEN="hf_your_token_here"
bash IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/run_all.sh
```

Outputs are written to:

```text
IndicFinetuning/outputs/turbo_pilot_ml_ta_5h/
```

Checkpoint audio is generated for Malayalam and Tamil every 250 steps. Compare
the checkpoint nearest the end of epoch one with the final checkpoint to detect
second-pass naturalness loss.

After training, generate custom Tamil text with the final adapter:

```bash
python IndicFinetuning/multilingual_indic/infer_text.py \
  --config-file IndicFinetuning/multilingual_indic/turbo_pilot_ml_ta_5h/config.py \
  --adapter-path IndicFinetuning/outputs/turbo_pilot_ml_ta_5h/indic_adapter \
  --prompt-wav /path/to/reference_longer_than_5_seconds.wav \
  --language ta \
  --text "தினமும் காலையில் எழுந்து நான் தேநீர் குடிப்பேன்." \
  --output IndicFinetuning/outputs/turbo_pilot_ml_ta_5h/tamil_test.wav
```
