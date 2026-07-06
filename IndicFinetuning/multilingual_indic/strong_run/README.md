# Strong Run

Use this only after the pilot and first real run sound stable.

Recommended target:

```text
12 languages x 20 hours each = 240 hours
10h female + 10h male per language when gender metadata is available
```

Start from the same 12-language tokenizer:

```bash
./IndicFinetuning/tokenizer/tokenizer_indic_12lang.json
```

Suggested config changes from `first_real_run/config.py`:

```python
csv_path = "./IndicFinetuning/multilingual_indic/strong_run/dataset/metadata.csv"
wav_dir = "./IndicFinetuning/multilingual_indic/strong_run/dataset/wavs"
preprocessed_dir = "./IndicFinetuning/multilingual_indic/strong_run/dataset/preprocess"
output_dir = "./IndicFinetuning/outputs/multilingual_indic_strong_run"
learning_rate = 6e-5
num_epochs = 2
save_steps = 2000
```

Recommended hardware: A100 80GB or H100. A100 40GB can work with lower batch size, but preprocessing and training will take much longer.
