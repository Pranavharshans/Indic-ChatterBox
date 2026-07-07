# Strong Run

Use this after the first real run if the adapter learns language/accent identity but needs better text following.

Target:

```text
12 languages x 20 hours each = 240 hours
10h female + 10h male per language when gender metadata is available
```

Start from the same 12-language tokenizer:

```bash
./IndicFinetuning/tokenizer/tokenizer_indic_12lang.json
```

Prepare and train:

```bash
export HF_TOKEN="hf_your_token_here"
bash IndicFinetuning/multilingual_indic/strong_run/prepare_dataset.sh
bash IndicFinetuning/multilingual_indic/strong_run/train.sh
```

Recommended hardware: A100 80GB or H100. A 24GB GPU may work with `batch_size = 24` if the first real run did, but lower it to `16` if it OOMs or slows down.
