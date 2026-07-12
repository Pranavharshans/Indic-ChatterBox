# Core 8-Language 45-Hour Run

This run trains one LoRA adapter for eight core Indic languages using one pass
over substantially more unique audio than RIMA-TTS v1.

Languages:

```text
hi bn mr gu ta te ml kn
```

Dataset plan:

- 45 hours per language
- 22.5 hours female and 22.5 hours male per language
- 360 unique hours total
- audio duration filter: 1-15 seconds
- one training epoch

Training configuration:

| Setting | Value |
| --- | ---: |
| Batch size | 24 |
| Gradient accumulation | 1 |
| Learning rate | 4e-5 |
| Epochs | 1 |
| LoRA rank / alpha | 128 / 256 |
| Checkpoint interval | 1,000 steps |
| Checkpoints retained | 6 |
| Eval samples | 4 per language every 1,000 steps |

Run from the repository root:

```bash
export HF_TOKEN="hf_your_token_here"
bash IndicFinetuning/multilingual_indic/core_8lang_45h/prepare_dataset.sh
bash IndicFinetuning/multilingual_indic/core_8lang_45h/train.sh
```

The dataset is written under `core_8lang_45h/dataset`. Training outputs and
checkpoint listening samples are written under
`IndicFinetuning/outputs/multilingual_indic_core_8lang_45h`.

This run starts from the Chatterbox base model. It does not continue the
RIMA-TTS v1 adapter, keeping the comparison with the earlier 12-language run
clean.
