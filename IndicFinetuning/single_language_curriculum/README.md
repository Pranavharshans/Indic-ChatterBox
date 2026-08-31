# Malayalam Rasa + IV-R curriculum

This run trains one continuous Chatterbox LoRA adapter across three fixed stages:

| Stage | Pilot steps | Dataset manifest | Learning rate |
| --- | ---: | --- | ---: |
| 1 | 6,000 | 100% filtered Malayalam IV-R | `6e-5` |
| 2 | 6,000 | 50% Rasa / 50% IV-R | `3e-5` |
| 3 | 3,000 | 80% Rasa / 20% expressive-clean IV-R | `1e-5` |

The model stays in one Python process. Model weights and Adam optimizer moments continue between stages; only the fixed manifest and learning rate change. There is no speaker-balanced runtime sampler.

## Data contract

Export both Hugging Face datasets to local WAV files and UTF-8 JSONL catalogs:

```bash
export HF_TOKEN="your_hugging_face_token"
python IndicFinetuning/single_language_curriculum/export_hf_catalogs.py \
  --output /data/malayalam/curriculum \
  --hf-token "$HF_TOKEN" \
  --resume
```

Rasa is gated, so accept its terms on the Hub first. The exporter uses streaming metadata, filters IV-R before decoding rejected audio, records dataset/config/revision information, and can resume from its append-only catalogs. Each catalog has one row per recording:

Plan for substantial local storage: the source WAVs, preprocessed tensors, checkpoints, and temporary Hub cache can require well over 100 GB together.

```json
{"id":"rasa_000001","source":"rasa","text":"...","audio_path":"/absolute/path.wav","speaker_id":"female","gender":"female","language_id":"ml","duration":5.2}
{"id":"ivr_000001","source":"ivr","text":"...","audio_path":"/absolute/path.wav","speaker_id":"S425...","gender":"female","language_id":"ml","duration":6.1,"scenario":"Extempore","cer":0.03,"snr":31.2,"c50":28.0,"pitch_std":24.1,"speaking_rate":16.0}
```

The intended sources are [`ai4bharat/Rasa`](https://huggingface.co/datasets/ai4bharat/Rasa), Malayalam (33,851 released utterances), and [`trysem/indicvoices_r-ML`](https://huggingface.co/datasets/trysem/indicvoices_r-ML) (31,106 utterances). Rasa is gated and requires accepting its Hub terms before downloading.

Build the fixed 90/5/5 splits and stage manifests:

```bash
python IndicFinetuning/single_language_curriculum/build_plan.py \
  --rasa-catalog /data/malayalam/curriculum/rasa/catalog.jsonl \
  --ivr-catalog /data/malayalam/curriculum/ivr/catalog.jsonl \
  --output ./IndicFinetuning/single_language_curriculum/work/plan \
  --seed 42
```

The planner:

- removes duplicate recordings and per-speaker duplicate transcripts before splitting while preserving valid recordings of the same sentence by different speakers;
- keeps both Rasa speakers in train, validation, and test;
- keeps IV-R validation/test speakers disjoint from training;
- filters IV-R to 2–20 seconds, CER <= 0.15, SNR >= 20 dB, and C50 >= 10, then removes the outer 1% speaking-rate tails;
- derives expressive-clean IV-R only from the IV-R training split;
- cycles the smaller source in Stage 2 so the fixed manifest is exactly 50/50 without dropping the larger 30–33k source;
- writes separate untouched Rasa and IV-R test manifests.

Inspect `work/plan/summary.json` before preprocessing or training. In particular, confirm that the expressive-clean pool is large enough and listen to a representative sample.

## Train

Build the existing Malayalam tokenizer and base model files first, then run:

```bash
bash IndicFinetuning/single_language_curriculum/train.sh
```

Preprocessing is source-aware and writes each encoded sample once under `work/preprocessed/<source>/`. Each stage writes its own checkpoints, adapter, optimizer boundary state, and metrics under `IndicFinetuning/outputs/malayalam_curriculum_pilot/`. The final adapter is `final_adapter/`.

Do not select the final model from loss alone. Compare fixed Malayalam prompts and unseen reference speakers at the base model and every stage boundary before expanding beyond the 15k-step pilot.
