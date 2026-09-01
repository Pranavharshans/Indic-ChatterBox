# Malayalam Rasa + IV-R curriculum

This run trains one continuous Chatterbox LoRA adapter across three fixed stages:

| Stage | Passes | Dataset manifest | Learning rate |
| --- | ---: | --- | ---: |
| 1 | 1 | 100% filtered Malayalam IV-R | `6e-5` |
| 2 | 1 | 50% Rasa / 50% IV-R | `3e-5` |
| 3 | 1 | 80% Rasa / 20% expressive-clean IV-R | `1e-5` |

Together these three single passes are one curriculum epoch: every fixed stage manifest is traversed exactly once. The model stays in one Python process. Model weights and Adam optimizer moments continue between stages; only the fixed manifest and learning rate change. There is no speaker-balanced runtime sampler.

## Data contract

Export both Hugging Face datasets to local WAV files and UTF-8 JSONL catalogs:

```bash
hf auth login
python IndicFinetuning/single_language_curriculum/export_hf_catalogs.py \
  --output /data/malayalam/curriculum \
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

Preprocessing is source-aware and writes each encoded sample once under `work/preprocessed/<source>/`. Each stage writes its own checkpoints, adapter, optimizer boundary state, and metrics under `IndicFinetuning/outputs/malayalam_curriculum_1epoch/`. The final adapter is `final_adapter/`.

Two fixed Malayalam WAVs are generated at every 1,000 cumulative optimizer steps and at each stage boundary under `audio_samples/step-XXXXXX/`. They use the same held-out Rasa reference recording, texts, seed, temperature, and repetition penalty for direct listening comparisons.

Do not select the final model from loss alone. Listen to the fixed samples and compare unseen reference speakers before deciding whether to run additional epochs.

## RTX 5090 VM quickstart

The exact training datasets are:

- `ai4bharat/Rasa`, subset `Malayalam`, split `train` (gated);
- `trysem/indicvoices_r-ML`, subset `default`, split `train`.

Clone only this branch and prepare a Python 3.11 environment:

```bash
cd /workspace
git clone --branch feat/malayalam-curriculum-training --single-branch \
  https://github.com/Pranavharshans/Indic-ChatterBox.git
cd Indic-ChatterBox

sudo apt-get update
sudo apt-get install -y ffmpeg git git-lfs jq python3-venv tmux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-rtx5090.txt
python -m pip install --upgrade huggingface_hub
```

Verify the 5090 before downloading data:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); assert torch.cuda.is_available()'
```

Accept the Rasa conditions at <https://huggingface.co/datasets/ai4bharat/Rasa>, then authenticate without putting the token in shell history:

```bash
hf auth login
hf auth whoami
```

Download/export and build the deterministic plan:

```bash
export HF_HOME=/workspace/hf-cache
mkdir -p /workspace/data/malayalam-curriculum

python IndicFinetuning/single_language_curriculum/export_hf_catalogs.py \
  --output /workspace/data/malayalam-curriculum \
  --resume

python IndicFinetuning/single_language_curriculum/build_plan.py \
  --rasa-catalog /workspace/data/malayalam-curriculum/rasa/catalog.jsonl \
  --ivr-catalog /workspace/data/malayalam-curriculum/ivr/catalog.jsonl \
  --output ./IndicFinetuning/single_language_curriculum/work/plan \
  --seed 42

jq . IndicFinetuning/single_language_curriculum/work/plan/summary.json
```

Download the standard Chatterbox model and build the Malayalam tokenizer:

```bash
python setup.py --mode standard
python IndicFinetuning/tokenizer/build_indic_tokenizer.py \
  --base-tokenizer ./pretrained_models/tokenizer.json \
  --output-tokenizer ./IndicFinetuning/tokenizer/tokenizer_indic.json \
  --languages ml
```

Start the one-curriculum-epoch run in `tmux`:

```bash
tmux new -s ml-curriculum
source .venv/bin/activate
export HF_HOME=/workspace/hf-cache
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash IndicFinetuning/single_language_curriculum/train.sh 2>&1 | tee malayalam_curriculum_1epoch.log
```

Detach with `Ctrl-b`, then `d`. Reattach and inspect generated samples with:

```bash
tmux attach -t ml-curriculum
find IndicFinetuning/outputs/malayalam_curriculum_1epoch/audio_samples \
  -type f -name '*.wav' | sort
```
