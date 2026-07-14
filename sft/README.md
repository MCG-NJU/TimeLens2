# 🧠 TimeLens2 SFT

This directory contains the complete XTuner runtime used by TimeLens2, together
with the two official 4B/8B SFT recipes. Framework code, acceleration paths,
tests, CI, documentation, and dependency declarations are retained; historical
experiment and ablation recipes are intentionally excluded from the public
release. The published recipes preserve the final optimization settings.

## Install

```bash
pip install -e ".[video]"
```

## Prepare TimeLens2-93K

Download and extract `MCG-NJU/TimeLens2-93K`, then convert its public
conversation file:

```bash
python tools/prepare_timelens2_93k.py \
  --input "$TIMELENS2_93K_ROOT/TimeLens2-93K_conversations.jsonl" \
  --output "$TIMELENS2_93K_ROOT/TimeLens2-93K_xtuner.jsonl"
```

The released SFT mixture also includes TimeLens-100K and Ego4D-NLQ. Set
`TIMELENS_100K_ROOT`, `EGO4D_NLQ_ROOT`, and `EGO4D_VIDEO_ROOT` to their prepared
XTuner-format copies. Edit `configs/data/timelens2_sft.json` if training on a
subset of the released mixture.

## Train

```bash
export TIMELENS2_93K_ROOT=/path/to/TimeLens2-93K
export TIMELENS_100K_ROOT=/path/to/TimeLens-100K
export EGO4D_NLQ_ROOT=/path/to/Ego4D-NLQ/annotations
export EGO4D_VIDEO_ROOT=/path/to/Ego4D/videos

bash scripts/train_sft_4b.sh
# or
bash scripts/train_sft_8b.sh
```

The main defaults are 102,400-token packing, global batch size 256, one epoch,
learning rate `5e-6`, frozen vision encoder, FSDP full recomputation, and square
loss reduction. Override paths and scale settings with the environment variables
used in `configs/_base.py` and `scripts/train.sh`.

Public Hugging Face data uses local media paths. To reproduce the original
Petrel-backed input path, set `PETREL_CONF_PATH` to the corresponding client
configuration before launch; the model, optimizer, packing, and FSDP paths are
otherwise shared.
