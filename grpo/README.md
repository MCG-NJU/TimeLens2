# 🎯 TimeLens2 Rollout and GRPO

This module contains the complete TimeLens2 rollout, reward, trainer, and data
implementation plus only the official public recipes. Historical ablation and
cluster-specific launchers are excluded because they are not imported or called
by the final pipeline. Rollout and GRPO share one JSON data-source format, and
rollout emits a `train_data_config.json` that can be passed directly to GRPO.

## Install

```bash
pip install -r requirements.txt
```

If your platform requires a CUDA-specific PyTorch or FlashAttention wheel,
install those two packages separately before the remaining requirements.

## Prepare TimeLens2-93K for rollout

```bash
export TIMELENS2_93K_ROOT=/path/to/TimeLens2-93K
python tools/prepare_timelens2_93k.py \
  --input "$TIMELENS2_93K_ROOT/TimeLens2-93K_raw_annotations.jsonl" \
  --video-root "$TIMELENS2_93K_ROOT/videos" \
  --output "$TIMELENS2_93K_ROOT/TimeLens2-93K_grpo.jsonl"
```

## Rollout

Use the SFT checkpoint produced by `../sft`:

```bash
MODEL_PATH=/path/to/sft-4b bash scripts/rollout_timelens2_93k.sh
```

The Gemini-refined TimeLens-100K entry is separate:

```bash
export TIMELENS_100K_ROOT=/path/to/TimeLens-100K
MODEL_PATH=/path/to/sft-4b bash scripts/rollout_timelens_100k.sh
```

`TimeLens-100K_grpo.jsonl` uses the grouped schema documented in
`timelens/data/sources.py`. Its public Hugging Face ID should be added here once
the dataset is visible in the TimeLens2 collection.

## GRPO

The released recipe uses 8 generations, a 16K token budget, 512 frames at 2 FPS,
global batch size 64, learning rate `1e-6`, `beta=0.04`, and the
`twass1,tiou,parse_penalty` reward combination.

```bash
MODEL_PATH=/path/to/sft-4b \
DATA_CONFIG=outputs/rollout/timelens2_93k/train_data_config.json \
bash scripts/train_grpo_4b.sh

MODEL_PATH=/path/to/sft-8b \
DATA_CONFIG=outputs/rollout/timelens2_93k/train_data_config.json \
bash scripts/train_grpo_8b.sh
```

To train with both rollout sets, merge the two emitted JSON objects into one
data config; source names must remain unique.
