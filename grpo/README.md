# 🎯 TimeLens2 Rollout and GRPO

This module contains the complete TimeLens2 rollout, reward, trainer, and data
implementation plus only the official public recipes. Historical ablation and
cluster-specific launchers are excluded because they are not imported or called
by the final pipeline. Rollout and GRPO share one JSON data-source format, and
rollout emits a `train_data_config.json` that can be passed directly to GRPO.

## 🛠️ Install

```bash
pip install -r requirements.txt
```

If your platform requires a CUDA-specific PyTorch or FlashAttention wheel,
install those two packages separately before the remaining requirements.

## 🗂️ Prepare TimeLens2-93K for rollout

```bash
export TIMELENS2_93K_ROOT=/path/to/TimeLens2-93K
python tools/prepare_timelens2_93k.py \
  --input "$TIMELENS2_93K_ROOT/TimeLens2-93K_raw_annotations.jsonl" \
  --video-root "$TIMELENS2_93K_ROOT/videos" \
  --output "$TIMELENS2_93K_ROOT/TimeLens2-93K_grpo.jsonl"
```

The two public data configs contain literal `${TIMELENS2_93K_ROOT}` and
`${TIMELENS_100K_ROOT}` placeholders. Export the corresponding variable before
launching, and make sure the annotation and video roots are visible from every
GPU worker.

## 🎲 Rollout

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
`timelens/data/sources.py`. The released data are available from
[TimeLens2-93K](https://huggingface.co/datasets/MCG-NJU/TimeLens2-93K) and
[TimeLens-100K](https://huggingface.co/datasets/MCG-NJU/TimeLens-100K).

Each rollout script generates one worker per visible GPU and finally writes a
merged JSONL plus `train_data_config.json` under `PRED_ROOT`. Use a fresh
`PRED_ROOT` for each model/data/seed combination: concurrent jobs writing the
same shard names will corrupt or overwrite one another. Do not start GRPO until
the merge succeeds and every training row has the expected 8 rollout responses.

The verified non-Slurm path is one node per rollout job (typically 8 GPUs), with
the two data sources submitted separately. Automatic multi-node shard indexing
in `scripts/rollout.sh` currently follows Slurm variables; on another scheduler,
either keep rollout single-node or provide equivalent node-rank sharding before
using a shared `PRED_ROOT`.

For a quick smoke test, use a tiny annotation subset, set `NUM_ROLLOUTS=1`, and
write to a disposable `PRED_ROOT`. This tests model/video loading without
regenerating the complete public rollout set.

## 🏋️ GRPO

The released recipe uses 8 generations, a 16K token budget, 512 frames at 2 FPS,
global batch size 64, learning rate `1e-6`, `beta=0.04`, and the
`twass1,tiou,parse_penalty` reward combination. The 4B recipe runs for 200
optimizer steps and the 8B recipe for 300, with a model-only checkpoint every 50
steps.

To match the original recipe, `scripts/train.sh` defaults to
`CUDA_LAUNCH_BLOCKING=1`. This is useful for deterministic CUDA error reporting
but may reduce kernel overlap; set `CUDA_LAUNCH_BLOCKING=0` only after validating
the training stack on your hardware.

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

Existing rollout results are reusable: `DATA_CONFIG` may point directly to a
previous complete `train_data_config.json`, as long as its media and annotation
paths are visible on all training nodes.

One training launcher process must run on every node. For non-Slurm multi-node
launchers, pass consistent `NNODES`, `NPROC_PER_NODE`, `DIST_RUN_ID`, and
`MASTER_PORT`, plus a distinct `NODE_RANK` (or `RANK`) per node. The repository
path must be shared because rendezvous and run-name files are exchanged through
`.dist_addr/` and `.deepspeed_hostfiles/`.

The default `GLOBAL_BATCH_SIZE=64` must be divisible by
`BATCH_PER_DEVICE * NNODES * NPROC_PER_NODE`. The released wrapper therefore
works unchanged with 1, 2, 4, or 8 nodes × 8 GPUs; for other world sizes, edit
the wrapper or invoke `scripts/train.sh` with compatible batch settings. The
first log lines should report the intended `world_size`, gradient accumulation,
and output directory before training starts.

Outputs are written under `outputs/grpo/timelens2-4b/` or
`outputs/grpo/timelens2-8b/`. Each saved checkpoint is a model-only checkpoint;
use the desired checkpoint directory as the local model path for final
evaluation.
