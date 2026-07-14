# TimeLens2

Config-driven **GRPO** training and **off-policy rollout** for Qwen3-VL, focused on
video temporal grounding (and general MLLM RL). Minimalist by design: one JSON data
config per run, the full prompt baked into the data, and per-source reward selection.

## Why this layout

Adding a data source used to mean threading many flags through bash. Now a run is
described by a single flat JSON file (à la xtuner) -- every source is configured
independently, nothing is shared:

```json
{
  "longvideodb": {
    "anno_path": "/abs/lvdb.jsonl",
    "mp4_root":  "/abs/videos",
    "frames_root": "/abs/frames",
    "reward": "tiou",
    "prompt_template": "When does '{query}' happen in the video?\nReturn the result strictly as a JSON array of [start, end] pairs ..."
  },
  "timelens-100k": {
    "anno_path": "/abs/timelens-100k.jsonl",
    "mp4_root": "/abs/videos",
    "reward": "tiou,format",
    "prompt_template": "Locate '{query}'. ..."
  }
}
```

Key ideas:

- **Query is the final prompt.** The `query` in a source is fed to the model verbatim.
  Either write the full instruction (task + format) into the jsonl, or set a source
  `prompt_template` containing `{query}` and it is substituted deterministically (so the
  rollout and training stages always agree on a sample's prompt). No prompt logic lives
  in the training code — swapping tasks is a data change, not a code change.
- **Per-source reward.** Each source sets its own `reward`, a comma-separated spec with
  optional weights, e.g. `"tiou"`, `"tiou,format"`, `"tiou:1.0,format:0.2"`. The global
  `--reward_funcs` is the default when a source omits it. Register new scorers in
  `timelens/rewards/funcs.py` (`INTERVAL_REWARDS`) — the trainer needs no changes.
- **Off-policy weighting.** Pre-generated `rollouts` let GRPO weight samples by
  rollout-reward dispersion (`sample_weight_std_power`), computed in one global pass
  over all sources (each sample uses its own source's reward); samples without rollouts
  fall back to uniform.

## Layout

```
timelens/
  config.py          # ModelArguments / DataArguments / GRPOArguments
  model.py           # Qwen3-VL Auto-class loader
  train.py           # GRPO training entry
  rollout.py         # off-policy rollout entry
  trainer.py         # QwenvlGRPOTrainer (TRL-based)
  train_utils.py     # peft save / param stats
  sharding.py        # shard / resume / merge for rollout
  data/
    sources.py       # data config -> canonical annotations
    dataset.py       # GroundingDataset (per source) + HybridDataset
    rollout_dataset.py
    video.py         # mp4 / frame-dir video content
    prompt.py        # verbatim query + optional template
  rewards/
    funcs.py         # reward registry + per-sample dispatch
    sampling.py      # rollout-std sample weights
    metrics.py       # IoU / GIoU / precision / timestamp parsing
configs/
  data/              # per-run data configs
  deepspeed/         # zero1.json / zero3.json
scripts/
  train.sh           # config-driven GRPO (single / multi node)
  rollout.sh         # config-driven off-policy rollout
tools/               # checkpoint ensemble, dual-anno helpers
```

## Annotation format

Each source's jsonl line is either **grouped** or **flat**:

```jsonc
// grouped (raw data): one video, many events
{"video_path": "vid/", "duration": 200, "sample_fps": 2,
 "events": [{"query": "a chapel", "span": [[0, 34]]}]}

// flat (rollout output / prepared): one training sample
{"video_path": "/abs/v.mp4", "duration": 100, "query": "<final prompt>",
 "span": [[10, 20]], "rollouts": ["[[10, 20]]", "[[0, 5]]"]}
```

`video_path` resolution: absolute paths are used as-is; a relative `*.mp4` joins with
`mp4_root`; any other relative path is a pre-extracted frame directory, joins with
`frames_root`, and requires `sample_fps` (row-level or source default).

Field defaults: `prompt_template` unset → the raw `query` is used verbatim;
`reward` unset → falls back to the trainer's global `--reward_funcs` (default `tiou`).

## Usage

Environment: `/data/videop1-shared/zhuyuhan/miniconda3/envs/qwen3/bin/python`.

### 1. Off-policy rollout

```bash
DATA_CONFIG=configs/data/example_lvdb_timelens.json \
MODEL_PATH=/path/to/sft_ckpt \
PRED_ROOT=output/rollout/qwen3vl_4b_64k_2fps_1024f \
bash scripts/rollout.sh
```

Writes `PRED_ROOT/<source>.jsonl` (with a `rollouts` field) and a train-ready
`PRED_ROOT/train_data_config.json` (each source's `anno_path` points at the merged
rollouts; paths absolute, prompt already baked).

### 2. GRPO training

```bash
DATA_CONFIG=output/rollout/qwen3vl_4b_64k_2fps_1024f/train_data_config.json \
MODEL_PATH=/path/to/sft_ckpt \
bash scripts/train.sh
```

Both scripts auto-detect single vs. multi-node Slurm (one task per node, 8 GPUs/node)
and accept overrides via env vars (see the top of each script).

## Requirements

```bash
pip install -r requirements_train.txt
```
