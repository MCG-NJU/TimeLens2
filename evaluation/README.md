# 📊 TimeLens2 Evaluation

This directory contains the full VLMEvalKit runtime used by TimeLens2, including
its model/dataset registries, tests, workflows, documentation, and assets. The
public aliases `TimeLens2-4B` and `TimeLens2-8B` are added without replacing the
original registry. Historical experiment launchers and internal infrastructure
wrappers are excluded; the official grounding entry and the reusable frame
pre-extraction helper are retained.

## Install

```bash
pip install -e .
pip install -U flash-attn --no-build-isolation
```

## Benchmark roots

Set the roots for the benchmarks you want to run:

```bash
export TIMELENS_BENCH_ROOT=/path/to/TimeLens-Bench
export VUE_TR_ROOT=/path/to/VUE_TR
export VUE_TR_V2_ROOT=/path/to/VUE_TR_V2
export MOMENT_SEEKER_ROOT=/path/to/MomentSeeker
export EGO4D_NLQ_V2_ROOT=/path/to/Ego4D-NLQ-v2/annotations
export EGO4D_NLQ_V2_VIDEOS_DIR=/path/to/Ego4D/videos
```

The TimeLens benchmark root contains the Charades, ActivityNet, and
QVHighlights subsets expected by `vlmeval/dataset/timelens.py`.

## Run

```bash
bash scripts/srun_eval_all/run_grounding.sh
```

Run one model or a subset of datasets with space-separated overrides:

```bash
MODELS="TimeLens2-4B" \
DATASETS="VUE_TR_V2_1fps_limit_2048_px480_ctx128k" \
bash scripts/srun_eval_all/run_grounding.sh
```

Set `TIMELENS2_4B_MODEL` or `TIMELENS2_8B_MODEL` to evaluate a local checkpoint
instead of the Hugging Face model ID.

For repeated evaluation, frames can be decoded once with:

```bash
python scripts/pre_extract_video_frames/extract_video_frames.py \
  --dataset TimeLens_Charades_4fps
```
