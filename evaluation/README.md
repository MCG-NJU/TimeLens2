# 📊 TimeLens2 Evaluation

This directory contains the full VLMEvalKit runtime used by TimeLens2, including
its model/dataset registries, tests, workflows, documentation, and assets. The
public aliases `TimeLens2-4B` and `TimeLens2-8B` are added without replacing the
original registry. Historical experiment launchers and internal infrastructure
wrappers are excluded; the official grounding entry and the reusable frame
pre-extraction helper are retained.

## 🛠️ Install

```bash
pip install -e .
pip install -U flash-attn --no-build-isolation
```

## 🎞️ Benchmark roots

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

`LMUData` must point to a writable location. VLMEvalKit stores metadata, cached
frames, and intermediate files there, so a shared path is recommended when jobs
may be resumed or inspected from another machine.

## 🚀 Run

```bash
bash scripts/srun_eval_all/run_grounding.sh
```

The default command evaluates **both released models on all seven datasets**.
For a local checkpoint, always set the model alias, checkpoint path, dataset,
and a checkpoint-specific work directory explicitly:

```bash
TIMELENS2_4B_MODEL=/path/to/hf-99 \
MODELS="TimeLens2-4B" \
DATASETS="TimeLens_Charades_4fps" \
NPROC_PER_NODE=8 \
CHECK_EXTRACTED_FRAMES=False \
WORK_DIR=outputs/sft4b-hf99 \
bash scripts/srun_eval_all/run_grounding.sh
```

Run one model or a subset of datasets with space-separated overrides:

```bash
MODELS="TimeLens2-4B" \
DATASETS="VUE_TR_V2_1fps_limit_2048_px480_ctx128k" \
bash scripts/srun_eval_all/run_grounding.sh
```

If a benchmark uses an external judge, pass the registered judge name with
`JUDGE`, for example:

```bash
JUDGE="qwen3-235b-a22b-thinking-2507" \
DATASETS="VUE_TR_V2_1fps_limit_2048_px480_ctx128k" \
bash scripts/srun_eval_all/run_grounding.sh
```

The official evaluation uses no external judge for the three TimeLens subsets.
It uses a configured judge for VUE-TR, VUE-TR-V2, MomentSeeker, and Ego4D-NLQ.
The example judge name is the one used in our environment; replace it with a
judge registered and reachable in yours. Inference can finish without that API,
but judge-dependent scoring cannot.

Set `TIMELENS2_4B_MODEL` or `TIMELENS2_8B_MODEL` to evaluate a local checkpoint
instead of the Hugging Face model ID.

Run each benchmark as a separate 8-GPU job. This keeps failures isolated and
matches the released evaluation setup. `run.py` is called with `--reuse`, so do
not reuse the same `WORK_DIR` and model alias for a different checkpoint unless
you intentionally want existing predictions to be reused; a fresh work
directory per checkpoint avoids stale results.

`CHECK_EXTRACTED_FRAMES=True` validates cached frames before inference and can
add a long startup scan. Set it to `False` when decoding videos directly or when
the frame cache has already been validated. For repeated runs, pre-extracting
frames once is usually faster:

```bash
python scripts/pre_extract_video_frames/extract_video_frames.py \
  --dataset TimeLens_Charades_4fps
```
