# 📊 TimeLens2 Evaluation

We employ [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate performance on the video temporal grounding benchmark.

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

The default command evaluates **both TimeLens2-4B and TimeLens2-8B on all seven datasets**.
For a local checkpoint, replace the corresponding `model_path` in
`vlmeval/config.py`, then set the model alias, dataset, and a
checkpoint-specific output directory explicitly:

```bash
MODELS="TimeLens2-4B" \
DATASETS="TimeLens_Charades_4fps" \
N_GPU=8 \
CHECK_EXTRACTED_FRAMES=False \
OUTPUT_DIR=outputs/timelens2-4b \
bash scripts/srun_eval_all/run_grounding.sh
```

Run one model or a subset of datasets with space-separated overrides:

```bash
MODELS="TimeLens2-4B" \
DATASETS="VUE_TR_V2_1fps_limit_2048_px480_ctx128k" \
bash scripts/srun_eval_all/run_grounding.sh
```

The entry point exposes `USE_LLM_JUDGE=auto|true|false`. Its default is `auto`:
the three TimeLens benchmarks use exact parsing, while VUE-TR, VUE-TR-V2,
MomentSeeker, and Ego4D-NLQ use the configured LLM judge.

```bash
USE_LLM_JUDGE=auto \
LLM_JUDGE="qwen3-235b-a22b-thinking-2507" \
DATASETS="TimeLens_Charades_4fps Ego4D-NLQ-v2_2fps_limit_2048_px480_ctx128k" \
bash scripts/srun_eval_all/run_grounding.sh
```

Set `USE_LLM_JUDGE=true` to force the judge for every selected benchmark, or
`USE_LLM_JUDGE=false` to disable it everywhere. The
example judge is the one used in our environment; its served-model alias
defaults to `LOCAL_LLM=qwen3-235b`. For another judge service, set `LLM_JUDGE`
and `LOCAL_LLM` as required by that service.

`run.py` is called with `--reuse`, so do
not reuse the same `OUTPUT_DIR` and model alias for a different checkpoint unless
you intentionally want existing predictions to be reused.

`CHECK_EXTRACTED_FRAMES=True` validates cached frames before inference. Set it to `False` when decoding videos directly or when
the frame cache has already been validated. For repeated runs, pre-extracting
frames once is much faster:

```bash
python scripts/pre_extract_video_frames/extract_video_frames.py \
  --dataset TimeLens_Charades_4fps
```
