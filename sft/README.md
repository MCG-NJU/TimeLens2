# 🧠 TimeLens2 SFT

This directory contains the complete XTuner runtime used by TimeLens2, together
with the two official 4B/8B SFT recipes.

## 🛠️ Install

```bash
pip install -e ".[video]"
```

## 🎞️ Training data

The annotation used by the SFT recipe are included in
`training_data_annotations/`. The data config is written
in `configs/data/timelens2_sft.json`.

> Some entries use original MP4 videos, while others use pre-extracted frames
> to accelerate video loading. Both input formats are supported.

We use the standard XTuner JSONL structure, see
[mllm_sft_video_example_data.jsonl](https://github.com/InternLM/xtuner/blob/main/tests/resource/mllm_sft_video_example_data.jsonl).

Before launching, verify the following:

- Update each `media_root` in `configs/data/timelens2_sft.json` to the local
  path of the corresponding videos or extracted frames. The default config
  reads them directly from object storage; this is also supported, but requires
  the storage backends to be configured in `~/petreloss.conf`.
- Set `model_path` in the selected Python config to either a Hugging Face model
  ID or a local path to the corresponding Qwen3-VL base checkpoint.

## 🚀 Train

```bash
bash scripts/train_sft_4b.sh
# or
bash scripts/train_sft_8b.sh
```

The 4B and 8B recipes are fully defined in
`configs/qwen3_vl_4b_sft.py` and `configs/qwen3_vl_8b_sft.py`; edit the path and
hyperparameter constants at the top of the selected file when needed.

One launcher process must run on every node. Each node may use all eight GPUs,
or another value through `NPROC_PER_NODE`. For non-Slurm multi-node launchers,
pass the same `NNODES`, `NPROC_PER_NODE`, `DIST_RUN_ID`, and `MASTER_PORT` to
every node; the scheduler must provide a distinct `NODE_RANK` (or `RANK`). The
repository directory must be shared because rank 0 publishes its rendezvous
address under `.dist_addr/`.

The default global batch size is 256, so it should be divisible by the total
world size (`NNODES * NPROC_PER_NODE`). For example, 1, 2, 4, or 8 nodes with 8
GPUs per node work without changing it; otherwise adjust `global_batch_size` in
the selected config.
