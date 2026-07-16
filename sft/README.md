# 🧠 TimeLens2 SFT

This directory contains the complete XTuner runtime used by TimeLens2, together
with the two official 4B/8B SFT recipes. Framework code, acceleration paths,
tests, CI, documentation, and dependency declarations are retained; historical
experiment and ablation recipes are intentionally excluded from the public
release. The published recipes preserve the final optimization settings.

## 🛠️ Install

```bash
pip install -e ".[video]"
```

## 🎞️ Training data

The seven JSONL annotation splits used by the final SFT recipe are included in
`training_data_annotations/`. The official object-storage URIs are written
directly in `configs/data/timelens2_sft.json`, matching the original training
repo. Petrel reads them using `~/petreloss.conf`.

> [!IMPORTANT]
> The entries in `configs/data/timelens2_sft.json` do not all use the same
> media representation. Some annotations point to original video files, while
> others point to directories of frames extracted in advance. Keep each
> `media_root` aligned with its annotation format.

| Annotation split | Media representation | `media_root` | `video_url.url` in JSONL |
| --- | --- | --- | --- |
| TimeLens2-93K short | Original videos | `pnorm2:s3://longvideodb-373k-videos-decompress/videos/` | Relative video file ending in `.mp4` |
| TimeLens2-93K long | Pre-extracted frames | `videogpu2:s3://lvdb-2fps-10min/videos_2fps/` | Relative frame-directory path |
| TimeLens-100K 0–180 s | Original videos | `pnorm2:s3://videochat3/TimeLens-100K/videos/` | Relative video file ending in `.mp4` |
| TimeLens-100K over 180 s | Pre-extracted frames | `pnorm2:s3://videochat3/TimeLens-100K/videos_frames/` | Relative frame-directory path |
| Ego4D-NLQ | Original videos | `pnorm2:s3://ego4d_v2/320p/` | Relative video file ending in `.mp4` |

Video-file annotations carry fields such as `origin_video_length` and
`origin_fps`; pre-extracted-frame annotations use fields such as
`processed_video_length` and `processed_fps`. Do not redirect a frame-based
entry to a video directory, or vice versa, without updating the corresponding
JSONL records. For the standard XTuner multimodal video JSONL structure, see
[mllm_sft_video_example_data.jsonl](https://github.com/InternLM/xtuner/blob/main/tests/resource/mllm_sft_video_example_data.jsonl).

Annotation paths in `configs/data/timelens2_sft.json` are relative to this
`sft/` directory. Both standalone model configs resolve these annotation paths
against the module root, so they do not depend on the shell's current working
directory. If using local media instead of the official object-store locations,
replace only the corresponding `media_root` values.

Before launching, verify the following:

- `~/petreloss.conf` defines the storage backends used by every `media_root`
  (`pnorm2` and `videogpu2` in the released config). This file is not included.
- Every annotation file under `training_data_annotations/` exists and its media
  representation matches the table above. Checking one sample from each split
  catches most path mistakes before tokenization and packing begin.
- `model_path` at the top of the selected Python config points to a complete
  Qwen3-VL base checkpoint. The launch wrappers do **not** read `MODEL_PATH`.

## 🚀 Train

```bash
bash scripts/train_sft_4b.sh
# or
bash scripts/train_sft_8b.sh
```

The main defaults are 102,400-token packing, global batch size 256, one epoch,
learning rate `5e-6`, frozen vision encoder, FSDP full recomputation, and square
loss reduction. The 4B and 8B recipes are fully defined in
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

## 📦 Outputs and resume behavior

- 4B outputs and per-node logs are written under `outputs/sft-4b/`; 8B uses
  `outputs/sft-8b/`.
- The trainer has `auto_resume=True`. Reusing a populated output directory
  resumes that run; use a new `work_dir` and `cache_dir` for a genuinely new
  experiment.
- Hugging Face exports are named `hf-*`. Use a complete export directory—not an
  intermediate distributed checkpoint—as `MODEL_PATH` for rollout/GRPO or as
  `TIMELENS2_4B_MODEL` / `TIMELENS2_8B_MODEL` for evaluation.
