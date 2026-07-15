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

## Training data

The seven JSONL annotation splits used by the final SFT recipe are included in
`training_data_annotations/`. Download or symlink the corresponding videos and
extracted frames into this ignored local-data layout:

```text
data/
├── timelens2-93k/
│   ├── videos/
│   └── videos_2fps/
├── timelens-100k/
│   ├── videos/
│   └── videos_frames/
└── ego4d/
    └── videos/
```

> [!IMPORTANT]
> The entries in `configs/data/timelens2_sft.json` do not all use the same
> media representation. Some annotations point to original video files, while
> others point to directories of frames extracted in advance. Keep each
> `media_root` aligned with its annotation format.

| Annotation split | Media representation | `media_root` | `video_url.url` in JSONL |
| --- | --- | --- | --- |
| TimeLens2-93K short | Original videos | `data/timelens2-93k/videos` | Relative video file ending in `.mp4` |
| TimeLens2-93K long | Pre-extracted frames | `data/timelens2-93k/videos_2fps` | Relative frame-directory path |
| TimeLens-100K 0–180 s | Original videos | `data/timelens-100k/videos` | Relative video file ending in `.mp4` |
| TimeLens-100K over 180 s | Pre-extracted frames | `data/timelens-100k/videos_frames` | Relative frame-directory path |
| Ego4D-NLQ | Original videos | `data/ego4d/videos` | Relative video file ending in `.mp4` |

Video-file annotations carry fields such as `origin_video_length` and
`origin_fps`; pre-extracted-frame annotations use fields such as
`processed_video_length` and `processed_fps`. Do not redirect a frame-based
entry to a video directory, or vice versa, without updating the corresponding
JSONL records. For the standard XTuner multimodal video JSONL structure, see
[mllm_sft_video_example_data.jsonl](https://github.com/InternLM/xtuner/blob/main/tests/resource/mllm_sft_video_example_data.jsonl).

Annotation paths in `configs/data/timelens2_sft.json` are relative to this
`sft/` directory. Both standalone model configs resolve annotation and media
paths against the module root, so they do not depend on the shell's current
working directory.

## Train

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
