"""Build the ``{"type": "video", ...}`` content dict for qwen_vl_utils.

Handles two cases:
- a raw ``.mp4`` file: pass the path with fps / pixel budget;
- a directory of pre-extracted frames (``sample_fps`` given): subsample to the
  target fps, optionally clip to ``[video_start, video_end]`` and cap the number
  of frames at ``fps_max_frames``.

The numeric budget knobs come from a ``data_args``-like object with attributes
``min_tokens``, ``total_tokens``, ``fps`` and ``fps_max_frames``.
"""

from __future__ import annotations

import math
from pathlib import Path

_FRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def _path_to_file_uri(path: Path | str) -> str:
    return Path(path).expanduser().resolve().as_uri()


def _list_sorted_frame_paths(video_dir: Path) -> list[Path]:
    paths = [
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _FRAME_SUFFIXES
    ]
    paths.sort(key=lambda p: p.name)
    return paths


def _frame_range_from_time(total_frames, video_fps, video_start, video_end):
    """Match qwen_vl_utils.calculate_video_frame_range (inclusive indices)."""
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if video_start is None and video_end is None:
        return 0, total_frames - 1

    max_duration = total_frames / video_fps
    if video_start is not None:
        start_frame = math.ceil(max(0.0, min(float(video_start), max_duration)) * video_fps)
    else:
        start_frame = 0
    if video_end is not None:
        end_frame = math.floor(max(0.0, min(float(video_end), max_duration)) * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range for frame directory: start_frame={start_frame}, "
            f"end_frame={end_frame}, total_frames={total_frames}, video_fps={video_fps}, "
            f"video_start={video_start}, video_end={video_end}"
        )
    return start_frame, end_frame


def _linspace_indices(n: int, k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def _effective_sample_fps(picked, sample_fps, fallback_fps):
    if len(picked) <= 1:
        return float(fallback_fps)
    span = (picked[-1] - picked[0]) / sample_fps
    if span <= 0:
        return float(fallback_fps)
    return (len(picked) - 1) / span


def _build_frame_dir_content(anno, data_args, dir_path: Path, include_video_range: bool):
    sample_fps = float(anno["sample_fps"])
    target_fps = float(data_args.fps)
    if sample_fps <= 0 or target_fps <= 0:
        raise ValueError("sample_fps and data_args.fps must be positive")
    ratio = sample_fps / target_fps
    step = int(round(ratio))
    if not math.isclose(ratio, float(step), rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(
            f"anno['sample_fps'] ({sample_fps}) must be an integer multiple of "
            f"data_args.fps ({target_fps}), got ratio {ratio}."
        )
    if step < 1:
        raise ValueError(f"Invalid subsample step {step} from sample_fps={sample_fps}, fps={target_fps}")

    frame_paths = _list_sorted_frame_paths(dir_path)
    n_frames = len(frame_paths)
    if n_frames == 0:
        raise ValueError(f"No image frames found under directory: {dir_path}")

    if include_video_range and (
        anno.get("video_start") is not None or anno.get("video_end") is not None
    ):
        start_frame, end_frame = _frame_range_from_time(
            n_frames, sample_fps, anno.get("video_start"), anno.get("video_end")
        )
    else:
        start_frame, end_frame = 0, n_frames - 1

    picked = [j for j in range(0, n_frames, step) if start_frame <= j <= end_frame]
    if not picked:
        raise ValueError(
            f"No frames left after subsampling (step={step}) and range "
            f"[{start_frame}, {end_frame}] for {dir_path}"
        )

    fps_max = getattr(data_args, "fps_max_frames", None)
    if fps_max is not None:
        fps_max = int(fps_max)
        if fps_max < 1:
            raise ValueError("fps_max_frames must be >= 1 when set")
        if len(picked) > fps_max:
            picked = [picked[i] for i in _linspace_indices(len(picked), fps_max)]

    return {
        "type": "video",
        "video": [_path_to_file_uri(frame_paths[i]) for i in picked],
        "sample_fps": float(_effective_sample_fps(picked, sample_fps, target_fps)),
        "min_pixels": int(data_args.min_tokens * 32 * 32),
        "total_pixels": int(data_args.total_tokens * 32 * 32),
    }


def build_video_content(anno, data_args, include_video_range: bool = False):
    """Return a qwen_vl_utils video content dict for ``anno['video_path']``."""
    path_obj = Path(anno["video_path"]).expanduser()
    use_frame_dir = anno.get("sample_fps") is not None and (
        path_obj.is_dir() or anno.get("force_frame_directory", False)
    )
    if use_frame_dir:
        if not path_obj.is_dir():
            raise FileNotFoundError(
                f"Expected pre-extracted frame directory for video_path={path_obj} "
                f"(sample_fps={anno.get('sample_fps')}, force_frame_directory=True)."
            )
        return _build_frame_dir_content(anno, data_args, path_obj, include_video_range)

    content = {
        "type": "video",
        "video": anno["video_path"],
        "min_pixels": int(data_args.min_tokens * 32 * 32),
        "total_pixels": int(data_args.total_tokens * 32 * 32),
        "fps": float(data_args.fps),
    }
    if include_video_range:
        vs, ve = anno.get("video_start"), anno.get("video_end")
        if vs is not None and ve is not None:
            content["video_start"] = float(vs)
            content["video_end"] = float(ve)
    if getattr(data_args, "fps_max_frames", None) is not None:
        content["max_frames"] = int(data_args.fps_max_frames)
    return content
