#!/usr/bin/env python3
"""Convert public TimeLens2-93K raw annotations to the GRPO grouped schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decord import VideoReader, cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def video_duration(path: Path) -> float:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"Invalid FPS for {path}: {fps}")
    return len(reader) / fps


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            row = json.loads(line)
            video_name = f"{row['video_id']}.mp4"
            video_path = args.video_root / video_name
            if not video_path.is_file():
                raise FileNotFoundError(f"Line {line_no}: {video_path}")
            events = [
                {"query": item["text"], "span": item["timestamps"]}
                for item in row["annotations"]
            ]
            converted = {
                "source": "timelens2_93k",
                "video_path": video_name,
                "duration": video_duration(video_path),
                "events": events,
            }
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
