#!/usr/bin/env python3
"""Convert the public conversations JSONL to XTuner's multimodal JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-suffix", default=".mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            row = json.loads(line)
            messages = row["messages"]
            if not messages or messages[0].get("role") != "user":
                raise ValueError(f"Row {index} does not start with a user message")
            first = dict(messages[0])
            text = first.get("content", "")
            if not isinstance(text, str):
                raise ValueError(f"Row {index} already has non-text first-message content")
            video_name = f"{row['video_id']}{args.video_suffix}"
            first["content"] = [
                {"type": "video_url", "video_url": {"url": video_name}},
                {"type": "text", "text": text},
            ]
            converted = {"id": index, "messages": [first, *messages[1:]]}
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
