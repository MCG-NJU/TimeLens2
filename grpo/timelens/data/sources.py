"""Config-driven data sources.

A training / rollout run is described by a single JSON config (see
``configs/data/*.json``): a flat mapping of ``source_name -> source config``. Adding a
new data source means adding one entry there -- no code or shell-argument changes.
Every source is configured independently (nothing is shared across sources):

    {
      "<source_name>": {
        "anno_path": "/abs/path/to.jsonl",   // required
        "mp4_root": "/abs/mp4_videos",        // root for relative *.mp4 video_path
        "frames_root": "/abs/frame_dirs",     // root for relative frame-directory video_path
        "reward": "tiou",                      // per-source reward spec (see rewards.funcs)
        "prompt_template": null,               // per-source "...{query}..." wrapper
        "sample_fps": null,                    // default fps for frame-dir rows
        "force_frame_directory": true,         // treat relative non-mp4 paths as frame dirs
        "data_type": "grounding"
      }
    }

A ``video_path`` ending in ``.mp4`` is a raw video file (resolved under ``mp4_root``);
otherwise it is a directory of pre-extracted frames (resolved under ``frames_root``).
Absolute ``video_path`` values are used as-is and need no root.

Defaults when a field is omitted:
- ``reward``: falls back to the trainer's global ``--reward_funcs`` (default ``"tiou"``).
- ``prompt_template``: ``None`` -> the raw ``query`` is sent to the model verbatim.

Annotation jsonl lines may be either:
- grouped:  ``{"video_path", "duration", ["sample_fps"], "events": [{"query","span"}, ...]}``
- flat:     ``{"video_path", "duration", "query", "span", ["sample_fps","rollouts","span2",...]}``

The flat form is what the off-policy rollout stage emits (with a ``rollouts`` field
and the final, ready-to-use ``query``).
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from timelens.data.prompt import apply_prompt

_SOURCE_FIELDS = {
    "anno_path",
    "mp4_root",
    "frames_root",
    "reward",
    "prompt_template",
    "sample_fps",
    "force_frame_directory",
    "data_type",
}


@dataclass
class Source:
    name: str
    anno_path: str
    mp4_root: Optional[str] = None
    frames_root: Optional[str] = None
    reward: Optional[str] = None
    prompt_template: Optional[str] = None
    sample_fps: Optional[float] = None
    force_frame_directory: bool = True
    data_type: str = "grounding"

    def root_for(self, is_mp4: bool) -> Optional[str]:
        return self.mp4_root if is_mp4 else self.frames_root


def load_data_config(path: str) -> list[Source]:
    """Parse a data config JSON (flat ``name -> source config``) into ``Source`` objects."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Data config must be a non-empty JSON object: {path}")

    sources: list[Source] = []
    for name, entry in cfg.items():
        entry = dict(entry or {})
        unknown = set(entry) - _SOURCE_FIELDS
        if unknown:
            raise ValueError(f"Source {name!r} has unknown fields: {sorted(unknown)}")
        if not entry.get("anno_path"):
            raise ValueError(f"Source {name!r} is missing required field 'anno_path'.")
        for key in ("anno_path", "mp4_root", "frames_root"):
            if entry.get(key):
                entry[key] = os.path.expanduser(os.path.expandvars(entry[key]))
        sources.append(Source(name=name, **entry))
    return sources


def _iter_rows(row: dict):
    """Yield ``(query, span, extra)`` for grouped (events) or flat rows."""
    if "events" in row:
        for event in row["events"]:
            yield event["query"], event["span"], event
    elif "query" in row and "span" in row:
        yield row["query"], row["span"], row
    # otherwise: skip silently (e.g. malformed line)


def _intervals_from_rollouts(row: dict) -> list[list[list[float]]]:
    """Parse the ``rollouts`` field into a list (per rollout) of ``[[s, e], ...]``."""
    raw = row.get("rollouts")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    out: list[list[list[float]]] = []
    for item in raw:
        out.append(_one_rollout_intervals(item))
    return out


def _one_rollout_intervals(item) -> list[list[float]]:
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return []
        try:
            item = ast.literal_eval(s)
        except (SyntaxError, ValueError, MemoryError):
            return []
    if isinstance(item, (list, tuple)) and len(item) == 2 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in item
    ):
        s, e = float(item[0]), float(item[1])
        return [[s, e]] if s < e else []
    if not isinstance(item, (list, tuple)):
        return []
    out = []
    for p in item:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                s, e = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if s < e:
                out.append([s, e])
    return out


def load_source(source: Source) -> list[dict]:
    """Read one source's jsonl into canonical anno dicts."""
    path = Path(source.anno_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source {source.name!r}: anno_path not found: {path}")

    annos: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video_path_raw = str(row.get("video_path", "")).strip()
            if not video_path_raw:
                continue
            is_mp4 = video_path_raw.lower().endswith(".mp4")

            if os.path.isabs(video_path_raw):
                video_path = video_path_raw
            else:
                root = source.root_for(is_mp4)
                if not root:
                    need = "mp4_root" if is_mp4 else "frames_root"
                    raise ValueError(
                        f"Source {source.name!r} line {line_no}: relative video_path "
                        f"{video_path_raw!r} needs {need!r}."
                    )
                video_path = os.path.join(root, video_path_raw)

            for raw_query, span, extra in _iter_rows(row):
                anno = {
                    "source": row.get("source", source.name),
                    "data_type": source.data_type,
                    "video_path": video_path,
                    "duration": row.get("duration"),
                    "query": apply_prompt(raw_query, source.prompt_template),
                    "span": span,
                }
                if source.reward:
                    anno["reward"] = source.reward

                if is_mp4:
                    anno["sample_fps"] = None
                else:
                    fps = row.get("sample_fps", source.sample_fps)
                    if fps is None:
                        raise ValueError(
                            f"Source {source.name!r} line {line_no}: frame-dir video "
                            f"{video_path_raw!r} requires 'sample_fps' (row or source config)."
                        )
                    anno["sample_fps"] = float(fps)
                    anno["force_frame_directory"] = bool(
                        row.get("force_frame_directory", source.force_frame_directory)
                    )

                for key in ("span2", "video_start", "video_end"):
                    if extra.get(key) is not None:
                        anno[key] = extra[key]
                    elif row.get(key) is not None:
                        anno[key] = row[key]

                anno["rollout_intervals"] = _intervals_from_rollouts(row)
                annos.append(anno)
    return annos
