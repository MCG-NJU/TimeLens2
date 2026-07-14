"""Sharded write / resume / merge helpers for off-policy rollout.

A rollout for one source is written by ``N`` data-parallel workers to
``{base}_{index}.jsonl`` and later merged into ``{base}.jsonl``. Every sample has a
stable ``sample_key`` (video_path + final query + GT span) used for resume and merge.

Resume is topology-agnostic: workers only process samples not yet complete anywhere in
the shard directory, split evenly by the *current* ``chunk``, and append to their own
``{base}_{index}.jsonl``. Merge loads every shard file and deduplicates by ``sample_key``.
"""

from __future__ import annotations

import json
import math
import numbers
import re
import sys
import time
from pathlib import Path


def _endpoint_pair_as_floats(pair):
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return None
    a, b = pair[0], pair[1]
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return [float(a), float(b)]
    return None


def span_for_sample_key(span):
    """Canonical span for the sample key: float endpoints, flat pair or list of pairs."""
    if not span:
        return []
    flat = _endpoint_pair_as_floats(span)
    if flat is not None:
        return flat
    if len(span) == 1:
        flat = _endpoint_pair_as_floats(span[0])
        if flat is not None:
            return flat
    out = []
    for x in span:
        flat = _endpoint_pair_as_floats(x)
        if flat is not None:
            out.append(flat)
    return out


def sample_key(anno: dict) -> str:
    return json.dumps(
        {
            "vp": anno["video_path"],
            "q": anno["query"],
            "s": span_for_sample_key(anno["span"]),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def collect_output_paths(pred_base: str):
    base = Path(pred_base)
    parent, name = base.parent, base.name
    shard_re = re.compile(rf"^{re.escape(name)}_(\d+)\.jsonl$")
    shard_paths = sorted(
        (p for p in parent.iterdir() if p.is_file() and shard_re.match(p.name)),
        key=lambda p: int(shard_re.match(p.name).group(1)),
    ) if parent.is_dir() else []
    merged = parent / f"{name}.jsonl"
    return shard_paths, merged


def split_evenly_across_ranks(items: list, n_ranks: int, rank: int) -> list:
    """Contiguous balanced partition: sizes differ by at most 1."""
    if n_ranks <= 0:
        raise ValueError("chunk (global worker count) must be positive")
    n = len(items)
    if n == 0:
        return []
    base, rem = divmod(n, n_ranks)
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return items[start:end]


def assign_worker_annos(
    annos_full: list,
    pred_base: str,
    *,
    chunk: int,
    index: int,
    resume: bool,
    min_rollouts: int,
) -> list:
    """Pick samples for this worker.

    Fresh run: balanced split of the full list (same as resume on all samples).
    Resume: skip globally complete keys, then balanced split of the remainder using the
    current ``chunk`` (node count may differ from the previous run).
    """
    if resume:
        done_keys = load_done_keys(pred_base, min_rollouts)
        pool = [a for a in annos_full if sample_key(a) not in done_keys]
    else:
        pool = annos_full
    return split_evenly_across_ranks(pool, chunk, index)


def _read_rows_by_key(paths) -> dict:
    rows_by_key = {}
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: skip bad JSON in {p} line {line_no}", file=sys.stderr)
                    continue
                rows_by_key[sample_key(row)] = row
    return rows_by_key


def load_all_done_rows(pred_base: str) -> dict:
    shard_paths, merged = collect_output_paths(pred_base)
    paths = list(shard_paths)
    if merged.is_file():
        paths.append(merged)
    return _read_rows_by_key(paths)


def load_done_keys(pred_base: str, min_rollouts: int) -> set:
    """Keys whose stored row already has >= ``min_rollouts`` rollouts (resume)."""
    done = set()
    for k, row in load_all_done_rows(pred_base).items():
        r = row.get("rollouts")
        if isinstance(r, list) and len(r) >= min_rollouts:
            done.add(k)
    return done


def load_done_keys_from_shard(shard_path: Path, min_rollouts: int) -> set:
    done: set[str] = set()
    if not shard_path.is_file():
        return done
    with shard_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            r = row.get("rollouts")
            if isinstance(r, list) and len(r) >= min_rollouts:
                done.add(sample_key(row))
    return done


def merge_if_complete(pred_base: str, expected_keys_in_order: list, rm_shards: bool) -> bool:
    """Write ``{base}.jsonl`` only when every expected key is present in the shards."""
    expected_set = set(expected_keys_in_order)
    rows_by_key = load_all_done_rows(pred_base)
    if len(rows_by_key) < len(expected_set) or not expected_set.issubset(rows_by_key.keys()):
        have = len(set(rows_by_key.keys()) & expected_set)
        print(f"Merge skipped: {have}/{len(expected_set)} samples present (loaded {len(rows_by_key)} rows).")
        return False

    _, merged = collect_output_paths(pred_base)
    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", encoding="utf-8") as f:
        for k in expected_keys_in_order:
            f.write(json.dumps(rows_by_key[k], ensure_ascii=False) + "\n")
    print(f"Merged {len(expected_keys_in_order)} rows -> {merged}")

    if rm_shards:
        shard_paths, _ = collect_output_paths(pred_base)
        for p in shard_paths:
            p.unlink(missing_ok=True)
        print(f"Removed {len(shard_paths)} shard file(s).")
    return True


def _format_eta_seconds(sec: float) -> str:
    if sec < 0 or not math.isfinite(sec):
        return "?"
    sec = int(round(sec))
    if sec >= 3600:
        return f"{sec // 3600}h{(sec % 3600) // 60}m"
    if sec >= 60:
        return f"{sec // 60}m{sec % 60}s"
    return f"{sec}s"


def progress_line(host, rank, cur, total, start_perf=None) -> str:
    base = f"[{host}] [rank {rank}] {cur}/{total}"
    if start_perf is None or cur <= 0 or cur >= total:
        return base
    elapsed = time.perf_counter() - start_perf
    if elapsed <= 0:
        return base
    eta = (elapsed / cur) * (total - cur)
    return f"{base} eta={_format_eta_seconds(eta)}"
