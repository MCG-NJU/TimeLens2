"""Off-policy rollout: pre-generate ``num_rollouts`` samples per training example.

Driven by the same data config as training. For each source it writes a flat jsonl
(one row per sample) with the final ``query`` baked in and a ``rollouts`` field (a
list of decoded strings). GRPO then consumes the merged jsonl and weights samples by
rollout-reward dispersion.

Multi-node / multi-GPU: launch one process per GPU with ``--index`` (global worker id)
and ``--chunk`` (global worker count). After all workers finish, run once with
``--merge`` (and optionally ``--emit_train_config``) to assemble per-source jsonl and
a ready-to-train data config.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import types
from functools import partial
from pathlib import Path

import random

import numpy as np
import torch
try:
    from nncore.engine import set_random_seed
except ModuleNotFoundError:
    def set_random_seed(seed):
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from timelens.data import RolloutDataset, collate_fn, load_data_config, load_source
from timelens.sharding import (
    assign_worker_annos,
    collect_output_paths,
    load_done_keys,
    load_done_keys_from_shard,
    merge_if_complete,
    progress_line,
    sample_key,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.set_defaults(resume=True)
    p.add_argument("--data_config", required=True, help="Data config JSON (same as training).")
    p.add_argument("--pred_root", required=True, help="Output dir; per-source base = pred_root/<name>.")
    p.add_argument("--model_path", default=None, help="Checkpoint to roll out (not needed for --merge).")
    p.add_argument("--device", default="auto")
    p.add_argument("--chunk", type=int, default=1, help="Global worker count.")
    p.add_argument("--index", type=int, default=0, help="Global worker id.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--min_tokens", type=int, default=1)
    p.add_argument("--total_tokens", type=int, default=14336)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--fps_max_frames", type=int, default=None)

    p.add_argument("--num_rollouts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--min_p", type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="Deterministic decode (do_sample=False).")

    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--fsync_each_row", action="store_true")
    p.add_argument("--progress_interval", type=int, default=1)

    p.add_argument("--merge", action="store_true", help="Merge shards into per-source jsonl, then exit.")
    p.add_argument("--rm_shards", action="store_true")
    p.add_argument("--emit_train_config", default=None, help="With --merge: write a train-ready data config here.")
    return p.parse_args()


def _video_args(args):
    return types.SimpleNamespace(
        min_tokens=args.min_tokens,
        total_tokens=args.total_tokens,
        fps=args.fps,
        fps_max_frames=args.fps_max_frames,
    )


def _generation_config(args) -> GenerationConfig:
    if args.greedy:
        return GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    if args.min_p is not None:
        kwargs["min_p"] = args.min_p
    return GenerationConfig(**kwargs)


def _sorted_annos(source):
    annos = load_source(source)
    annos.sort(key=lambda x: (x.get("duration") or 0.0), reverse=True)
    return annos


def _rollout_source(source, model, processor, gen_cfg, args):
    base = os.path.join(args.pred_root, source.name)
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    shard_path = Path(f"{base}_{args.index}.jsonl")
    host = socket.gethostname()

    annos_full = _sorted_annos(source)
    annos = assign_worker_annos(
        annos_full,
        base,
        chunk=args.chunk,
        index=args.index,
        resume=args.resume,
        min_rollouts=args.num_rollouts,
    )
    if args.resume:
        done_n = len(load_done_keys(base, args.num_rollouts))
        print(
            f"[{host}] source={source.name} | chunk={args.chunk} index={args.index} | "
            f"resume: {done_n} done, {len(annos_full) - done_n} remaining -> "
            f"this worker {len(annos)} sample(s) -> {shard_path}",
            flush=True,
        )
    else:
        print(
            f"[{host}] source={source.name} | chunk={args.chunk} index={args.index} | "
            f"this worker {len(annos)} sample(s) -> {shard_path}",
            flush=True,
        )
    if not annos:
        return

    if not args.resume:
        shard_path.unlink(missing_ok=True)

    pending_keys = {sample_key(a) for a in annos}
    if args.resume and shard_path.is_file():
        pending_keys -= load_done_keys_from_shard(shard_path, args.num_rollouts)
        annos = [a for a in annos if sample_key(a) in pending_keys]
        if not annos:
            return

    dataset = RolloutDataset(annos, _video_args(args))
    loader_workers = int(os.environ.get("TIMELENS_ROLLOUT_NUM_WORKERS", "0"))
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),
    )
    if loader_workers > 0:
        loader_kwargs["prefetch_factor"] = int(os.environ.get("TIMELENS_ROLLOUT_PREFETCH_FACTOR", "2"))
    loader = DataLoader(**loader_kwargs)

    total = len(annos)
    start_perf = time.perf_counter()
    mode = "a" if args.resume and shard_path.exists() else "w"
    with shard_path.open(mode, encoding="utf-8") as fout:
        for step, data in enumerate(loader, start=1):
            inputs = data["inputs"].to("cuda", non_blocking=True)
            anno = data["annos"][0]

            rollout_texts = []
            for ri in range(args.num_rollouts):
                set_random_seed(int(args.seed) + step * 1_000_003 + ri * 97)
                output_ids = model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    bos_token_id=processor.tokenizer.bos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    use_model_defaults=False,
                )
                gen_trimmed = output_ids[0][inputs.input_ids.shape[1] :]
                text = processor.batch_decode(
                    [gen_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                rollout_texts.append(text)
                if args.greedy:
                    break

            row = {k: v for k, v in anno.items() if k != "rollout_intervals"}
            row["rollouts"] = rollout_texts
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if args.fsync_each_row:
                os.fsync(fout.fileno())

            pi = args.progress_interval
            if pi and (step % pi == 0 or step == total):
                print(progress_line(host, args.index, step, total, start_perf), flush=True)


def _merge_all(sources, args):
    emitted = {}
    ok_all = True
    for source in sources:
        base = os.path.join(args.pred_root, source.name)
        expected = [sample_key(a) for a in _sorted_annos(source)]
        if merge_if_complete(base, expected, args.rm_shards):
            _, merged = collect_output_paths(base)
            entry = {"anno_path": str(merged), "prompt_template": None}
            if source.reward:
                entry["reward"] = source.reward
            emitted[source.name] = entry
        else:
            ok_all = False

    if args.emit_train_config and emitted:
        out = Path(args.emit_train_config)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(emitted, f, ensure_ascii=False, indent=2)
        print(f"Wrote train-ready data config -> {out}")
    return ok_all


def main():
    args = parse_args()
    sources = load_data_config(args.data_config)

    if args.merge:
        sys.exit(0 if _merge_all(sources, args) else 1)

    if args.model_path is None:
        raise SystemExit("--model_path is required unless --merge.")
    if args.device != "auto":
        raise ValueError('Only device="auto" is supported.')

    args.seed = set_random_seed(args.seed)
    print(f"Random seed (base) = {args.seed}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation=os.environ.get("TIMELENS_ATTN_IMPLEMENTATION", "flash_attention_2"),
        device_map=args.device,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path, padding_side="left", do_resize=False, trust_remote_code=True
    )
    gen_cfg = _generation_config(args)

    for source in sources:
        _rollout_source(source, model, processor, gen_cfg, args)


if __name__ == "__main__":
    main()
