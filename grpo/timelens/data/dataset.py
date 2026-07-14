"""GRPO training datasets, built from a data config.

``GroundingDataset`` wraps a single source (load -> filter -> prompt building).
``HybridDataset`` concatenates several sources and computes per-sample sampling
weights in one global pass (each sample weighted by its own source's reward).

The user text fed to the model is ``anno["query"]`` verbatim -- the final prompt is
baked into the data source (or produced by its ``prompt_template``), so this dataset
is task-agnostic and contains no grounding-specific prompt logic.
"""

from __future__ import annotations

import copy
from itertools import accumulate

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.data.sources import Source, load_data_config, load_source
from timelens.data.video import build_video_content
from timelens.rewards.sampling import build_sample_weights


def normalize_spans(span):
    if isinstance(span, tuple):
        return [list(span)]
    if isinstance(span, list) and len(span) > 0 and isinstance(span[0], (list, tuple)):
        return [list(s) for s in span]
    if isinstance(span, list) and len(span) == 2 and isinstance(span[0], (int, float)):
        return [span]
    raise ValueError(f"Unsupported span format: {span}")


def _passes_filters(anno, data_args) -> bool:
    num_words = len(anno["query"].split(" "))
    if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
        return False
    if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
        return False
    duration = anno.get("duration")
    if data_args.min_video_len >= 0 and (duration or float("inf")) < data_args.min_video_len:
        return False
    if data_args.max_video_len >= 0 and (duration or 0) > data_args.max_video_len:
        return False
    return True


class GroundingDataset(Dataset):
    def __init__(self, processor, model_args, data_args, training_args, source: Source):
        super().__init__()
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.source = source

        annos = []
        for anno in load_source(source):
            if not _passes_filters(anno, data_args):
                continue
            duration = anno.get("duration")
            spans = normalize_spans(anno["span"])
            if duration and not any(0 <= s <= e <= duration for s, e in spans):
                continue
            anno = dict(anno)
            anno["span"] = spans
            s2 = anno.get("span2")
            if s2:
                s2n = normalize_spans(s2)
                if duration and not any(0 <= s <= e <= duration for s, e in s2n):
                    continue
                anno["span2"] = s2n
            else:
                anno.pop("span2", None)
            annos.append(anno)

        # ``rollout_intervals`` is kept here and consumed once, globally, by
        # HybridDataset to build sampling weights (then dropped).
        self.annos = annos

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])
        messages = [
            {
                "role": "user",
                "content": [
                    build_video_content(anno, self.data_args, include_video_range=True),
                    {"type": "text", "text": anno["query"]},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = [text]

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["prompt"] = messages
        inputs["prompt_text"] = text[0]
        inputs["anno"] = anno
        return inputs


class HybridDataset(Dataset):
    """Concatenate several sources; sample weights come from a single global pass over
    all samples (each sample weighted by its own source's reward-based rollout std)."""

    def __init__(self, processor, model_config, model_args, data_args, training_args):
        super().__init__()
        if not getattr(data_args, "data_config", None):
            raise ValueError("data_args.data_config (path to a data config JSON) is required.")

        sources = load_data_config(data_args.data_config)
        datasets = [
            GroundingDataset(processor, model_args, data_args, training_args, src)
            for src in sources
        ]
        non_empty = [(src, d) for src, d in zip(sources, datasets) if len(d) > 0]
        if not non_empty:
            raise ValueError("All data sources are empty after filtering.")
        sources, datasets = [s for s, _ in non_empty], [d for _, d in non_empty]

        cum = [0] + list(accumulate(len(d) for d in datasets))
        self.idx_ranges = [(cum[i], cum[i + 1]) for i in range(len(datasets))]
        self.datasets = datasets

        # Global weighting: each anno carries its source's ``reward`` (used inside
        # build_sample_weights), so per-source rewards are respected automatically.
        default_spec = str(getattr(training_args, "reward_funcs", "") or "tiou")
        std_power = float(getattr(data_args, "sample_weight_std_power", 2.0))
        mean_power = float(getattr(data_args, "sample_weight_mean_power", 0.0))
        mean_spec = getattr(data_args, "sample_weight_mean_spec", None) or None
        all_annos = [a for d in datasets for a in d.annos]
        self.sample_weights = build_sample_weights(
            all_annos,
            default_spec,
            std_power=std_power,
            mean_power=mean_power,
            mean_spec=mean_spec,
        )
        for a in all_annos:
            a.pop("rollout_intervals", None)

        for src, d in zip(sources, datasets):
            print(f"[data] source={src.name} size={len(d)}")

    def __len__(self):
        return self.idx_ranges[-1][1]

    def __getitem__(self, idx):
        for (start, end), dataset in zip(self.idx_ranges, self.datasets):
            if start <= idx < end:
                return dataset[idx - start]
        raise IndexError(f"Index out of range: {idx}")
