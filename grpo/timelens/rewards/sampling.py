"""Off-policy GRPO sample weights from rollout reward dispersion and mean.

For each sample we score its pre-computed ``rollout_intervals`` against the GT under
the sample's reward spec, take the population std/mean across rollouts, and set the
sampling weight to ``(std + eps) ** std_power * (1 - mean + eps) ** mean_power``
(then normalised). Larger ``std_power`` pulls more mass toward high-disagreement
samples; larger ``mean_power`` favours lower mean reward (harder samples). Samples
without rollouts get a uniform fallback weight.
"""

from __future__ import annotations

import numpy as np
import torch

from timelens.rewards.funcs import score_intervals


def rollout_mean(anno: dict, default_spec: str, mean_spec: str | None = None) -> float:
    intervals_list = anno.get("rollout_intervals") or []
    if len(intervals_list) < 2:
        return 0.0
    rewards = [
        score_intervals(
            ivs,
            anno,
            default_spec,
            spec_override=mean_spec if mean_spec is not None else None,
        )
        for ivs in intervals_list
    ]
    return float(np.mean(np.asarray(rewards, dtype=np.float64)))


def rollout_reward_stats(
    anno: dict,
    default_spec: str,
    mean_spec: str | None = None,
) -> tuple[float, float]:
    intervals_list = anno.get("rollout_intervals") or []
    if len(intervals_list) < 2:
        return 0.0, 0.0
    rewards = [score_intervals(ivs, anno, default_spec) for ivs in intervals_list]
    arr = np.asarray(rewards, dtype=np.float64)
    if mean_spec is not None:
        mean_rewards = [
            score_intervals(ivs, anno, default_spec, spec_override=mean_spec)
            for ivs in intervals_list
        ]
        mean = float(np.mean(np.asarray(mean_rewards, dtype=np.float64)))
    else:
        mean = float(np.mean(arr))
    return float(np.std(arr, ddof=0)), mean


def rollout_reward_std(anno: dict, default_spec: str) -> float:
    return rollout_reward_stats(anno, default_spec)[0]


def build_sample_weights(
    annos: list[dict],
    default_spec: str,
    eps: float = 1e-8,
    std_power: float = 2.0,
    mean_power: float = 0.0,
    mean_spec: str | None = None,
) -> torch.Tensor:
    if not annos:
        return torch.tensor([], dtype=torch.float32)
    if std_power != 0.0:
        stats = [rollout_reward_stats(a, default_spec, mean_spec=mean_spec) for a in annos]
        stds = np.maximum(np.array([s for s, _ in stats], dtype=np.float64), 0.0)
        means = np.clip(np.array([m for _, m in stats], dtype=np.float64), 0.0, 1.0)
    else:
        stds = np.zeros(len(annos), dtype=np.float64)
        means = np.clip(
            np.array(
                [rollout_mean(a, default_spec, mean_spec=mean_spec) for a in annos],
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )
    base = np.power(stds + eps, float(std_power))
    if mean_power != 0.0:
        base *= np.power(1.0 - means + eps, float(mean_power))
    w = torch.tensor(base, dtype=torch.float32)
    return w / w.sum()
