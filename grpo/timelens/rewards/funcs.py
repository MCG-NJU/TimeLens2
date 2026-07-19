"""Reward registry and per-sample reward dispatch.

A reward *spec* is a comma-separated list of reward names, each with an optional
``:weight`` (default ``1.0``), e.g. ``"tiou"``, ``"tiou,format"``,
``"tiou:1.0,format:0.2"`` or ``"tiou,parse:0.2"``. ``twass`` accepts an optional
numeric suffix for ``beta`` in ``R = exp(-beta * W1 / |merge(target)|)``, e.g. ``twass1`` (beta=1),
``twass3`` (beta=3); bare ``twass`` uses beta=3. ``tcgauss`` also accepts
``a<alpha>`` (e.g. ``tcgauss1a0.25``) and ``tbgauss`` accepts ``s<sigma_ratio>``
(e.g. ``tbgauss1s0.08``). The spec used for a sample is
``anno["reward"]`` if set (from its data-source config), otherwise the global
default passed in. ``r2twass`` follows the W2 quantile formula with the same beta suffix.

Interval rewards score parsed prediction intervals against ``anno["span"]``; the
``ngiou`` reward is MUSEG's sequential local matching reward: matched segments
use normalized temporal GIoU and unmatched segments contribute exactly zero. The
special ``format`` reward scores CoT tag structure; ``parse`` rewards successful
JSON interval extraction; ``parse_penalty`` returns 0 on success and -1 on failure.
``length_ratio_penalty`` returns ``-abs(log10((pred_len + eps) / (gt_len + eps)))``
to discourage overly short or overly long predicted intervals.
New tasks can register
their own scorers in ``INTERVAL_REWARDS`` (or extend the dispatch) without touching
the trainer.
"""

from __future__ import annotations

import math
import re

from timelens.rewards.metrics import (
    COT_FORMAT_PATTERN,
    extract_answer,
    extract_time,
    giou,
    giou_span,
    iou,
    span_pairwise_intersection,
    temporal_precision,
    temporal_wasserstein_reward,
    temporal_endpoint_dirac_wasserstein_reward,
    temporal_center_gaussian_wasserstein_reward,
    temporal_boundary_gaussian_wasserstein_reward,
    temporal_wasserstein2_reward,
)

# R = IoU**alpha * Precision**(1-alpha); alpha in [0.5, 0.7] favours IoU.
PRECISION_WEIGHTED_IOU_ALPHA = 0.6
LENGTH_RATIO_EPS = 1e-6

_FORMAT_PATTERN = COT_FORMAT_PATTERN
_VISION_TOKEN_PATTERN = re.compile(r"<\|(video_pad|image_pad|vision_start|vision_end)\|>")
_TWASS_NAME_RE = re.compile(r"^twass(\d+(?:\.\d+)?)?$")
_R2TWASS_NAME_RE = re.compile(r"^r2twass(\d+(?:\.\d+)?)?$")
_TDIRAC_NAME_RE = re.compile(r"^tdirac(\d+(?:\.\d+)?)?$")
_TCGAUSS_NAME_RE = re.compile(r"^tcgauss(?P<beta>\d+(?:\.\d+)?)?(?:a(?P<alpha>\d+(?:\.\d+)?))?$")
_TBGAUSS_NAME_RE = re.compile(r"^tbgauss(?P<beta>\d+(?:\.\d+)?)?(?:s(?P<sigma_ratio>\d+(?:\.\d+)?))?$")


def _pw_iou(pred_ivs, gt_iou, gt_prec):
    alpha = max(0.0, min(1.0, float(PRECISION_WEIGHTED_IOU_ALPHA)))
    iou_v = iou(pred_ivs, gt_iou)
    prec_v = temporal_precision(pred_ivs, gt_prec)
    return float((iou_v ** alpha) * (prec_v ** (1.0 - alpha)))


def _r_tiou(pred_ivs, anno):
    return float(iou(pred_ivs, anno["span"]))


def _r_tgiou(pred_ivs, anno):
    return float(giou(pred_ivs, anno["span"]))


def _r_ngiou(pred_ivs, anno):
    """MUSEG local matching: mean normalized GIoU after start-time sorting."""

    def _sorted_valid_intervals(intervals):
        if not intervals:
            return []
        if (
            isinstance(intervals, (list, tuple))
            and len(intervals) >= 2
            and not isinstance(intervals[0], (list, tuple))
        ):
            intervals = [intervals]

        valid = []
        for span in intervals:
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                continue
            try:
                start, end = float(span[0]), float(span[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and start < end:
                valid.append([start, end])
        return sorted(valid, key=lambda span: (span[0], span[1]))

    pred = _sorted_valid_intervals(pred_ivs)
    gt = _sorted_valid_intervals(anno["span"])
    pair_count = max(len(pred), len(gt))
    if pair_count == 0:
        return 0.0

    matched_score = sum(
        0.5 * (1.0 + giou_span(pred_span, gt_span))
        for pred_span, gt_span in zip(pred, gt)
    )
    return float(matched_score / pair_count)


def _r_twass(pred_ivs, anno, beta: float = 3.0):
    return float(temporal_wasserstein_reward(pred_ivs, anno["span"], beta=beta))


def _r_r2twass(pred_ivs, anno, beta: float = 3.0):
    return float(temporal_wasserstein2_reward(pred_ivs, anno["span"], beta=beta))


def _r_tdirac(pred_ivs, anno, beta: float = 3.0):
    return float(temporal_endpoint_dirac_wasserstein_reward(pred_ivs, anno["span"], beta=beta))


def _r_tcgauss(pred_ivs, anno, beta: float = 3.0, alpha: float = 0.5):
    return float(
        temporal_center_gaussian_wasserstein_reward(
            pred_ivs, anno["span"], beta=beta, alpha=alpha
        )
    )


def _r_tbgauss(pred_ivs, anno, beta: float = 3.0, sigma_ratio: float = 0.08):
    return float(
        temporal_boundary_gaussian_wasserstein_reward(
            pred_ivs, anno["span"], beta=beta, sigma_ratio=sigma_ratio
        )
    )


def _interval_total_length(intervals) -> float:
    if not intervals:
        return 0.0

    def _span_len(span) -> float:
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            return 0.0
        try:
            s, e = float(span[0]), float(span[1])
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, e - s)

    if (
        isinstance(intervals, (list, tuple))
        and len(intervals) >= 2
        and not isinstance(intervals[0], (list, tuple))
    ):
        return _span_len(intervals)

    return float(sum(_span_len(span) for span in intervals))


def _r_length_ratio_penalty(pred_ivs, anno):
    pred_len = _interval_total_length(pred_ivs)
    gt_len = _interval_total_length(anno["span"])
    ratio = (pred_len + LENGTH_RATIO_EPS) / (gt_len + LENGTH_RATIO_EPS)
    return -abs(math.log10(ratio))


def _resolve_reward_name(name: str) -> tuple[str, dict]:
    """Map a spec token to a canonical reward name and optional scorer kwargs."""
    m = _TWASS_NAME_RE.match(name)
    if m:
        beta_str = m.group(1)
        beta = float(beta_str) if beta_str is not None else 3.0
        return "twass", {"beta": beta}
    m = _R2TWASS_NAME_RE.match(name)
    if m:
        beta_str = m.group(1)
        beta = float(beta_str) if beta_str is not None else 3.0
        return "r2twass", {"beta": beta}
    m = _TDIRAC_NAME_RE.match(name)
    if m:
        beta_str = m.group(1)
        beta = float(beta_str) if beta_str is not None else 3.0
        return "tdirac", {"beta": beta}
    m = _TCGAUSS_NAME_RE.match(name)
    if m:
        beta_str = m.group("beta")
        alpha_str = m.group("alpha")
        beta = float(beta_str) if beta_str is not None else 3.0
        alpha = float(alpha_str) if alpha_str is not None else 0.5
        return "tcgauss", {"beta": beta, "alpha": alpha}
    m = _TBGAUSS_NAME_RE.match(name)
    if m:
        beta_str = m.group("beta")
        sigma_ratio_str = m.group("sigma_ratio")
        beta = float(beta_str) if beta_str is not None else 3.0
        sigma_ratio = float(sigma_ratio_str) if sigma_ratio_str is not None else 0.08
        return "tbgauss", {"beta": beta, "sigma_ratio": sigma_ratio}
    return name, {}


def _interval_reward_score(name: str, pred_ivs, anno, reward_kwargs: dict | None = None) -> float:
    reward_kwargs = reward_kwargs or {}
    if name == "twass":
        return _r_twass(pred_ivs, anno, beta=reward_kwargs.get("beta", 3.0))
    if name == "r2twass":
        return _r_r2twass(pred_ivs, anno, beta=reward_kwargs.get("beta", 3.0))
    if name == "tdirac":
        return _r_tdirac(pred_ivs, anno, beta=reward_kwargs.get("beta", 3.0))
    if name == "tcgauss":
        return _r_tcgauss(
            pred_ivs,
            anno,
            beta=reward_kwargs.get("beta", 3.0),
            alpha=reward_kwargs.get("alpha", 0.5),
        )
    if name == "tbgauss":
        return _r_tbgauss(
            pred_ivs,
            anno,
            beta=reward_kwargs.get("beta", 3.0),
            sigma_ratio=reward_kwargs.get("sigma_ratio", 0.08),
        )
    return INTERVAL_REWARDS[name](pred_ivs, anno)


def _r_precision_weight_iou(pred_ivs, anno):
    return _pw_iou(pred_ivs, anno["span"], anno["span"])


def _r_dual_anno_precision_weight_iou(pred_ivs, anno):
    gt_iou = anno["span"]
    s2 = anno.get("span2")
    gt_prec = span_pairwise_intersection(gt_iou, s2) if s2 else gt_iou
    return _pw_iou(pred_ivs, gt_iou, gt_prec)


# name -> scorer(pred_intervals, anno) -> float
INTERVAL_REWARDS = {
    "tiou": _r_tiou,
    "tgiou": _r_tgiou,
    "ngiou": _r_ngiou,
    "twass": _r_twass,
    "r2twass": _r_r2twass,
    "tdirac": _r_tdirac,
    "tcgauss": _r_tcgauss,
    "tbgauss": _r_tbgauss,
    "length_ratio_penalty": _r_length_ratio_penalty,
    "precision_weight_iou": _r_precision_weight_iou,
    "dual_anno_precision_weight_iou": _r_dual_anno_precision_weight_iou,
}


def format_score(text: str) -> float:
    return 1.0 if _FORMAT_PATTERN.fullmatch(text or "") else 0.0


def parse_score(text: str) -> float:
    """1.0 if the completion parses into valid ``[[start, end], ...]`` intervals."""
    pred_ivs, _ = _intervals_from_text(text)
    return 1.0 if pred_ivs else 0.0


def parse_penalty_score(text: str) -> float:
    """0.0 if valid ``[[start, end], ...]`` intervals; -1.0 if parsing fails."""
    pred_ivs, _ = _intervals_from_text(text)
    return 0.0 if pred_ivs else -1.0


def parse_spec(spec: str) -> list[tuple[str, float, dict]]:
    """Parse ``"name[:weight],..."`` into ``[(name, weight, kwargs), ...]``."""
    items: list[tuple[str, float, dict]] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            raw_name, w = token.split(":", 1)
            weight = float(w)
        else:
            raw_name, weight = token, 1.0
        raw_name = raw_name.strip()
        name, reward_kwargs = _resolve_reward_name(raw_name)
        if name not in ("format", "parse", "parse_penalty") and name not in INTERVAL_REWARDS:
            raise ValueError(
                f"Unknown reward {raw_name!r}; known: {sorted(INTERVAL_REWARDS)} + "
                f"'format' + 'parse' + 'parse_penalty' + suffixes like "
                f"'twass1'/'r2twass1'/'tdirac1'/'tcgauss1a0.25'/'tbgauss1s0.08'."
            )
        items.append((name, weight, reward_kwargs))
    if not items:
        raise ValueError(f"Empty reward spec: {spec!r}")
    return items


def _intervals_from_text(text: str):
    answer = extract_answer(text)
    ivs = [[float(s), float(e)] for s, e in extract_time(answer)]
    valid = bool(ivs) and all(s < e for s, e in ivs)
    return (ivs if valid else []), answer


def score_completion(text: str, anno: dict, default_spec: str) -> float:
    """Total reward for one model completion under the sample's reward spec."""
    spec = anno.get("reward") or default_spec
    items = parse_spec(spec)
    pred_ivs, _ = _intervals_from_text(text)
    total = 0.0
    for name, weight, reward_kwargs in items:
        if name == "format":
            total += weight * format_score(text)
        elif name == "parse":
            total += weight * parse_score(text)
        elif name == "parse_penalty":
            total += weight * parse_penalty_score(text)
        elif pred_ivs:
            total += weight * _interval_reward_score(name, pred_ivs, anno, reward_kwargs)
    return total


def score_intervals(
    pred_ivs: list, anno: dict, default_spec: str, spec_override: str | None = None
) -> float:
    """Same as ``score_completion`` but for already-parsed rollout intervals (no ``format``)."""
    spec = spec_override or anno.get("reward") or default_spec
    items = parse_spec(spec)
    if not pred_ivs or any(s >= e for s, e in pred_ivs):
        return 0.0
    total = 0.0
    for name, weight, reward_kwargs in items:
        if name == "format":
            continue
        total += weight * _interval_reward_score(name, pred_ivs, anno, reward_kwargs)
    return total


def make_reward_func(default_spec: str, verbose: bool = True):
    """Build the single dispatch reward function passed to the GRPO trainer."""

    def reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
        texts = [c[0]["content"] for c in completions]
        rewards = [score_completion(t, anno[i], default_spec) for i, t in enumerate(texts)]
        if verbose:
            clean = [_VISION_TOKEN_PATTERN.sub("", pt) for pt in prompt_text]
            for i, r in enumerate(rewards):
                _, ans = _intervals_from_text(texts[i])
                print(
                    f"prompt: {clean[i]} | answer: {ans} | gt: {anno[i]['span']} | reward: {r}"
                )
        return rewards

    reward.__name__ = "reward"
    return reward
