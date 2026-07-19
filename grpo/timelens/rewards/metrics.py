# Parser utilities for temporal grounding (extract_time, extract_answer, iou)
# extract_time: JSON [[start,end], ...] in seconds (single ANSWER_TEMPLATES entry in format_version2).

from __future__ import annotations

import json
import math
import re

# CoT blocks: <thinking> (lvdb_cot prompt) or <think> (legacy).
_THINKING_TAG = r"(?:redacted_)?thinking"
COT_FORMAT_PATTERN = re.compile(
    rf"<{_THINKING_TAG}>.*?</{_THINKING_TAG}>\s*<answer>.*?</answer>",
    re.DOTALL,
)
_COT_WITH_ANSWER_RE = re.compile(
    rf"<{_THINKING_TAG}>.*?</{_THINKING_TAG}>\s*<answer>(.*?)</answer>",
    re.DOTALL,
)
_ANSWER_ONLY_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STRUCTURE_TAGS = (
    "<thinking>",
    "</thinking>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
)


def extract_answer(content: str) -> str:
    """
    Extract the answer within <answer></answer> tags.
    Supports optional <thinking> or <think> before <answer>.
    If the format is not correct, return content as is.
    """
    text = content or ""
    match = _COT_WITH_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _ANSWER_ONLY_RE.search(text)
    if match:
        return match.group(1).strip()
    if any(tag in text for tag in _STRUCTURE_TAGS):
        return text
    return text


def iou_span(a, b) -> float:
    """Temporal IoU between two spans [start, end] (seconds)."""
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = max1 - min0
    if denom <= 0:
        return 0.0
    return max(min1 - max0, 0) / denom


def _is_flat_interval(x) -> bool:
    return (
        isinstance(x, (list, tuple))
        and len(x) == 2
        and all(isinstance(v, (int, float)) for v in x)
    )


def _as_interval_list(x) -> list[list[float]]:
    if x is None or x == []:
        return []
    if _is_flat_interval(x):
        return [[float(x[0]), float(x[1])]]
    out: list[list[float]] = []
    for p in x:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
    return out


def _merge_time_spans(intervals: list[list[float]]) -> list[list[float]]:
    """Merge overlapping/adjacent 1D intervals (same rule as VLMEvalKit vidi_vue_tr._merge_time_spans)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = [list(intervals[0])]
    for current in intervals[1:]:
        prev_end = merged[-1][1]
        curr_start, curr_end = current[0], current[1]
        if curr_start <= prev_end:
            merged[-1][1] = max(prev_end, curr_end)
        else:
            merged.append([curr_start, curr_end])
    return merged


def iou_overlap_ratio(pred, gt) -> float:
    """
    IoU between two multi-span sets (VLMEvalKit vidi_vue_tr._overlap_ratio / VUE_TR qa_eval).
    Pred spans are merged before computing overlap sums; GT is not merged (same as reference).
    """
    pred_ivs = _as_interval_list(pred)
    gt_ivs = _as_interval_list(gt)
    if not gt_ivs:
        return 1.0 if not pred_ivs else 0.0
    if not pred_ivs:
        return 0.0

    pred_m = _merge_time_spans(pred_ivs)
    pred_m = [[s, e] for s, e in pred_m if s <= e]
    if not pred_m:
        return 0.0

    len_gt = sum(e - s for s, e in gt_ivs)
    len_pred = sum(e - s for s, e in pred_m)
    union = len_pred + len_gt
    intersect = 0.0
    for pi in pred_m:
        for gj in gt_ivs:
            sta = max(pi[0], gj[0])
            end = min(pi[1], gj[1])
            intersect += max(0.0, end - sta)
    union -= intersect
    return float(max(0.0, min(1.0, intersect / (union + 1e-16))))


def iou(A, B):
    """
    Two flat spans [s,e] -> classic temporal IoU (same formula as single-span overlap_ratio).
    Otherwise -> multi-span overlap IoU (merged pred, pairwise pred×gt intersection sum).
    """
    if _is_flat_interval(A) and _is_flat_interval(B):
        return iou_span(A, B)
    return iou_overlap_ratio(A, B)


def giou_span(a, b) -> float:
    """
    Temporal GIoU between two spans [start, end] (seconds).

    GIoU = IoU - (|C| - |U|) / |C|, where C is the smallest interval enclosing both spans
    and U is their union length. Range [-1, 1]; uses the same IoU definition as ``iou_span``.
    """
    a0, a1 = float(a[0]), float(a[1])
    b0, b1 = float(b[0]), float(b[1])
    c_len = max(a1, b1) - min(a0, b0)
    if c_len <= 0:
        return 0.0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    iou_v = iou_span(a, b)
    return float(max(-1.0, min(1.0, iou_v - (c_len - union) / c_len)))


def giou_overlap_ratio(pred, gt) -> float:
    """
    GIoU between two multi-span sets (same merge / intersection rules as ``iou_overlap_ratio``).
    C is the smallest interval enclosing all pred (merged) and GT spans.
    """
    pred_ivs = _as_interval_list(pred)
    gt_ivs = _as_interval_list(gt)
    if not gt_ivs:
        return 1.0 if not pred_ivs else 0.0
    if not pred_ivs:
        return 0.0

    pred_m = _merge_time_spans(pred_ivs)
    pred_m = [[s, e] for s, e in pred_m if s <= e]
    if not pred_m:
        return 0.0

    len_gt = sum(e - s for s, e in gt_ivs)
    len_pred = sum(e - s for s, e in pred_m)
    union = len_pred + len_gt
    intersect = 0.0
    for pi in pred_m:
        for gj in gt_ivs:
            sta = max(pi[0], gj[0])
            end = min(pi[1], gj[1])
            intersect += max(0.0, end - sta)
    union -= intersect

    all_ivs = pred_m + gt_ivs
    c_len = max(e for _, e in all_ivs) - min(s for s, _ in all_ivs)
    if c_len <= 0:
        return 0.0

    iou_v = intersect / (union + 1e-16)
    giou_v = iou_v - (c_len - union) / c_len
    return float(max(-1.0, min(1.0, giou_v)))


def giou(A, B):
    """
    Two flat spans [s,e] -> ``giou_span``; otherwise multi-span GIoU (merged pred, GT not merged).
    """
    if _is_flat_interval(A) and _is_flat_interval(B):
        return giou_span(A, B)
    return giou_overlap_ratio(A, B)


def span_pairwise_intersection(
    a,
    b,
) -> list[list[float]]:
    """
    Intersection of two time-span sets on the 1D timeline: points that lie in both
    (union of pairwise interval intersections, then merged). Empty if no overlap
    or either input is empty.
    """
    a_ivs = _as_interval_list(a)
    b_ivs = _as_interval_list(b)
    if not a_ivs or not b_ivs:
        return []
    out: list[list[float]] = []
    for pi in a_ivs:
        for gj in b_ivs:
            sta = max(float(pi[0]), float(gj[0]))
            end = min(float(pi[1]), float(gj[1]))
            if end > sta:
                out.append([sta, end])
    if not out:
        return []
    return _merge_time_spans(out)


def temporal_precision_flat(pred, gt) -> float:
    """|pred ∩ gt| / |pred| for two flat spans [start, end] (seconds)."""
    len_p = float(pred[1]) - float(pred[0])
    if len_p <= 0:
        return 0.0
    sta = max(pred[0], gt[0])
    end = min(pred[1], gt[1])
    inter = max(0.0, end - sta)
    return float(max(0.0, min(1.0, inter / (len_p + 1e-16))))


def temporal_precision_overlap_ratio(pred, gt) -> float:
    """
    Temporal precision |pred ∩ gt| / |pred| with the same merge / intersection rules as
    ``iou_overlap_ratio`` (merged pred, sum of pred×gt overlaps, GT not merged).
    """
    pred_ivs = _as_interval_list(pred)
    gt_ivs = _as_interval_list(gt)
    if not pred_ivs:
        return 0.0
    if not gt_ivs:
        return 0.0

    pred_m = _merge_time_spans(pred_ivs)
    pred_m = [[s, e] for s, e in pred_m if s <= e]
    if not pred_m:
        return 0.0

    len_pred = sum(e - s for s, e in pred_m)
    if len_pred <= 0:
        return 0.0

    intersect = 0.0
    for pi in pred_m:
        for gj in gt_ivs:
            sta = max(pi[0], gj[0])
            end = min(pi[1], gj[1])
            intersect += max(0.0, end - sta)

    return float(max(0.0, min(1.0, intersect / (len_pred + 1e-16))))


def temporal_precision(A, B) -> float:
    """
    |pred ∩ gt| / |pred| where ``A`` is prediction and ``B`` is ground truth.
    Branching matches ``iou`` so IoU and precision are comparable on the same inputs.
    """
    if _is_flat_interval(A) and _is_flat_interval(B):
        return temporal_precision_flat(A, B)
    return temporal_precision_overlap_ratio(A, B)


def temporal_wasserstein_reward(pred, gt, beta: float = 3.0) -> float:
    """
    pred, gt: [[start, end], ...]
    R = exp(-beta * W1 / |merge(gt)|)

    Pred / GT are merged, then normalized to uniform distributions on their own
    merged support. W1 is the L1 distance between their CDFs, and the reward is
    normalized only by the merged target duration so broad predictions cannot
    enlarge their own normalization factor.
    """

    def merge_intervals(intervals):
        intervals = sorted(
            (float(s), float(e))
            for s, e in intervals
            if float(e) > float(s)
        )
        if not intervals:
            return []

        merged = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def total_length(intervals):
        return sum(e - s for s, e in intervals)

    def cdf_at(intervals, density, x):
        value = 0.0
        for s, e in intervals:
            if x <= s:
                continue
            value += max(0.0, min(x, e) - s) * density
        return min(1.0, max(0.0, value))

    def slope_at(intervals, density, x):
        return sum(density for s, e in intervals if s <= x < e)

    pred = merge_intervals(_as_interval_list(pred))
    gt = merge_intervals(_as_interval_list(gt))

    pred_len = total_length(pred)
    gt_len = total_length(gt)

    if pred_len == 0 and gt_len == 0:
        return 1.0
    if pred_len == 0 or gt_len == 0:
        return 0.0

    pred_density = 1.0 / pred_len
    gt_density = 1.0 / gt_len

    breakpoints = sorted({x for interval in pred + gt for x in interval})

    w1 = 0.0

    for a, b in zip(breakpoints, breakpoints[1:]):
        if b <= a:
            continue

        mid = (a + b) / 2.0

        y0 = cdf_at(pred, pred_density, a) - cdf_at(gt, gt_density, a)
        slope = slope_at(pred, pred_density, mid) - slope_at(gt, gt_density, mid)
        y1 = y0 + slope * (b - a)

        if abs(slope) > 1e-12 and y0 * y1 < 0:
            zero = a - y0 / slope
            w1 += abs(y0) * (zero - a) / 2.0
            w1 += abs(y1) * (b - zero) / 2.0
        else:
            w1 += (abs(y0) + abs(y1)) * (b - a) / 2.0

    return math.exp(-beta * w1 / max(gt_len, 1e-12))


def _valid_merged_intervals(intervals) -> list[tuple[float, float]]:
    """Convert an interval-like object into valid, merged temporal intervals."""
    valid: list[list[float]] = []
    for s, e in _as_interval_list(intervals):
        s = float(s)
        e = float(e)
        if math.isfinite(s) and math.isfinite(e) and e > s:
            valid.append([s, e])
    return [tuple(span) for span in _merge_time_spans(valid)]


def _interval_union_length(intervals: list[tuple[float, float]]) -> float:
    return float(sum(e - s for s, e in intervals))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _integrate_abs_cdf_gap(cdf_pred, cdf_gt, lo: float, hi: float, steps: int = 512) -> float:
    """Trapezoidal integration of |F_pred - F_gt| on a finite 1D domain."""
    if hi <= lo:
        return 0.0
    steps = max(16, int(steps))
    dx = (hi - lo) / steps
    prev = abs(cdf_pred(lo) - cdf_gt(lo))
    total = 0.0
    for idx in range(1, steps + 1):
        x = lo + idx * dx
        cur = abs(cdf_pred(x) - cdf_gt(x))
        total += 0.5 * (prev + cur) * dx
        prev = cur
    return float(total)


def temporal_endpoint_dirac_wasserstein_reward(pred, gt, beta: float = 3.0) -> float:
    """
    Choice 2: endpoint Dirac mixture.

    Each merged interval contributes two atoms, one at its start and one at its end.
    W1 is the exact 1D EMD between the two endpoint empirical distributions.
    """
    pred_ivs = _valid_merged_intervals(pred)
    gt_ivs = _valid_merged_intervals(gt)
    if not pred_ivs and not gt_ivs:
        return 1.0
    if not pred_ivs or not gt_ivs:
        return 0.0

    pred_points = sorted(x for s, e in pred_ivs for x in (s, e))
    gt_points = sorted(x for s, e in gt_ivs for x in (s, e))
    breakpoints = sorted(set(pred_points + gt_points))
    if len(breakpoints) <= 1:
        return 1.0

    def cdf_at(points, x):
        return sum(1 for p in points if p <= x) / len(points)

    w1 = 0.0
    for a, b in zip(breakpoints, breakpoints[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        w1 += abs(cdf_at(pred_points, mid) - cdf_at(gt_points, mid)) * (b - a)

    norm = _interval_union_length(_merge_time_spans([list(x) for x in pred_ivs + gt_ivs]))
    if norm <= 0:
        norm = breakpoints[-1] - breakpoints[0]
    return math.exp(-beta * w1 / max(norm, 1e-12))


def temporal_center_gaussian_wasserstein_reward(
    pred,
    gt,
    beta: float = 3.0,
    alpha: float = 0.5,
    steps: int = 512,
) -> float:
    """
    Choice 3: center Gaussian mixture.

    Each merged interval [s, e] becomes N(c, (alpha*l)^2), c=(s+e)/2, l=e-s.
    Multi-interval Gaussian mixtures use the W1 CDF integral with finite-grid
    quadrature over a 6-sigma padded domain.
    """
    pred_ivs = _valid_merged_intervals(pred)
    gt_ivs = _valid_merged_intervals(gt)
    if not pred_ivs and not gt_ivs:
        return 1.0
    if not pred_ivs or not gt_ivs:
        return 0.0

    def components(intervals):
        out = []
        for s, e in intervals:
            length = e - s
            sigma = max(abs(alpha) * length, 1e-6)
            out.append(((s + e) / 2.0, sigma))
        return out

    pred_comp = components(pred_ivs)
    gt_comp = components(gt_ivs)
    all_comp = pred_comp + gt_comp
    lo = min(mu - 6.0 * sigma for mu, sigma in all_comp)
    hi = max(mu + 6.0 * sigma for mu, sigma in all_comp)

    def make_cdf(comp):
        inv_n = 1.0 / len(comp)
        return lambda x: sum(_normal_cdf((x - mu) / sigma) for mu, sigma in comp) * inv_n

    w1 = _integrate_abs_cdf_gap(make_cdf(pred_comp), make_cdf(gt_comp), lo, hi, steps=steps)
    norm = _interval_union_length(_merge_time_spans([list(x) for x in pred_ivs + gt_ivs]))
    return math.exp(-beta * w1 / max(norm, 1e-12))


def temporal_boundary_gaussian_wasserstein_reward(
    pred,
    gt,
    beta: float = 3.0,
    sigma_ratio: float = 0.08,
    steps: int = 512,
) -> float:
    """
    Choice 4: boundary Gaussian mixture.

    Each merged interval contributes two Gaussians centered at s and e. Sigma is
    proportional to interval length, giving a soft boundary-uncertainty reward.
    """
    pred_ivs = _valid_merged_intervals(pred)
    gt_ivs = _valid_merged_intervals(gt)
    if not pred_ivs and not gt_ivs:
        return 1.0
    if not pred_ivs or not gt_ivs:
        return 0.0

    def components(intervals):
        out = []
        for s, e in intervals:
            length = e - s
            sigma = max(abs(sigma_ratio) * length, 1e-6)
            out.append((s, sigma))
            out.append((e, sigma))
        return out

    pred_comp = components(pred_ivs)
    gt_comp = components(gt_ivs)
    all_comp = pred_comp + gt_comp
    lo = min(mu - 6.0 * sigma for mu, sigma in all_comp)
    hi = max(mu + 6.0 * sigma for mu, sigma in all_comp)

    def make_cdf(comp):
        inv_n = 1.0 / len(comp)
        return lambda x: sum(_normal_cdf((x - mu) / sigma) for mu, sigma in comp) * inv_n

    w1 = _integrate_abs_cdf_gap(make_cdf(pred_comp), make_cdf(gt_comp), lo, hi, steps=steps)
    norm = _interval_union_length(_merge_time_spans([list(x) for x in pred_ivs + gt_ivs]))
    return math.exp(-beta * w1 / max(norm, 1e-12))


def temporal_wasserstein2_reward(pred, gt, beta: float = 3.0) -> float:
    """
    pred, gt: [[start, end], ...]
    R = exp(-beta * W2 / L)

    Pred / GT are merged for overlaps, normalized to uniform distributions on their
    own union support. W2 follows the 1D quantile formula:
        W2 = sqrt(int_0^1 (F_pred^-1(u) - F_gt^-1(u))^2 du)
    """

    def merge_intervals(intervals):
        intervals = sorted(
            (float(s), float(e))
            for s, e in intervals
            if float(e) > float(s)
        )
        if not intervals:
            return []

        merged = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def total_length(intervals):
        return sum(e - s for s, e in intervals)

    def mass_breakpoints(intervals, length):
        out = [0.0]
        acc = 0.0
        for s, e in intervals:
            acc += (e - s) / length
            out.append(min(1.0, max(0.0, acc)))
        out[-1] = 1.0
        return out

    def quantile_at(intervals, length, u):
        if u <= 0.0:
            return intervals[0][0]
        if u >= 1.0:
            return intervals[-1][1]
        target = u * length
        acc = 0.0
        for s, e in intervals:
            span = e - s
            next_acc = acc + span
            if target <= next_acc:
                return s + max(0.0, target - acc)
            acc = next_acc
        return intervals[-1][1]

    pred = merge_intervals(_as_interval_list(pred))
    gt = merge_intervals(_as_interval_list(gt))

    pred_len = total_length(pred)
    gt_len = total_length(gt)

    if pred_len == 0 and gt_len == 0:
        return 1.0
    if pred_len == 0 or gt_len == 0:
        return 0.0

    union_len = total_length(merge_intervals(pred + gt))
    if union_len == 0:
        return 0.0

    breakpoints = sorted(set(mass_breakpoints(pred, pred_len) + mass_breakpoints(gt, gt_len)))
    sq_dist = 0.0
    slope_diff = pred_len - gt_len

    for a, b in zip(breakpoints, breakpoints[1:]):
        if b <= a:
            continue
        width = b - a
        mid = (a + b) / 2.0
        diff_mid = quantile_at(pred, pred_len, mid) - quantile_at(gt, gt_len, mid)
        sq_dist += diff_mid * diff_mid * width + slope_diff * slope_diff * width**3 / 12.0

    w2 = math.sqrt(max(0.0, sq_dist))
    return math.exp(-beta * w2 / union_len)


# --- extract_time: JSON array of [start, end] pairs in seconds ---

_JSON_SCALAR = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _json_scalar_to_float(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str) and _JSON_SCALAR.match(x.strip()):
        try:
            v = float(x.strip())
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def _coerce_interval_pair(p) -> tuple[float, float] | None:
    """Unwrap nested single-element lists (e.g. [[[a,b]]]) until a pair of numeric endpoints."""
    cur = p
    for _ in range(8):
        if not isinstance(cur, (list, tuple)) or not cur:
            return None
        if len(cur) >= 2:
            a, b = cur[0], cur[1]
            sa, sb = _json_scalar_to_float(a), _json_scalar_to_float(b)
            if sa is not None and sb is not None:
                return (sa, sb)
        if len(cur) == 1 and isinstance(cur[0], (list, tuple)):
            cur = cur[0]
            continue
        return None
    return None


def _intervals_from_json_data(data) -> list[tuple[float, float]] | None:
    """Parse JSON list into (start,end) seconds; numeric pairs only."""
    if not isinstance(data, list) or not data:
        return None
    while len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if not data:
        return None
    first = data[0]
    if isinstance(first, (int, float)):
        if len(data) >= 2 and all(isinstance(x, (int, float)) for x in data[:2]):
            return [(float(data[0]), float(data[1]))]
        return None
    if isinstance(first, (list, tuple)):
        pairs: list[tuple[float, float]] = []
        for p in data:
            if not isinstance(p, (list, tuple)):
                continue
            pair = _coerce_interval_pair(p)
            if pair is not None:
                pairs.append(pair)
        return pairs if pairs else None
    if len(data) >= 2:
        sa, sb = _json_scalar_to_float(data[0]), _json_scalar_to_float(data[1])
        if sa is not None and sb is not None:
            return [(sa, sb)]
    return None


def _balanced_arrays_from_double_bracket(s: str) -> list[str]:
    """Each segment starts at '[[' and ends at the matching bracket depth (supports [[a,b], [c,d]])."""
    out: list[str] = []
    i = 0
    while True:
        start = s.find("[[", i)
        if start == -1:
            break
        depth = 0
        pos = start
        while pos < len(s):
            ch = s[pos]
            if ch == "[":
                depth += 1
                pos += 1
            elif ch == "]":
                depth -= 1
                pos += 1
                if depth == 0:
                    out.append(s[start:pos])
                    i = pos
                    break
            else:
                pos += 1
        else:
            break
    return out


def _json_candidates(paragraph: str) -> list[str]:
    s = paragraph.strip()
    out = [s]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        out.append(fence.group(1).strip())
    out.extend(_balanced_arrays_from_double_bracket(s))
    seen: set[str] = set()
    dedup: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def extract_time(paragraph: str) -> list[tuple[float, float]]:
    """Parse JSON [[start,end], ...] timestamps in seconds (format_version2 ANSWER_TEMPLATES)."""
    if not paragraph or not paragraph.strip():
        return []

    for cand in _json_candidates(paragraph):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        ivs = _intervals_from_json_data(data)
        if ivs:
            return [(min(s, e), max(s, e)) for s, e in ivs]

    return []
