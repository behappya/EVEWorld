"""Aggregate MLR numbers into the figures the paper reports.

The metric itself is the Multi-Instance Localisation Rate (MLR): the share of
sampled timestamps of a clip at which the instruction's target is missing, with
a deviation counted only when it persists and when it is not explained by robot
occlusion (``persistence`` and ``occlusion`` modules). A clip is *eligible*
when its instruction asks for a target and the generated clip has somewhere for
that target to be; ineligible clips are excluded from both the numerator and
the denominator.

Reference figures, quoted from the paper's released numbers:

* DreamGenBench ``U_63``: 63 of 126 clips are eligible, i.e. a coverage of
  50.00%, and the MLR is reported over those 63 clips.
* WorldArena: 157 eligible clips, MLR 19.22%.
* The sensitivity grid over detection and occlusion thresholds moves the MLR
  by 70.0-88.9% in relative terms, which is why ``aggregate`` reports coverage
  and clip counts next to the MLR itself.

Every rate is reported in percent, so a clip with an MLR of 1.0 means 100%.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "WILSON_Z",
    "aggregate",
    "clip_mlr",
    "common_eligible",
    "coverage",
    "exact_mcnemar",
    "missing_rate",
    "wilson_interval",
]

WILSON_Z = 1.959963984540054


def _as_bool(values: Any) -> np.ndarray:
    """Coerce a sequence of booleans or 0/1 numbers into a ``bool`` array."""
    if isinstance(values, np.ndarray) and values.dtype == bool:
        return values
    return np.asarray(values, dtype=bool)


def _clip_eligible(eligible: Any, size: int) -> np.ndarray:
    """Build a per-timestamp eligibility mask of shape ``(T,)``."""
    scalar = np.isscalar(eligible)
    if scalar:
        return np.full(size, bool(eligible), dtype=bool)
    mask = _as_bool(eligible)
    if mask.shape != (size,):
        raise ValueError(
            f"eligible mask of shape {mask.shape} does not match {size} timestamps"
        )
    return mask


def clip_mlr(events: Sequence[bool] | np.ndarray, eligible: Any = True) -> float:
    """Return the MLR of one clip in percent.

    ``events`` is a per-timestamp boolean sequence of shape ``(T,)`` that is
    ``True`` where an MLR event was detected; ``eligible`` is either a scalar
    or a boolean mask of shape ``(T,)``. Timestamps that are not eligible are
    dropped from both the numerator and the denominator. An empty clip, or a
    clip with no eligible timestamps, yields ``0.0``.
    """
    array = _as_bool(events).reshape(-1)
    if array.size == 0:
        return 0.0
    mask = _clip_eligible(eligible, array.size)
    selected = array[mask]
    if selected.size == 0:
        return 0.0
    return float(100.0 * selected.mean())


def coverage(num_eligible: int, num_total: int) -> float:
    """Return the share of eligible clips in percent, guarding a zero total."""
    total = int(num_total)
    if total <= 0:
        return 0.0
    return float(100.0 * float(num_eligible) / float(total))


def missing_rate(rates: Iterable[Any], eligible: Sequence[bool] | None = None) -> tuple[float, int]:
    """Average per-clip MLR values across clips into ``(mean, contributing)``.

    ``rates`` holds one entry per clip; a boolean entry means a clip whose
    event flag is already known (``True`` -> ``100.0``), while a number is read
    as a fraction when ``abs(value) <= 1`` and as a percentage otherwise.
    ``None``, ``nan`` and ``inf`` entries are skipped. ``eligible`` is an
    optional per-clip boolean mask that must match the length of ``rates``.
    The returned pair is the mean over the clips that contributed and the
    number of those clips; an empty selection yields ``(0.0, 0)``.
    """
    values = list(rates)
    mask = None
    if eligible is not None:
        mask = _as_bool(eligible).reshape(-1)
        if mask.size != len(values):
            raise ValueError(
                f"eligible mask of size {mask.size} does not match {len(values)} clips"
            )

    total = 0.0
    contributing = 0
    for index, value in enumerate(values):
        if mask is not None and not mask[index]:
            continue
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            total += 100.0 if bool(value) else 0.0
            contributing += 1
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        if number < 0.0:
            raise ValueError(f"negative rate at clip {index}: {number!r}")
        total += number * 100.0 if number <= 1.0 else number
        contributing += 1

    if contributing == 0:
        return (0.0, 0)
    return (total / contributing, contributing)


def wilson_interval(
    events: int | Sequence[bool] | np.ndarray, count: int, z: float = WILSON_Z
) -> tuple[float, float]:
    """Return the Wilson score interval of a proportion in percent.

    ``events`` is either the number of clips that saw an event or a boolean
    sequence whose sum is used; ``count`` is the number of clips. ``z`` is the
    normal quantile, defaulting to the 97.5% point. Both bounds are clamped to
    ``[0, 100]``; a zero ``count`` yields ``(0.0, 0.0)``.
    """
    total = int(count)
    if total <= 0:
        return (0.0, 0.0)
    if isinstance(events, (bool, np.bool_)):
        observed = int(bool(events))
    elif isinstance(events, (int, np.integer)):
        observed = int(events)
    else:
        observed = int(np.count_nonzero(_as_bool(events)))
    p_hat = observed / total
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / total
    centre = (p_hat + z2 / (2.0 * total)) / denominator
    margin = float(z) * math.sqrt(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total ** 2)) / denominator
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    return (100.0 * low, 100.0 * high)


def _mcnemar_tail(a: int, b: int) -> float:
    """Return the exact two-sided McNemar p-value without scipy."""
    total = a + b
    if total <= 0:
        return 1.0
    tail = min(a, b)
    cumulative = sum(math.comb(total, k) for k in range(tail + 1))
    return min(1.0, 2.0 * cumulative / float(2 ** total))


def exact_mcnemar(a: int, b: int) -> float:
    """Return the exact (two-sided) McNemar test p-value for paired counts.

    ``a`` and ``b`` are the two discordant cell counts; the statistic follows a
    binomial distribution with ``n = a + b`` and ``p = 0.5``. ``scipy`` is used
    when it is installed, otherwise the binomial sum is evaluated directly with
    ``math.comb``. The released WorldArena comparison of 34 against 12 gives
    ``0.0016414913408482334``.
    """
    left = int(a)
    right = int(b)
    if left < 0 or right < 0:
        raise ValueError(f"discordant counts must be non-negative, got {left} and {right}")
    total = left + right
    if total <= 0:
        return 1.0
    try:
        from scipy.stats import binomtest
    except ImportError:
        return _mcnemar_tail(left, right)
    return float(binomtest(min(left, right), total, 0.5).pvalue)


def common_eligible(per_model: Mapping[str, Sequence[bool]] | Iterable[Sequence[bool]]) -> np.ndarray:
    """Return the intersection of per-model eligibility masks.

    ``per_model`` is either a mapping from model name to a boolean mask of
    shape ``(C,)`` or an iterable of such masks. All masks must share the same
    length, and the result has shape ``(C,)`` with ``True`` only where every
    model considered the clip eligible. An empty input yields an empty array.
    """
    masks = list(per_model.values()) if isinstance(per_model, Mapping) else list(per_model)
    if not masks:
        return np.zeros(0, dtype=bool)
    first = _as_bool(masks[0]).reshape(-1)
    result = first.copy()
    for mask in masks[1:]:
        other = _as_bool(mask).reshape(-1)
        if other.size != result.size:
            raise ValueError(
                f"eligibility mask of size {other.size} does not match {result.size} clips"
            )
        result &= other
    return result


def _row_events(row: Mapping[str, Any]) -> np.ndarray:
    """Extract a per-timestamp event array from an aggregate result row."""
    for key in ("events", "deviations"):
        value = row.get(key)
        if value is None:
            continue
        array = _as_bool(value)
        if array.ndim == 1:
            return array
    return np.zeros(0, dtype=bool)


def aggregate(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise per-clip MLR results into the reported figures.

    Each row may carry an ``events`` array of shape ``(T,)`` (preferred), a
    boolean ``event`` flag, or an ``mlr`` percentage, plus optional ``eligible``
    and ``error`` fields. The returned mapping has exactly the keys ``mlr``
    (mean over eligible clips, percent), ``coverage`` (share of eligible clips,
    percent), ``num_eligible``, ``num_clips``, ``events`` (eligible clips with
    at least one event) and ``errors`` (sorted error strings).
    """
    rows = list(results)
    num_clips = len(rows)
    num_eligible = 0
    with_events = 0
    rates: list[float] = []
    errors: list[str] = []

    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"result rows must be mappings, got {type(row).__name__}")
        error = row.get("error")
        if error:
            errors.append(str(error))
            continue
        if not bool(row.get("eligible", True)):
            continue
        num_eligible += 1
        rate: float | None = None
        events = _row_events(row)
        if events.size > 0:
            rate = clip_mlr(events, row.get("eligible", True))
            if bool(np.any(events)):
                with_events += 1
        elif "event" in row:
            flagged = bool(row["event"])
            rate = 100.0 if flagged else 0.0
            if flagged:
                with_events += 1
        elif row.get("mlr") is not None:
            rate = float(row["mlr"])
            if rate > 0.0:
                with_events += 1
        else:
            continue
        rates.append(rate)

    mean_mlr = float(np.mean(rates)) if rates else 0.0
    return {
        "mlr": mean_mlr,
        "coverage": coverage(num_eligible, num_clips),
        "num_eligible": num_eligible,
        "num_clips": num_clips,
        "events": with_events,
        "errors": sorted(errors),
    }
