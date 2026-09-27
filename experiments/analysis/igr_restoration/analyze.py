#!/usr/bin/env python3
"""Per-region restoration probe for the IGR weight map.

A clip that enters the IGR training objective carries a pasted interaction
region whose pixels differ from the clean clip, and a weight map that marks
exactly where the objective spends its reconstruction capacity. This script
re-runs the probe that reads the map back: it takes the clean clip, the
corrupted clip the sampler was conditioned on, the model output and the weight
map, and reports one row per region of the map.

For every region the probe reports the mean absolute error of the output
against the clean clip, the size of the perturbation the clip carried, and two
normalised readings of the error against the perturbation:

```text
restore   = mean |pred - corrupted| / mean |target - corrupted|
retention = mean |pred - target| / mean |target - corrupted|
```

``restore`` is one when the output matches the clean target and zero when it
keeps the corrupted content; ``retention`` is the opposite reading, one when
the injected content survives and zero when the model has removed it. The two
add up to one only when the output is a point on the segment between the clean
and corrupted clips. The direction cosine reports where the residual points:
``cos(pred - corrupted, target - corrupted)`` is one for a residual that moves
exactly along the restoration axis, zero for an orthogonal move and negative
when the output drifts further along the injected direction. The direction
cosine is the reading used by the earlier single-region restoration audit, kept
here so the per-region table and the clip-level number stay comparable.

Inputs are ``.npz`` files holding one array per clip in ``(T, H, W, C)`` layout;
``(T, C, H, W)`` arrays are detected and transposed, and arrays are looked up
under the common names ``pred``/``x0_hat``, ``target``/``clean`` and
``corrupted``/``input``. The weight map holds integer ``zones`` (or ``weights``,
from which zones are derived from the distinct values) on the frame grid, or on
the latent grid, in which case the labels are resampled to the frame grid with
nearest neighbours. Passing ``--corrupted`` is optional: without it the
perturbation size and the three normalised columns are reported as ``n/a``.

Run ``python analyze.py --pred a.npz --pred b.npz --target clean.npz \\
--corrupted corrupted.npz --weight-map wmap.npz`` to compare two arms over the
same region layout.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["main", "region_metrics"]

EPS = 1e-6
ZONE_KEYS = ("zones", "zone", "labels")
WEIGHT_KEYS = ("weights", "weight", "w")
ARRAY_KEYS = {
    "pred": ("pred", "x0_hat", "prediction", "output"),
    "target": ("target", "clean", "gt", "reference"),
    "corrupted": ("corrupted", "input", "erased", "perturbed"),
}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="npz with the model output; repeat the flag to compare several arms",
    )
    parser.add_argument("--target", required=True, help="npz with the clean clip")
    parser.add_argument("--corrupted", default=None, help="npz with the corrupted input clip")
    parser.add_argument("--weight-map", required=True, help="npz with the zone or weight map")
    parser.add_argument("--out", default=None, help="optional file for the markdown report")
    parser.add_argument(
        "--eps",
        type=float,
        default=EPS,
        help="denominator below which a normalised column is reported as n/a",
    )
    return parser.parse_args(argv)


def _pick(npz: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in npz:
            return npz[key]
    return None


def _load(path: str, role: str, keys: Sequence[str]) -> np.ndarray:
    try:
        with np.load(path) as npz:
            raw = _pick(npz, keys)
            if raw is None:
                available = ", ".join(sorted(npz.files))
                raise SystemExit(f"{path}: no {role} array found, keys are: {available}")
            array = np.asarray(raw, dtype=float)
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if array.ndim != 4:
        raise SystemExit(f"{path}: expected a 4-D array, got shape {array.shape}")
    if array.shape[-3] <= 4 < array.shape[-1]:
        array = np.transpose(array, (0, 2, 3, 1))
    return array


def _names_from(raw: Any) -> Mapping[int, str] | None:
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if getattr(raw, "shape", ()) == ():
        return None
    values = np.atleast_1d(np.asarray(raw)).tolist()
    if not all(isinstance(value, str) for value in values):
        return None
    return {index: value for index, value in enumerate(values)}


def _load_labels(path: str) -> tuple[np.ndarray, np.ndarray | None, Mapping[int, str] | None]:
    try:
        with np.load(path) as npz:
            zones = _pick(npz, ZONE_KEYS)
            weights = _pick(npz, WEIGHT_KEYS)
            names = npz["zone_names"] if "zone_names" in npz.files else None
            if zones is None and weights is None:
                available = ", ".join(sorted(npz.files))
                raise SystemExit(f"{path}: no zones or weights array found, keys are: {available}")
            if zones is None:
                _, zones = np.unique(np.asarray(weights, dtype=float), return_inverse=True)
            zones = np.asarray(zones)
            if zones.ndim != 3:
                raise SystemExit(f"{path}: expected zones of shape (T, H, W), got {zones.shape}")
            weights = np.asarray(weights, dtype=float) if weights is not None else None
            named = None if names is None else _names_from(names)
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if weights is not None and weights.shape != zones.shape:
        raise SystemExit(f"{path}: weights {weights.shape} do not match zones {zones.shape}")
    return zones.astype(int), weights, named


def _nearest(labels: np.ndarray, axis: int, size: int) -> np.ndarray:
    current = labels.shape[axis]
    index = np.minimum(np.arange(size) * current // size, current - 1)
    return np.take(labels, index, axis=axis)


def _align_labels(labels: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Resample a label map to ``shape`` with nearest neighbours, axis by axis."""
    for axis, size in enumerate(shape):
        if labels.shape[axis] != size:
            labels = _nearest(labels, axis, size)
    return labels


def _cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float | None:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < eps:
        return None
    return float(np.dot(a.ravel(), b.ravel()) / norm)


def region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    corrupted: np.ndarray | None,
    mask: np.ndarray,
    *,
    eps: float = EPS,
) -> dict[str, float]:
    """Metrics of one region; ``mask`` is a boolean array over frames and cells.

    Every mean is taken over the channels of the masked cells as well. The two
    normalised readings share the perturbation as denominator and are reported
    as ``nan`` when it falls below ``eps``. The direction cosine is computed per
    frame over the flattened masked region and averaged over the frames whose
    two vectors are both non-degenerate.
    """
    cells = int(np.count_nonzero(mask))
    out = {"cells": float(cells), "mae": float("nan")}
    if cells == 0:
        return out
    out["mae"] = float(np.abs(pred - target)[mask].mean())
    if corrupted is None:
        out.update(
            {
                "perturbation": float("nan"),
                "restore": float("nan"),
                "retention": float("nan"),
                "cosine": float("nan"),
            }
        )
        return out
    perturbation = np.abs(target - corrupted)[mask]
    span = float(perturbation.mean())
    out["perturbation"] = span
    if span < eps:
        out.update({"restore": float("nan"), "retention": float("nan")})
    else:
        out["restore"] = float(np.abs(pred - corrupted)[mask].mean()) / span
        out["retention"] = out["mae"] / span
    cosines = []
    for frame in range(pred.shape[0]):
        selected = mask[frame]
        if not np.any(selected):
            continue
        value = _cosine(
            (pred[frame] - corrupted[frame])[selected],
            (target[frame] - corrupted[frame])[selected],
            eps,
        )
        if value is not None:
            cosines.append(value)
    out["cosine"] = float(np.mean(cosines)) if cosines else float("nan")
    return out


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


HEADER = (
    "| Region | Cells | Cover % | Weight | Perturbation | MAE | Restore | Retention"
    " | Dir. cos. | Error share |"
)
SEPARATOR = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _row(name: str, metrics: Mapping[str, float], total_cells: int) -> str:
    cover = 100.0 * metrics["cells"] / total_cells if total_cells else 0.0
    return "| {} | {:d} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        name,
        int(metrics["cells"]),
        _fmt(cover, 2),
        _fmt(metrics.get("weight", float("nan")), 3),
        _fmt(metrics.get("perturbation", float("nan"))),
        _fmt(metrics.get("mae", float("nan"))),
        _fmt(metrics.get("restore", float("nan")), 3),
        _fmt(metrics.get("retention", float("nan")), 3),
        _fmt(metrics.get("cosine", float("nan")), 3),
        _fmt(metrics.get("error_share", float("nan")), 2),
    )


def _region_table(
    labels: np.ndarray,
    weights: np.ndarray | None,
    names: Mapping[int, str] | None,
    pred: np.ndarray,
    target: np.ndarray,
    corrupted: np.ndarray | None,
    eps: float,
) -> tuple[list[str], dict[int, float]]:
    total_cells = int(labels.size)
    error = np.abs(pred - target)
    masks = {int(value): labels == value for value in np.unique(labels)}
    weighted_error = {
        value: float(np.count_nonzero(mask)) * float(error[mask].mean())
        for value, mask in masks.items()
    }
    all_error = sum(weighted_error.values())
    restore_readings: dict[int, float] = {}
    lines = [HEADER, SEPARATOR]
    for value in sorted(masks):
        metrics = region_metrics(pred, target, corrupted, masks[value], eps=eps)
        if weights is not None:
            metrics["weight"] = float(weights[masks[value]].mean())
        share = 100.0 * weighted_error[value] / all_error if all_error > eps else float("nan")
        metrics["error_share"] = share
        name = (names or {}).get(value, f"region {value}")
        lines.append(_row(name, metrics, total_cells))
        restore_readings[value] = metrics.get("restore", float("nan"))
    metrics = region_metrics(pred, target, corrupted, np.ones_like(labels, dtype=bool), eps=eps)
    metrics["cells"] = float(total_cells)
    if weights is not None:
        metrics["weight"] = float(weights.mean())
    metrics["error_share"] = 100.0 if all_error > eps else float("nan")
    lines.append(_row("all labelled", metrics, total_cells))
    return lines, restore_readings


def _comparison(
    title: str,
    arms: Sequence[tuple[str, Mapping[int, float]]],
    names: Mapping[int, str] | None,
    digits: int,
) -> list[str]:
    labels = sorted({int(value) for _, table in arms for value in table})
    header = "| Region | " + " | ".join(name for name, _ in arms) + " |"
    lines = [f"## {title}", "", header, "|---|" + "---:|" * len(arms)]
    for label in labels:
        cells = [_fmt(table.get(label, float("nan")), digits) for _, table in arms]
        name = (names or {}).get(label, f"region {label}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    target = _load(args.target, "target", ARRAY_KEYS["target"])
    corrupted = None
    if args.corrupted is not None:
        corrupted = _load(args.corrupted, "corrupted", ARRAY_KEYS["corrupted"])
        if corrupted.shape != target.shape:
            raise SystemExit(f"{args.corrupted}: shape {corrupted.shape} does not match {target.shape}")
    labels, weights, names = _load_labels(args.weight_map)
    labels = _align_labels(labels, target.shape[:3])
    if weights is not None:
        weights = _align_labels(weights, target.shape[:3])
    arms = []
    for path in args.pred:
        pred = _load(path, "pred", ARRAY_KEYS["pred"])
        if pred.shape[:3] != target.shape[:3]:
            raise SystemExit(f"{path}: shape {pred.shape} does not match the clip {target.shape}")
        arms.append((path, pred))
    report = ["# IGR restoration probe", ""]
    report.append(f"clean clip `{args.target}`, region map `{args.weight_map}`")
    if args.corrupted is not None:
        report.append(f"corrupted clip `{args.corrupted}`")
    report.append("")
    restore_tables = []
    for path, pred in arms:
        report.extend([f"## `{path}`", ""])
        lines, restore = _region_table(labels, weights, names, pred, target, corrupted, args.eps)
        report.extend(lines)
        report.append("")
        restore_tables.append((path, restore))
    if len(arms) > 1:
        report.extend(_comparison("Restore, per region", restore_tables, names, 3))
    text = "\n".join(report).rstrip() + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
