#!/usr/bin/env python3
"""Recompute the MLR sensitivity table over the occlusion threshold and ``k``.

The released protocol fixes two knobs: the occlusion-overlap threshold
``tau_occ`` used by the exemption and the persistence requirement ``k``. This
script recomputes the MLR of one or more models from cached per-clip records,
so the whole ``tau_occ x k`` grid can be rebuilt without regenerating videos or
re-running detection; the cached counts, tracks, masks, queries and eligible
items stay fixed, which is what makes the cells comparable.

Every grid cell lives in its own directory named ``tau_<value>_k<order>`` (a
hyphen works in place of either underscore), holding one JSON file per clip:

```text
{"request_id": "0001",
 "eligible": true,
 "expected": 2,
 "counts": [[1, 1], [1, 0], [1, 1]],
 "occlusion_ratio": [[0.02, 0.00], [0.74, 0.11], [0.03, 0.02]]}
```

``counts[t][i]`` flags whether expected instance ``i`` was detected at
timestamp ``t`` and ``occlusion_ratio[t][i]`` is the fraction of that instance
covered by the robot arm after the evaluator's reliability gate. Both arrays
share the shape ``(T,)`` or ``(T, N)``; ``occlusion_ratio`` may be omitted for
a clip that never needs the exemption. A timestamp is exempted when every
instance missing there is occluded, which is what
:func:`eveworld.evaluation.mlr.adjust_counts` implements once the ratios are
turned into witnesses with the strict ``> tau_occ`` test of the released
evaluator; the persistence filter is the package's own
:func:`~eveworld.evaluation.mlr.detect_events`. Every clip with an ``error``
field is excluded, exactly as in a normal evaluation run.

The script prints one row per grid with the event count over the eligible set
and the MLR in percent, and, when ``--baseline-root`` points at a second sweep
with the same layout, the matching baseline cell and the relative reduction.
A second table reports the mean per-clip event share, which weights every clip
equally instead of counting each clip once.

Run it with the package importable, as in the released schedule:

```bash
PYTHONPATH=src python experiments/analysis/mlr_sensitivity/sweep.py \
    --root outputs/mlr_sweep/eveworld --baseline-root outputs/mlr_sweep/sft
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["main", "grid_name", "load_clip", "sweep_grid"]

GRID_PATTERN = re.compile(r"^tau[-_]?([0-9]*\.?[0-9]+)[-_]k(\d+)$")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="sweep root with one subdirectory per (tau, k) cell")
    parser.add_argument(
        "--baseline-root",
        default=None,
        help="optional second sweep with the same layout, reported as the baseline of every cell",
    )
    parser.add_argument("--out", default=None, help="optional file for the markdown report")
    return parser.parse_args(argv)


def _import_mlr() -> tuple[Callable[..., Any], ...]:
    try:
        from eveworld.evaluation.mlr import adjust_counts, aggregate, detect_events
    except ImportError as exc:
        raise SystemExit(
            f"the eveworld package is required for the MLR rule ({exc}); " "install it with 'pip install -e .' or run with PYTHONPATH=src"
        ) from exc
    return adjust_counts, aggregate, detect_events


def grid_name(directory: Path) -> tuple[float, int]:
    """Read the ``(tau_occ, k)`` cell out of a directory name."""
    match = GRID_PATTERN.match(directory.name)
    if match is None:
        raise SystemExit(f"{directory}: expected a directory name like tau_0.15_k2")
    tau_occ = float(match.group(1))
    order = int(match.group(2))
    if order < 1:
        raise SystemExit(f"{directory}: k must be a positive integer, got {order}")
    return tau_occ, order


def load_clip(path: Path) -> dict[str, Any]:
    """Read one cached per-clip record and validate its fields."""
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise SystemExit(f"{path}: expected a JSON object, got {type(record).__name__}")
    if record.get("error"):
        return dict(record)
    for field in ("expected", "counts"):
        if field not in record:
            raise SystemExit(f"{path}: missing field {field!r}")
    counts = np.asarray(record["counts"], dtype=int)
    if counts.ndim not in (1, 2):
        raise SystemExit(f"{path}: counts must have shape (T,) or (T, N), got {counts.shape}")
    ratios = record.get("occlusion_ratio")
    if ratios is None:
        ratios = np.zeros_like(counts, dtype=float)
    else:
        ratios = np.asarray(ratios, dtype=float)
        if ratios.shape != counts.shape:
            raise SystemExit(f"{path}: occlusion_ratio {ratios.shape} does not match counts {counts.shape}")
    if record.get("eligible", True) not in (True, False):
        raise SystemExit(f"{path}: eligible must be a boolean, got {record['eligible']!r}")
    return {
        "request_id": record.get("request_id", path.stem),
        "eligible": bool(record.get("eligible", True)),
        "expected": int(record["expected"]),
        "counts": counts,
        "occlusion_ratio": ratios,
    }


def sweep_grid(directory: Path, tau_occ: float, order: int, api: Sequence[Callable[..., Any]]) -> dict[str, Any]:
    """Rebuild every per-clip event trace of one grid cell and aggregate them."""
    adjust_counts, aggregate, detect_events = api
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise SystemExit(f"{directory}: no per-clip JSON files found")
    rows: list[dict[str, Any]] = []
    for path in paths:
        record = load_clip(path)
        if record.get("error"):
            rows.append({"eligible": False, "error": record["error"]})
            continue
        expected = record["expected"]
        witnesses = record["occlusion_ratio"] > tau_occ
        adjusted = adjust_counts(record["counts"], witnesses, expected)
        deviations = adjusted != expected
        events = detect_events(deviations, None, k=order)
        rows.append({"events": events, "eligible": record["eligible"]})
    result = aggregate(rows)
    eligible = int(result["num_eligible"])
    flagged = int(result["events"])
    return {
        "tau_occ": tau_occ,
        "k": order,
        "events": flagged,
        "num_eligible": eligible,
        "num_clips": int(result["num_clips"]),
        "event_share": 100.0 * flagged / eligible if eligible else 0.0,
        "mean_clip_rate": float(result["mlr"]),
        "errors": result["errors"],
    }


def _pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _reduction(cell: Mapping[str, Any], baseline: Mapping[str, Any]) -> float | None:
    base = float(baseline["event_share"])
    if base <= 0.0:
        return None
    return 100.0 * (1.0 - float(cell["event_share"]) / base)


def _report(cells: Sequence[dict[str, Any]], baselines: Mapping[tuple[float, int], dict[str, Any]]) -> str:
    lines = ["# MLR sensitivity", ""]
    if baselines:
        lines.append("| tau_occ | k | Baseline events / eligible | Baseline MLR (%) " "| Events / eligible | MLR (%) | Rel. reduction (%) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for cell in cells:
            baseline = baselines.get((cell["tau_occ"], cell["k"]))
            if baseline is None:
                lines.append(
                    f"| {_pct(cell['tau_occ'])} | {cell['k']} | n/a | n/a "
                    f"| {cell['events']} / {cell['num_eligible']} | {_pct(cell['event_share'])} | n/a |"
                )
                continue
            change = _reduction(cell, baseline)
            lines.append(
                f"| {_pct(cell['tau_occ'])} | {cell['k']} "
                f"| {baseline['events']} / {baseline['num_eligible']} | {_pct(baseline['event_share'])} "
                f"| {cell['events']} / {cell['num_eligible']} | {_pct(cell['event_share'])} "
                f"| {_pct(change, 1) if change is not None else 'n/a'} |"
            )
    else:
        lines.append("| tau_occ | k | Events / eligible | MLR (%) |")
        lines.append("|---:|---:|---:|---:|")
        for cell in cells:
            lines.append(f"| {_pct(cell['tau_occ'])} | {cell['k']} " f"| {cell['events']} / {cell['num_eligible']} | {_pct(cell['event_share'])} |")
    lines.append("")
    lines.append("Mean per-clip event share, one weight per clip:")
    lines.append("")
    if baselines:
        lines.append("| tau_occ | k | Baseline mean clip rate (%) | Mean clip rate (%) |")
        lines.append("|---:|---:|---:|---:|")
        for cell in cells:
            baseline = baselines.get((cell["tau_occ"], cell["k"]))
            base_text = _pct(baseline["mean_clip_rate"]) if baseline is not None else "n/a"
            lines.append(f"| {_pct(cell['tau_occ'])} | {cell['k']} | {base_text} " f"| {_pct(cell['mean_clip_rate'])} |")
    else:
        lines.append("| tau_occ | k | Mean clip rate (%) |")
        lines.append("|---:|---:|---:|")
        for cell in cells:
            lines.append(f"| {_pct(cell['tau_occ'])} | {cell['k']} | {_pct(cell['mean_clip_rate'])} |")
    lines.append("")
    taus = sorted({cell["tau_occ"] for cell in cells})
    orders = sorted({cell["k"] for cell in cells})
    eligible_counts = sorted({cell["num_eligible"] for cell in cells})
    lines.append(f"grids: {len(cells)}; tau_occ in [{', '.join(_pct(tau) for tau in taus)}]; " f"k in [{', '.join(str(order) for order in orders)}]")
    lines.append(
        "eligible clips per cell: " + (", ".join(str(count) for count in eligible_counts) if len(eligible_counts) > 1 else str(eligible_counts[0]))
    )
    errors = sorted({error for cell in cells for error in cell["errors"]})
    lines.append("clip errors: " + (str(len(errors)) + " distinct" if errors else "none"))
    if baselines:
        changes = []
        for cell in cells:
            baseline = baselines.get((cell["tau_occ"], cell["k"]))
            if baseline is None:
                continue
            change = _reduction(cell, baseline)
            if change is not None:
                changes.append(change)
        if changes:
            lines.append(
                f"relative reduction against the baseline: min {_pct(min(changes), 1)}%, "
                f"max {_pct(max(changes), 1)}% over {len(changes)} matched cells"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"{root}: not a directory")
    api = _import_mlr()
    cells = []
    for directory in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        if not GRID_PATTERN.match(directory.name):
            continue
        tau_occ, order = grid_name(directory)
        cells.append(sweep_grid(directory, tau_occ, order, api))
    if not cells:
        raise SystemExit(f"{root}: no subdirectories named like tau_0.15_k2")
    cells.sort(key=lambda cell: (cell["tau_occ"], cell["k"]))
    baselines: dict[tuple[float, int], dict[str, Any]] = {}
    if args.baseline_root:
        baseline_root = Path(args.baseline_root)
        if not baseline_root.is_dir():
            raise SystemExit(f"{baseline_root}: not a directory")
        for directory in sorted(entry for entry in baseline_root.iterdir() if entry.is_dir()):
            if not GRID_PATTERN.match(directory.name):
                continue
            tau_occ, order = grid_name(directory)
            baselines[(tau_occ, order)] = sweep_grid(directory, tau_occ, order, api)
    text = _report(cells, baselines)
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
