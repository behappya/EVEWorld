#!/usr/bin/env python3
"""Summarise the seed-repeat replays of the checkpoint-stability study.

Generation is stochastic and so is the instruction-following judge, so the
reported numbers are means over repeats. This script reads the per-seed replay
records of one or more checkpoints and prints, for every step and metric, the
mean, the sample standard deviation and the normal-approximation 95%
confidence interval of the mean. Everything is computed from the recorded
numbers, so only the standard library is needed.

Each checkpoint has its own directory named after the training step, holding
one JSON file per inference seed:

```text
<root>/250/seed004.json
{"seed": 4, "mlr": 2.5, "qwen_if": 60.1, "gemini_if": 58.4}
```

``seed`` is required and must agree with the file name; at least one of the
requested metrics must be present. A metric that is missing or has a null
value in one record leaves that seed out of the mean for that metric alone.
The record layout mirrors the step replay of the checkpoint ablation, whose
checkpoints at steps 150 and 250 are replayed over seeds 1 to 70 at 30
inference steps.

Run it over the replay output:

```bash
python experiments/analysis/seed_stability/aggregate.py \
    --root outputs/seed_sweep --steps 150 250
```
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Sequence

__all__ = ["main", "load_seed", "step_name", "summarise"]

STEP_PATTERN = re.compile(r"^(?:step[-_]?)?(\d+)$")
SEED_PATTERN = re.compile(r"^seed(\d+)\.json$")
DEFAULT_METRICS = ("gemini_if", "qwen_if", "mlr")
Z_95 = 1.96


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="replay root with one subdirectory per checkpoint step")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="optional subset of steps to summarise; every other step is skipped",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="metric keys to read from every seed record (default: %(default)s)",
    )
    parser.add_argument("--out", default=None, help="optional file for the markdown report")
    return parser.parse_args(argv)


def step_name(directory: Path) -> int:
    """Read the checkpoint step out of a directory name."""
    match = STEP_PATTERN.match(directory.name)
    if match is None:
        raise SystemExit(f"{directory}: expected a directory name like 250 or step_250")
    return int(match.group(1))


def load_seed(path: Path, metrics: Sequence[str]) -> dict[str, float]:
    """Read one seed record and return its numeric metric values."""
    match = SEED_PATTERN.match(path.name)
    if match is None:
        raise SystemExit(f"{path}: expected a file name like seed004.json")
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise SystemExit(f"{path}: expected a JSON object, got {type(record).__name__}")
    if "seed" not in record:
        raise SystemExit(f"{path}: missing field 'seed'")
    seed = record["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SystemExit(f"{path}: seed must be a non-negative integer, got {seed!r}")
    if seed != int(match.group(1)):
        raise SystemExit(f"{path}: seed {seed} does not match the file name seed {int(match.group(1))}")
    values: dict[str, float] = {}
    for metric in metrics:
        value = record.get(metric)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f"{path}: {metric} must be a finite number, got {value!r}")
        values[metric] = float(value)
    if not values:
        wanted = ", ".join(metrics)
        raise SystemExit(f"{path}: none of the requested metrics ({wanted}) is present")
    return values


def summarise(values: Sequence[float]) -> tuple[float, float | None, float | None]:
    """Return mean, sample SD and the 95% CI half-width of a sample."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None, None
    deviation = statistics.stdev(values)
    half_width = Z_95 * deviation / math.sqrt(len(values))
    return mean, deviation, half_width


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(steps: Sequence[int], rows: Sequence[tuple[int, str, list[float]]]) -> str:
    lines = ["| Step | Metric | n | Mean | SD | 95% CI |"]
    lines.append("|---:|---|---:|---:|---:|---:|")
    for step, metric, values in rows:
        if not values:
            lines.append(f"| {step} | {metric} | 0 | n/a | n/a | n/a |")
            continue
        mean, deviation, half_width = summarise(values)
        lines.append(f"| {step} | {metric} | {len(values)} | {_fmt(mean)} " f"| {_fmt(deviation)} | {_fmt(half_width)} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"{root}: not a directory")
    directories = sorted(entry for entry in root.iterdir() if entry.is_dir())
    steps: dict[int, Path] = {}
    for directory in directories:
        step = step_name(directory)
        if step in steps:
            raise SystemExit(f"{directory}: step {step} is already provided by {steps[step]}")
        steps[step] = directory
    if not steps:
        raise SystemExit(f"{root}: no checkpoint-step subdirectories found")
    if args.steps is not None:
        missing = sorted(set(args.steps) - set(steps))
        if missing:
            raise SystemExit(f"{root}: step {missing[0]} not found")
        selected = sorted(set(args.steps))
    else:
        selected = sorted(steps)
    rows: list[tuple[int, str, list[float]]] = []
    seeds_per_step: list[str] = []
    for step in selected:
        directory = steps[step]
        by_metric: dict[str, list[float]] = {metric: [] for metric in args.metrics}
        seen: dict[int, Path] = {}
        count = 0
        for path in sorted(directory.glob("*.json")):
            values = load_seed(path, args.metrics)
            count += 1
            seed_key = int(SEED_PATTERN.match(path.name).group(1))
            if seed_key in seen:
                raise SystemExit(f"{path}: duplicate seed {seed_key}, already read from {seen[seed_key]}")
            seen[seed_key] = path
            for metric, value in values.items():
                by_metric[metric].append(value)
        if count == 0:
            raise SystemExit(f"{directory}: no seed*.json files found")
        seeds_per_step.append(f"{step} -> {count}")
        for metric in args.metrics:
            rows.append((step, metric, by_metric[metric]))
    lines = [f"# Seed-repeat summary: {root}", "", _table(selected, rows), ""]
    lines.append(f"steps: {', '.join(str(step) for step in selected)}")
    lines.append("seeds per step: " + ", ".join(seeds_per_step))
    lines.append("metrics: " + ", ".join(args.metrics))
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
