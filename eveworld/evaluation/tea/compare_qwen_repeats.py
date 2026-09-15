#!/usr/bin/env python3
"""Paired Qwen statistics after averaging repeated judge measurements."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from eveworld.evaluation.tea.compare_qwen import METRICS, paired_summary
except ModuleNotFoundError:
    from compare_qwen import METRICS, paired_summary


SEED_PATTERN = re.compile(r"seed(\d+)")


def numeric(row: dict[str, str], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_repeats(paths: list[str]) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    rejected = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("error") or row.get("parsed_ok") == "0":
                    rejected.append(row.get("video_path", ""))
                    continue
                video_path = Path(row["video_path"])
                seed_match = SEED_PATTERN.search(str(video_path)) or SEED_PATTERN.search(path)
                if seed_match is None:
                    raise ValueError(f"cannot infer generation seed for {video_path}")
                key = f"seed{seed_match.group(1)}:{video_path.name}"
                grouped[key].append(row)
    return dict(grouped), rejected


def averaged_metric(rows: list[dict[str, str]], key: str) -> float | None:
    values = [numeric(row, key) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--method", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()

    baseline_rows, baseline_rejected = load_repeats(args.baseline)
    method_rows, method_rejected = load_repeats(args.method)
    common = sorted(set(baseline_rows) & set(method_rows))
    if len(common) < 2:
        raise ValueError(f"need at least 2 paired videos, got {len(common)}")

    repeat_counts = {
        "baseline": {key: len(rows) for key, rows in baseline_rows.items()},
        "method": {key: len(rows) for key, rows in method_rows.items()},
    }
    wrong_counts = {
        family: {key: count for key, count in counts.items() if count != args.expected_repeats}
        for family, counts in repeat_counts.items()
    }
    if any(wrong_counts.values()):
        raise ValueError(f"unexpected repeat counts: {wrong_counts}")

    rng = np.random.default_rng(args.seed)
    metrics = {}
    for key, direction in METRICS.items():
        pairs = [
            (averaged_metric(baseline_rows[name], key), averaged_metric(method_rows[name], key))
            for name in common
        ]
        pairs = [(baseline, method) for baseline, method in pairs if baseline is not None and method is not None]
        if len(pairs) < 2:
            continue
        baseline = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        method = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        metrics[key] = paired_summary(baseline, method, direction, rng, args.samples)

    result = {
        "baseline": args.baseline,
        "method": args.method,
        "n_independent_paired_videos": len(common),
        "repeats_per_video": args.expected_repeats,
        "missing_from_baseline": sorted(set(method_rows) - set(baseline_rows)),
        "missing_from_method": sorted(set(baseline_rows) - set(method_rows)),
        "rejected_baseline_rows": baseline_rejected,
        "rejected_method_rows": method_rejected,
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "metrics": metrics,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
