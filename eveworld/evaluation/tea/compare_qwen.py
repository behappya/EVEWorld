#!/usr/bin/env python3
"""Paired statistics for control/method Qwen laziness CSV files."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = {
    "laziness_severity": -1,
    "any_shortcut": -1,
    "process_completeness": 1,
    "q1": -1,
    "q2": -1,
    "q3": -1,
    "q4": -1,
    "q5": -1,
}


def load_rows(paths: list[str]) -> dict[str, dict[str, str]]:
    rows = {}
    for group, path in enumerate(paths):
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("error") or row.get("parsed_ok") == "0":
                    continue
                video = Path(row["video_path"]).name
                key = f"{group}:{video}"
                if key in rows:
                    raise ValueError(f"duplicate paired key {key} in {path}")
                rows[key] = row
    return rows


def numeric(row: dict[str, str], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def exact_sign_p(improved: int, worsened: int) -> float | None:
    n = improved + worsened
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(improved, worsened) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_summary(
    baseline: np.ndarray,
    method: np.ndarray,
    direction: int,
    rng: np.random.Generator,
    samples: int,
) -> dict:
    delta = method - baseline
    improvement = direction * delta
    n = len(delta)
    choices = rng.integers(0, n, size=(samples, n))
    boot_delta = delta[choices].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, n))
    permuted = (delta[None, :] * signs).mean(axis=1)
    observed = abs(float(delta.mean()))
    p_value = float((np.count_nonzero(np.abs(permuted) >= observed) + 1) / (samples + 1))
    improved = int(np.count_nonzero(improvement > 0))
    worsened = int(np.count_nonzero(improvement < 0))
    baseline_mean = float(baseline.mean())
    method_mean = float(method.mean())
    result = {
        "n_paired": n,
        "higher_is_better": direction > 0,
        "baseline_mean": baseline_mean,
        "method_mean": method_mean,
        "delta_method_minus_baseline": float(delta.mean()),
        "relative_improvement": (
            float(improvement.mean() / abs(baseline_mean)) if baseline_mean != 0 else 0.0
        ),
        "bootstrap_delta_ci95": [
            float(np.quantile(boot_delta, 0.025)),
            float(np.quantile(boot_delta, 0.975)),
        ],
        "fraction_improved": improved / n,
        "fraction_worsened": worsened / n,
        "paired_signflip_p": p_value,
    }
    values = set(np.unique(np.concatenate([baseline, method])).tolist())
    if values <= {0.0, 1.0}:
        result["discordant_improved"] = improved
        result["discordant_worsened"] = worsened
        result["exact_mcnemar_p"] = exact_sign_p(improved, worsened)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--method", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=6666)
    args = ap.parse_args()

    if len(args.baseline) != len(args.method):
        raise ValueError("baseline/method CSV group counts must match")
    baseline_rows = load_rows(args.baseline)
    method_rows = load_rows(args.method)
    common = sorted(set(baseline_rows) & set(method_rows))
    if len(common) < 2:
        raise ValueError(f"need at least 2 paired rows, got {len(common)}")

    rng = np.random.default_rng(args.seed)
    metrics = {}
    for key, direction in METRICS.items():
        pairs = [
            (numeric(baseline_rows[name], key), numeric(method_rows[name], key))
            for name in common
        ]
        pairs = [(base, method) for base, method in pairs if base is not None and method is not None]
        if len(pairs) < 2:
            continue
        baseline = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        method = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        metrics[key] = paired_summary(baseline, method, direction, rng, args.samples)

    result = {
        "baseline": args.baseline,
        "method": args.method,
        "n_common": len(common),
        "missing_from_baseline": sorted(set(method_rows) - set(baseline_rows)),
        "missing_from_method": sorted(set(baseline_rows) - set(method_rows)),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
