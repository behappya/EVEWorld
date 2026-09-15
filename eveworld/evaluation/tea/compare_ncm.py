#!/usr/bin/env python3
"""Paired statistical comparison for two ncm.py score JSON files."""

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = ("LAZY", "TELE", "JUMP", "STILL", "ROUGH")


def paired_summary(
    baseline: np.ndarray,
    method: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float | list[float]]:
    delta = method - baseline
    n = len(delta)
    choices = rng.integers(0, n, size=(samples, n))
    boot = delta[choices].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, n))
    permuted = (delta[None, :] * signs).mean(axis=1)
    observed = abs(float(delta.mean()))
    p_value = float((np.count_nonzero(np.abs(permuted) >= observed) + 1) / (samples + 1))
    baseline_mean = float(baseline.mean())
    method_mean = float(method.mean())
    return {
        "baseline_mean": baseline_mean,
        "method_mean": method_mean,
        "delta_method_minus_baseline": float(delta.mean()),
        "relative_reduction": float(-delta.mean() / baseline_mean) if baseline_mean != 0 else 0.0,
        "bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "fraction_improved": float((delta < 0).mean()),
        "paired_signflip_p": p_value,
    }


def load_rows(paths: list[str]) -> dict[str, dict]:
    rows = {}
    for group, path in enumerate(paths):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in blob["per_video"]:
            key = f"{group}:{row['video']}"
            if key in rows:
                raise ValueError(f"duplicate paired key {key} in {path}")
            rows[key] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--method", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=6666)
    args = ap.parse_args()

    if len(args.baseline) != len(args.method):
        raise ValueError("baseline/method JSON group counts must match")
    baseline_rows = load_rows(args.baseline)
    method_rows = load_rows(args.method)
    common = sorted(set(baseline_rows) & set(method_rows))
    if len(common) < 2:
        raise ValueError(f"need at least 2 paired videos, got {len(common)}")
    missing_baseline = sorted(set(method_rows) - set(baseline_rows))
    missing_method = sorted(set(baseline_rows) - set(method_rows))

    rng = np.random.default_rng(args.seed)
    metrics = {}
    for key in KEYS:
        baseline = np.asarray([baseline_rows[name][key] for name in common], dtype=np.float64)
        method = np.asarray([method_rows[name][key] for name in common], dtype=np.float64)
        metrics[key] = paired_summary(baseline, method, rng, args.samples)

    result = {
        "baseline": args.baseline,
        "method": args.method,
        "n_paired": len(common),
        "missing_from_baseline": missing_baseline,
        "missing_from_method": missing_method,
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "metrics": metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
