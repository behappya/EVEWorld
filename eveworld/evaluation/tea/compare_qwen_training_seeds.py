#!/usr/bin/env python3
"""Prompt-clustered Qwen statistics across training and generation seeds."""

from __future__ import annotations

import argparse
import csv
import json
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


def read_csv(path: str) -> dict[str, dict[str, str]]:
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("error") or row.get("parsed_ok") == "0":
                continue
            key = Path(row["video_path"]).name
            if key in rows:
                raise ValueError(f"duplicate video {key} in {path}")
            rows[key] = row
    return rows


def numeric(row: dict[str, str], metric: str) -> float | None:
    value = str(row.get(metric, "")).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def group_values(
    groups: list[list[dict[str, dict[str, str]]]],
    keys: list[str],
    metric: str,
) -> np.ndarray:
    output = np.empty((len(groups), len(keys)), dtype=np.float64)
    for group_index, group in enumerate(groups):
        for prompt_index, key in enumerate(keys):
            values = [numeric(rows[key], metric) for rows in group]
            if any(value is None for value in values):
                raise ValueError(f"metric {metric} missing for {key}")
            output[group_index, prompt_index] = float(np.mean(values))
    return output


def prompt_bootstrap(
    delta: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[list[float], float]:
    count = len(delta)
    boot = np.empty(samples, dtype=np.float64)
    permuted = np.empty(samples, dtype=np.float64)
    batch = 1000
    for start in range(0, samples, batch):
        stop = min(samples, start + batch)
        size = stop - start
        choices = rng.integers(0, count, size=(size, count))
        boot[start:stop] = delta[choices].mean(axis=1)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, count))
        permuted[start:stop] = (delta[None, :] * signs).mean(axis=1)
    observed = abs(float(delta.mean()))
    p_value = float((np.count_nonzero(np.abs(permuted) >= observed) + 1) / (samples + 1))
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-group", nargs="+", action="append", required=True)
    parser.add_argument("--method-group", nargs="+", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()

    baseline_groups = [[read_csv(path) for path in group] for group in args.baseline_group]
    method_groups = [[read_csv(path) for path in group] for group in args.method_group]
    all_rows = [rows for group in baseline_groups + method_groups for rows in group]
    common = sorted(set.intersection(*(set(rows) for rows in all_rows)))
    if len(common) < 2:
        raise ValueError(f"need at least two common prompts, got {len(common)}")
    if len(baseline_groups) not in (1, len(method_groups)):
        raise ValueError("baseline must have one group or match the number of method training seeds")

    rng = np.random.default_rng(args.seed)
    metrics = {}
    for metric, direction in METRICS.items():
        try:
            baseline = group_values(baseline_groups, common, metric)
            method = group_values(method_groups, common, metric)
        except ValueError:
            continue
        baseline_prompt = baseline.mean(axis=0)
        method_prompt = method.mean(axis=0)
        delta = method_prompt - baseline_prompt
        ci, p_value = prompt_bootstrap(delta, args.samples, rng)
        paired_baseline = np.repeat(baseline, len(method_groups), axis=0) if len(baseline) == 1 else baseline
        per_training_seed_delta = (method - paired_baseline).mean(axis=1)
        improvement = direction * delta
        baseline_mean = float(baseline_prompt.mean())
        metrics[metric] = {
            "higher_is_better": direction > 0,
            "baseline_mean": baseline_mean,
            "method_mean": float(method_prompt.mean()),
            "delta_method_minus_baseline": float(delta.mean()),
            "relative_improvement": (
                float(improvement.mean() / abs(baseline_mean)) if baseline_mean else 0.0
            ),
            "prompt_bootstrap_delta_ci95": ci,
            "prompt_signflip_p": p_value,
            "fraction_prompts_improved": float((improvement > 0).mean()),
            "per_training_seed_delta": per_training_seed_delta.tolist(),
            "training_seed_direction_consistency": float(
                (direction * per_training_seed_delta > 0).mean()
            ),
        }

    result = {
        "baseline_groups": args.baseline_group,
        "method_groups": args.method_group,
        "n_prompts": len(common),
        "n_baseline_training_groups": len(baseline_groups),
        "n_method_training_groups": len(method_groups),
        "generation_seeds_per_group": [len(group) for group in method_groups],
        "bootstrap_unit": "prompt",
        "bootstrap_samples": args.samples,
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
