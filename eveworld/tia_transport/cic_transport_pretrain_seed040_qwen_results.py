#!/usr/bin/env python3
"""Audit Pretrain seed040 Qwen-IF three-repeat results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (40,)
MODEL = "pretrain_raw_s000"
SPLITS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores: dict[int, list[float]] = {seed: [] for seed in SEEDS}
    split_scores = {seed: {split: [] for split in SPLITS} for seed in SEEDS}
    endpoints: set[str] = set()
    judges: set[str] = set()
    runs: dict[str, Any] = {}

    for seed in SEEDS:
        seed_runs = {}
        for repeat in range(1, 4):
            run_name = f"{MODEL}_seed{seed:03d}_repeat{repeat:02d}"
            run_dir = args.qwen_root / run_name
            config = json.loads(
                (run_dir / f"{run_name}_run_config.json").read_text(encoding="utf-8")
            )
            if config.get("disable_thinking") is not True:
                raise ValueError(f"{run_name}: thinking is not disabled")
            endpoints.add(config["qwen_base"])
            judges.add(config["qwen_model"])
            csv_path = run_dir / f"{run_name}_qwen_if.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 126 or len({row["key"] for row in rows}) != 126:
                raise ValueError(f"{run_name}: incomplete or duplicate rows")
            if {row["model"] for row in rows} != {MODEL}:
                raise ValueError(f"{run_name}: model label drift")
            if any(row.get("error") or not row.get("raw_text", "").strip() for row in rows):
                raise ValueError(f"{run_name}: error or empty response")
            predictions = [int(float(row["prediction"])) for row in rows]
            if set(predictions) - {0, 1}:
                raise ValueError(f"{run_name}: non-binary prediction")
            value = 100.0 * sum(predictions) / 126
            scores[seed].append(value)
            split_report = {}
            for split, count in SPLITS.items():
                subset = [row for row in rows if row["split"] == split]
                if len(subset) != count:
                    raise ValueError(f"{run_name}: {split} count drift")
                split_value = 100.0 * sum(
                    int(float(row["prediction"])) for row in subset
                ) / count
                split_scores[seed][split].append(split_value)
                split_report[split] = split_value
            seed_runs[f"repeat{repeat:02d}"] = {
                "score_percent": value,
                "splits_percent": split_report,
                "csv": str(csv_path),
            }
        runs[f"seed{seed:03d}"] = seed_runs

    if len(endpoints) != 1 or len(judges) != 1:
        raise ValueError("Qwen endpoint or model differs across runs")
    aggregate = {
        f"seed{seed:03d}": {
            "repeat_scores_percent": values,
            "mean_percent": statistics.mean(values),
            "sample_sd_percent": statistics.stdev(values),
            "splits": {
                split: {
                    "repeat_scores_percent": split_values,
                    "mean_percent": statistics.mean(split_values),
                    "sample_sd_percent": statistics.stdev(split_values),
                }
                for split, split_values in split_scores[seed].items()
            },
        }
        for seed, values in scores.items()
    }
    report = {
        "schema": "eve-pretrain-seed040-qwen-three-repeat-v1",
        "protocol": {
            "endpoint": next(iter(endpoints)),
            "judge_model": next(iter(judges)),
            "metric": "qwen_if",
            "inference_seeds": list(SEEDS),
            "frame_count": 49,
            "jpeg_quality": 85,
            "temperature": 0,
            "max_tokens": 32000,
            "thinking": False,
            "repeats": 3,
        },
        "runs": runs,
        "aggregate": aggregate,
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
