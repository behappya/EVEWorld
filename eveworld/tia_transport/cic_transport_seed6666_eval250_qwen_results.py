#!/usr/bin/env python3
"""Audit three Qwen-IF repeats for Transport seed6666 steps 150--250."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


STEPS = (150, 200, 250)
MODELS = tuple(f"transport_seed6666_raw_s{step:03d}" for step in STEPS)
SPLITS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def item_key(row: dict[str, str]) -> tuple[str, str]:
    return row["split"], row["index"]


def main() -> None:
    args = parse_args()
    per_repeat: dict[str, Any] = {}
    scores: dict[str, list[float]] = {model: [] for model in MODELS}
    split_scores = {
        model: {split: [] for split in SPLITS} for model in MODELS
    }
    endpoints: set[str] = set()
    judge_models: set[str] = set()

    for repeat in range(1, 4):
        repeat_models = {}
        for model in MODELS:
            run_name = f"{model}_repeat{repeat:02d}"
            run_dir = args.qwen_root / run_name
            csv_path = run_dir / f"{run_name}_qwen_if.csv"
            config_path = run_dir / f"{run_name}_run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("disable_thinking") is not True:
                raise ValueError(f"{run_name}: thinking is not explicitly disabled")
            if config.get("frame_count") != 49 or config.get("jpeg_quality") != 85:
                raise ValueError(f"{run_name}: frame protocol drift")
            endpoints.add(config["qwen_base"])
            judge_models.add(config["qwen_model"])

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 126:
                raise ValueError(f"{run_name}: expected 126 rows, got {len(rows)}")
            if len({item_key(row) for row in rows}) != 126:
                raise ValueError(f"{run_name}: duplicate item keys")
            if {row["model"] for row in rows} != {model}:
                raise ValueError(f"{run_name}: unexpected model label")
            if any(row.get("error") for row in rows):
                raise ValueError(f"{run_name}: Qwen errors remain")
            if any(not row.get("raw_text", "").strip() for row in rows):
                raise ValueError(f"{run_name}: empty response")
            predictions = [int(float(row["prediction"])) for row in rows]
            if set(predictions) - {0, 1}:
                raise ValueError(f"{run_name}: non-binary prediction")

            positive = sum(predictions)
            overall = 100.0 * positive / 126
            scores[model].append(overall)
            split_report = {}
            for split, count in SPLITS.items():
                subset = [row for row in rows if row["split"] == split]
                if len(subset) != count:
                    raise ValueError(
                        f"{run_name} {split}: expected {count}, got {len(subset)}"
                    )
                value = 100.0 * sum(
                    int(float(row["prediction"])) for row in subset
                ) / count
                split_scores[model][split].append(value)
                split_report[split] = value
            repeat_models[model] = {
                "positive": positive,
                "score_percent": overall,
                "splits_percent": split_report,
                "csv": str(csv_path),
            }
        per_repeat[f"repeat{repeat:02d}"] = repeat_models

    if len(endpoints) != 1 or len(judge_models) != 1:
        raise ValueError("Qwen endpoint or model differs across runs")
    aggregate = {
        model: {
            "repeat_scores_percent": values,
            "mean_percent": statistics.mean(values),
            "sample_sd_percent": statistics.stdev(values),
            "splits": {
                split: {
                    "repeat_scores_percent": split_values,
                    "mean_percent": statistics.mean(split_values),
                    "sample_sd_percent": statistics.stdev(split_values),
                }
                for split, split_values in split_scores[model].items()
            },
        }
        for model, values in scores.items()
    }
    ranking = sorted(
        (
            {
                "step": step,
                "model": model,
                "mean_percent": aggregate[model]["mean_percent"],
                "sample_sd_percent": aggregate[model]["sample_sd_percent"],
            }
            for step, model in zip(STEPS, MODELS, strict=True)
        ),
        key=lambda row: (-row["mean_percent"], row["sample_sd_percent"]),
    )
    report = {
        "schema": "eve-cic-transport-seed6666-qwen-thinking-off-three-repeat-v1",
        "protocol": {
            "endpoint": next(iter(endpoints)),
            "judge_model": next(iter(judge_models)),
            "metric": "qwen_if",
            "training_seed": 6666,
            "inference_seed": 4,
            "frame_count": 49,
            "jpeg_quality": 85,
            "temperature": 0,
            "max_tokens": 32000,
            "thinking": False,
            "repeats": 3,
        },
        "per_repeat": per_repeat,
        "aggregate": aggregate,
        "ranking": ranking,
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
