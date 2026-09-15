#!/usr/bin/env python3
"""Audit three thinking-off Qwen-IF repeats for raw steps 150, 200, and 250."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


STEPS = (150, 200, 250)
MODELS = tuple(
    f"{variant}_raw_s{step:03d}"
    for step in STEPS
    for variant in ("control", "transport")
)
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
    model_scores: dict[str, list[float]] = {model: [] for model in MODELS}
    model_split_scores = {
        model: {split: [] for split in SPLITS} for model in MODELS
    }
    resolved_models: set[str] = set()
    endpoints: set[str] = set()

    for repeat in (1, 2, 3):
        repeat_models = {}
        by_model: dict[str, dict[tuple[str, str], int]] = {}
        for model in MODELS:
            run_name = f"{model}_repeat{repeat:02d}"
            run_dir = args.qwen_root / run_name
            csv_path = run_dir / f"{run_name}_qwen_if.csv"
            config_path = run_dir / f"{run_name}_run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("disable_thinking") is not True:
                raise ValueError(f"{run_name}: thinking is not explicitly disabled")
            resolved_models.add(config["qwen_model"])
            endpoints.add(config["qwen_base"])

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 126:
                raise ValueError(f"{run_name}: expected 126 rows, got {len(rows)}")
            if len({item_key(row) for row in rows}) != 126:
                raise ValueError(f"{run_name}: duplicate item keys")
            if {row["model"] for row in rows} != {model}:
                raise ValueError(f"{run_name}: unexpected model key")
            if any(row.get("error") for row in rows):
                raise ValueError(f"{run_name}: Qwen errors remain")
            predictions = {item_key(row): int(float(row["prediction"])) for row in rows}
            if set(predictions.values()) - {0, 1}:
                raise ValueError(f"{run_name}: non-binary prediction")

            positive = sum(predictions.values())
            overall = 100.0 * positive / 126
            model_scores[model].append(overall)
            split_report = {}
            for split, expected_count in SPLITS.items():
                subset = [row for row in rows if row["split"] == split]
                if len(subset) != expected_count:
                    raise ValueError(
                        f"{run_name} {split}: expected {expected_count}, got {len(subset)}"
                    )
                split_positive = sum(int(float(row["prediction"])) for row in subset)
                score = 100.0 * split_positive / expected_count
                model_split_scores[model][split].append(score)
                split_report[split] = {
                    "positive": split_positive,
                    "count": expected_count,
                    "score_percent": score,
                }
            repeat_models[model] = {
                "positive": positive,
                "count": 126,
                "score_percent": overall,
                "splits": split_report,
                "csv": str(csv_path),
            }
            by_model[model] = predictions

        paired = {}
        for step in STEPS:
            control = by_model[f"control_raw_s{step:03d}"]
            transport = by_model[f"transport_raw_s{step:03d}"]
            wins = sum(transport[key] > control[key] for key in control)
            losses = sum(transport[key] < control[key] for key in control)
            paired[str(step)] = {
                "transport_wins": wins,
                "transport_losses": losses,
                "ties": 126 - wins - losses,
                "delta_positive": wins - losses,
            }
        per_repeat[f"repeat{repeat:02d}"] = {
            "models": repeat_models,
            "paired": paired,
        }

    if len(resolved_models) != 1 or len(endpoints) != 1:
        raise ValueError("Qwen model or endpoint differs across runs")
    aggregate = {}
    for model, scores in model_scores.items():
        aggregate[model] = {
            "repeat_scores_percent": scores,
            "mean_percent": statistics.mean(scores),
            "sample_sd_percent": statistics.stdev(scores),
            "splits": {
                split: {
                    "repeat_scores_percent": values,
                    "mean_percent": statistics.mean(values),
                    "sample_sd_percent": statistics.stdev(values),
                }
                for split, values in model_split_scores[model].items()
            },
        }
    comparisons = {}
    for step in STEPS:
        control = aggregate[f"control_raw_s{step:03d}"]
        transport = aggregate[f"transport_raw_s{step:03d}"]
        comparisons[str(step)] = {
            "control_mean_percent": control["mean_percent"],
            "transport_mean_percent": transport["mean_percent"],
            "transport_minus_control_pp": transport["mean_percent"] - control["mean_percent"],
        }

    report = {
        "schema": "eve-cic-transport-qwen-thinking-off-steps150-250-three-repeat-v1",
        "protocol": {
            "endpoint": next(iter(endpoints)),
            "model": next(iter(resolved_models)),
            "metric": "qwen_if",
            "frame_count": 49,
            "jpeg_quality": 85,
            "temperature": 0,
            "max_tokens": 32000,
            "thinking": False,
            "inference_seed": 4,
        },
        "per_repeat": per_repeat,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
