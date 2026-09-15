#!/usr/bin/env python3
"""Audit and aggregate three Gemini repeats for raw steps 150, 200, and 250."""

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
SPLITS = {
    "gr1_env": 29,
    "gr1_object": 50,
    "gr1_behavior": 47,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def item_key(row: dict[str, str]) -> tuple[str, str]:
    return row["split"], row["index"]


def main() -> None:
    args = parse_args()
    per_repeat: dict[str, Any] = {}
    model_scores: dict[str, list[float]] = {model: [] for model in MODELS}
    model_split_scores: dict[str, dict[str, list[float]]] = {
        model: {split: [] for split in SPLITS} for model in MODELS
    }

    for repeat in (1, 2, 3):
        csv_path = args.gemini_root / f"{args.run_prefix}_repeat{repeat:02d}" / "qwen_if.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 756:
            raise ValueError(f"repeat{repeat}: expected 756 rows, got {len(rows)}")
        if len({(row["model"], *item_key(row)) for row in rows}) != 756:
            raise ValueError(f"repeat{repeat}: duplicate model/item keys")
        if any(row.get("error") for row in rows):
            raise ValueError(f"repeat{repeat}: Gemini errors remain")

        by_model: dict[str, dict[tuple[str, str], int]] = {}
        repeat_models = {}
        for model in MODELS:
            subset = [row for row in rows if row["model"] == model]
            if len(subset) != 126:
                raise ValueError(f"repeat{repeat} {model}: expected 126 rows, got {len(subset)}")
            predictions = {item_key(row): int(float(row["prediction"])) for row in subset}
            if set(predictions.values()) - {0, 1}:
                raise ValueError(f"repeat{repeat} {model}: non-binary prediction")
            positive = sum(predictions.values())
            overall = 100.0 * positive / 126
            model_scores[model].append(overall)
            split_report = {}
            for split, expected_count in SPLITS.items():
                split_rows = [row for row in subset if row["split"] == split]
                if len(split_rows) != expected_count:
                    raise ValueError(
                        f"repeat{repeat} {model} {split}: expected {expected_count}, got {len(split_rows)}"
                    )
                split_positive = sum(int(float(row["prediction"])) for row in split_rows)
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
            "csv": str(csv_path),
            "models": repeat_models,
            "paired": paired,
        }

    aggregate = {}
    for model, scores in model_scores.items():
        aggregate[model] = {
            "repeat_scores_percent": scores,
            "mean_percent": statistics.mean(scores),
            "sample_sd_percent": statistics.stdev(scores),
            "splits": {
                split: {
                    "repeat_scores_percent": split_values,
                    "mean_percent": statistics.mean(split_values),
                    "sample_sd_percent": statistics.stdev(split_values),
                }
                for split, split_values in model_split_scores[model].items()
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
            "splits": {
                split: {
                    "control_mean_percent": control["splits"][split]["mean_percent"],
                    "transport_mean_percent": transport["splits"][split]["mean_percent"],
                    "transport_minus_control_pp": (
                        transport["splits"][split]["mean_percent"]
                        - control["splits"][split]["mean_percent"]
                    ),
                }
                for split in SPLITS
            },
        }

    report = {
        "schema": "eve-cic-transport-gemini-steps150-250-three-repeat-v1",
        "protocol": {
            "model": "gemini-3.5-flash",
            "metric": "qwen_if",
            "frame_count": 49,
            "jpeg_quality": 85,
            "temperature": 0,
            "thinking_level": "low",
            "include_thoughts": False,
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
