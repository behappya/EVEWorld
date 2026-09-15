#!/usr/bin/env python3
"""Audit and aggregate three Gemini repeats for the four-checkpoint campaign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


MODELS = (
    "control_raw_s050",
    "control_raw_s100",
    "transport_raw_s050",
    "transport_raw_s100",
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
        csv_path = (
            args.gemini_root
            / f"{args.run_prefix}_repeat{repeat:02d}"
            / "qwen_if.csv"
        )
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 504:
            raise ValueError(f"repeat{repeat}: expected 504 rows, got {len(rows)}")
        keys = {(row["model"], *item_key(row)) for row in rows}
        if len(keys) != 504:
            raise ValueError(f"repeat{repeat}: duplicate model/item keys")
        if any(row.get("error") for row in rows):
            raise ValueError(f"repeat{repeat}: Gemini errors remain")

        by_model: dict[str, dict[tuple[str, str], int]] = {}
        repeat_scores = {}
        for model in MODELS:
            subset = [row for row in rows if row["model"] == model]
            if len(subset) != 126:
                raise ValueError(
                    f"repeat{repeat} {model}: expected 126 rows, got {len(subset)}"
                )
            predictions = {item_key(row): int(float(row["prediction"])) for row in subset}
            if set(predictions.values()) - {0, 1}:
                raise ValueError(f"repeat{repeat} {model}: non-binary prediction")
            score = 100.0 * sum(predictions.values()) / 126
            model_scores[model].append(score)
            split_scores = {}
            for split, expected_count in SPLITS.items():
                split_rows = [row for row in subset if row["split"] == split]
                if len(split_rows) != expected_count:
                    raise ValueError(
                        f"repeat{repeat} {model} {split}: expected "
                        f"{expected_count} rows, got {len(split_rows)}"
                    )
                positive = sum(int(float(row["prediction"])) for row in split_rows)
                split_score = 100.0 * positive / expected_count
                model_split_scores[model][split].append(split_score)
                split_scores[split] = {
                    "positive": positive,
                    "count": expected_count,
                    "score_percent": split_score,
                }
            repeat_scores[model] = {
                "positive": sum(predictions.values()),
                "count": 126,
                "score_percent": score,
                "splits": split_scores,
            }
            by_model[model] = predictions

        paired = {}
        for step in (50, 100):
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
            "models": repeat_scores,
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
                    "repeat_scores_percent": split_scores,
                    "mean_percent": statistics.mean(split_scores),
                    "sample_sd_percent": statistics.stdev(split_scores),
                }
                for split, split_scores in model_split_scores[model].items()
            },
        }
    comparisons = {}
    for step in (50, 100):
        control = aggregate[f"control_raw_s{step:03d}"]["mean_percent"]
        transport = aggregate[f"transport_raw_s{step:03d}"]["mean_percent"]
        comparisons[str(step)] = {
            "control_mean_percent": control,
            "transport_mean_percent": transport,
            "transport_minus_control_pp": transport - control,
            "splits": {
                split: {
                    "control_mean_percent": aggregate[
                        f"control_raw_s{step:03d}"
                    ]["splits"][split]["mean_percent"],
                    "transport_mean_percent": aggregate[
                        f"transport_raw_s{step:03d}"
                    ]["splits"][split]["mean_percent"],
                    "transport_minus_control_pp": (
                        aggregate[f"transport_raw_s{step:03d}"]["splits"][split][
                            "mean_percent"
                        ]
                        - aggregate[f"control_raw_s{step:03d}"]["splits"][split][
                            "mean_percent"
                        ]
                    ),
                }
                for split in SPLITS
            },
        }

    report = {
        "schema": "eve-cic-transport-gemini-three-repeat-v1",
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
