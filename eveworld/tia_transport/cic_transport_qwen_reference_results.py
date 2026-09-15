#!/usr/bin/env python3
"""Audit three thinking-off Qwen-IF repeats for Pretrain and Standard SFT."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


REFERENCES = {
    "pretrain_raw_s000": "pretrain_raw_s000",
    "standard_sft_seed004": "seed004",
}
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
    scores: dict[str, list[float]] = {label: [] for label in REFERENCES}
    split_scores = {
        label: {split: [] for split in SPLITS} for label in REFERENCES
    }
    endpoints: set[str] = set()
    qwen_models: set[str] = set()

    for repeat in (1, 2, 3):
        repeat_report = {}
        for label, expected_model in REFERENCES.items():
            run_name = f"{label}_repeat{repeat:02d}"
            run_dir = args.qwen_root / run_name
            csv_path = run_dir / f"{run_name}_qwen_if.csv"
            config_path = run_dir / f"{run_name}_run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("disable_thinking") is not True:
                raise ValueError(f"{run_name}: thinking is not disabled")
            endpoints.add(config["qwen_base"])
            qwen_models.add(config["qwen_model"])

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 126 or len({item_key(row) for row in rows}) != 126:
                raise ValueError(f"{run_name}: expected 126 unique rows")
            if {row["model"] for row in rows} != {expected_model}:
                raise ValueError(f"{run_name}: unexpected manifest model key")
            if any(row.get("error") for row in rows):
                raise ValueError(f"{run_name}: Qwen errors remain")
            predictions = [int(float(row["prediction"])) for row in rows]
            if set(predictions) - {0, 1}:
                raise ValueError(f"{run_name}: non-binary prediction")

            positive = sum(predictions)
            overall = 100.0 * positive / 126
            scores[label].append(overall)
            split_report = {}
            for split, expected_count in SPLITS.items():
                subset = [row for row in rows if row["split"] == split]
                if len(subset) != expected_count:
                    raise ValueError(f"{run_name} {split}: wrong row count")
                split_positive = sum(int(float(row["prediction"])) for row in subset)
                score = 100.0 * split_positive / expected_count
                split_scores[label][split].append(score)
                split_report[split] = {
                    "positive": split_positive,
                    "count": expected_count,
                    "score_percent": score,
                }
            repeat_report[label] = {
                "positive": positive,
                "count": 126,
                "score_percent": overall,
                "splits": split_report,
                "csv": str(csv_path),
            }
        per_repeat[f"repeat{repeat:02d}"] = repeat_report

    if len(endpoints) != 1 or len(qwen_models) != 1:
        raise ValueError("Qwen endpoint or model differs across reference runs")
    aggregate = {
        label: {
            "repeat_scores_percent": values,
            "mean_percent": statistics.mean(values),
            "sample_sd_percent": statistics.stdev(values),
            "splits": {
                split: {
                    "repeat_scores_percent": split_values,
                    "mean_percent": statistics.mean(split_values),
                    "sample_sd_percent": statistics.stdev(split_values),
                }
                for split, split_values in split_scores[label].items()
            },
        }
        for label, values in scores.items()
    }
    pretrain = aggregate["pretrain_raw_s000"]["mean_percent"]
    sft = aggregate["standard_sft_seed004"]["mean_percent"]
    report = {
        "schema": "eve-cic-transport-qwen-thinking-off-references-three-repeat-v1",
        "protocol": {
            "endpoint": next(iter(endpoints)),
            "model": next(iter(qwen_models)),
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
        "standard_sft_minus_pretrain_pp": sft - pretrain,
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
