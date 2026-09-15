#!/usr/bin/env python3
"""Audit and aggregate three Gemini repeats for the pretrained baseline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


MODEL = "pretrain_raw_s000"
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


def item_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["split"], row["index"]


def main() -> None:
    args = parse_args()
    per_repeat: dict[str, Any] = {}
    overall_scores: list[float] = []
    split_scores: dict[str, list[float]] = {split: [] for split in SPLITS}

    for repeat in (1, 2, 3):
        csv_path = (
            args.gemini_root
            / f"{args.run_prefix}_repeat{repeat:02d}"
            / "qwen_if.csv"
        )
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 126:
            raise ValueError(f"repeat{repeat}: expected 126 rows, got {len(rows)}")
        if len({item_key(row) for row in rows}) != 126:
            raise ValueError(f"repeat{repeat}: duplicate model/item keys")
        if {row["model"] for row in rows} != {MODEL}:
            raise ValueError(f"repeat{repeat}: unexpected model key")
        if any(row.get("error") for row in rows):
            raise ValueError(f"repeat{repeat}: Gemini errors remain")
        predictions = [int(float(row["prediction"])) for row in rows]
        if set(predictions) - {0, 1}:
            raise ValueError(f"repeat{repeat}: non-binary prediction")

        positive = sum(predictions)
        overall = 100.0 * positive / 126
        overall_scores.append(overall)
        repeat_splits = {}
        for split, expected_count in SPLITS.items():
            subset = [row for row in rows if row["split"] == split]
            if len(subset) != expected_count:
                raise ValueError(
                    f"repeat{repeat} {split}: expected {expected_count}, got {len(subset)}"
                )
            split_positive = sum(int(float(row["prediction"])) for row in subset)
            score = 100.0 * split_positive / expected_count
            split_scores[split].append(score)
            repeat_splits[split] = {
                "positive": split_positive,
                "count": expected_count,
                "score_percent": score,
            }
        per_repeat[f"repeat{repeat:02d}"] = {
            "csv": str(csv_path),
            "positive": positive,
            "count": 126,
            "score_percent": overall,
            "splits": repeat_splits,
        }

    report = {
        "schema": "eve-cic-transport-pretrain-gemini-three-repeat-v1",
        "protocol": {
            "model": "gemini-3.5-flash",
            "metric": "qwen_if",
            "frame_count": 49,
            "jpeg_quality": 85,
            "temperature": 0,
            "thinking_level": "low",
            "include_thoughts": False,
            "inference_seed": 4,
            "generation_steps": 30,
            "generation_frames": 93,
            "generation_resolution": "768x480",
        },
        "model": MODEL,
        "per_repeat": per_repeat,
        "aggregate": {
            "repeat_scores_percent": overall_scores,
            "mean_percent": statistics.mean(overall_scores),
            "sample_sd_percent": statistics.stdev(overall_scores),
            "splits": {
                split: {
                    "repeat_scores_percent": scores,
                    "mean_percent": statistics.mean(scores),
                    "sample_sd_percent": statistics.stdev(scores),
                }
                for split, scores in split_scores.items()
            },
        },
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
