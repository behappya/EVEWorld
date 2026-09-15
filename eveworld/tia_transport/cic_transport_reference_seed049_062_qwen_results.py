#!/usr/bin/env python3
"""Audit Pretrain/SFT seed049/062 Qwen results and compare Transport s150."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


MODELS = ("pretrain_raw_s000", "standard_sft_raw_s150")
SEEDS = (49, 62)
SPLITS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}
TRANSPORT_MODEL = "transport_raw_s150"
DEFAULT_TRANSPORT_ROOT = Path(
    "/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1/"
    "qwen_transport_raw_s150_multiseed70_thinking_off"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport-root", type=Path, default=DEFAULT_TRANSPORT_ROOT)
    return parser.parse_args()


def read_run(csv_path: Path, expected_model: str) -> tuple[float, dict[str, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 126 or len({row["key"] for row in rows}) != 126:
        raise ValueError(f"{csv_path}: incomplete or duplicate rows")
    if {row["model"] for row in rows} != {expected_model}:
        raise ValueError(f"{csv_path}: model label drift")
    if any(row.get("error") or not row.get("raw_text", "").strip() for row in rows):
        raise ValueError(f"{csv_path}: error or empty response")
    predictions = [int(float(row["prediction"])) for row in rows]
    if set(predictions) - {0, 1}:
        raise ValueError(f"{csv_path}: non-binary prediction")
    splits: dict[str, float] = {}
    for split, count in SPLITS.items():
        subset = [row for row in rows if row["split"] == split]
        if len(subset) != count:
            raise ValueError(f"{csv_path}: {split} count drift")
        splits[split] = 100.0 * sum(
            int(float(row["prediction"])) for row in subset
        ) / count
    return 100.0 * sum(predictions) / 126, splits


def aggregate(values: list[float], split_values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "repeat_scores_percent": values,
        "mean_percent": statistics.mean(values),
        "sample_sd_percent": statistics.stdev(values),
        "splits": {
            split: {
                "repeat_scores_percent": scores,
                "mean_percent": statistics.mean(scores),
                "sample_sd_percent": statistics.stdev(scores),
            }
            for split, scores in split_values.items()
        },
    }


def main() -> None:
    args = parse_args()
    endpoints: set[str] = set()
    judges: set[str] = set()
    report_runs: dict[str, Any] = {}
    report_aggregate: dict[str, Any] = {}

    for model in MODELS:
        model_runs: dict[str, Any] = {}
        model_aggregate: dict[str, Any] = {}
        for seed in SEEDS:
            values: list[float] = []
            split_values = {split: [] for split in SPLITS}
            seed_runs: dict[str, Any] = {}
            for repeat in range(1, 4):
                run_name = f"{model}_seed{seed:03d}_repeat{repeat:02d}"
                run_dir = args.qwen_root / run_name
                config = json.loads(
                    (run_dir / f"{run_name}_run_config.json").read_text(encoding="utf-8")
                )
                if config.get("disable_thinking") is not True:
                    raise ValueError(f"{run_name}: thinking is not disabled")
                endpoints.add(config["qwen_base"])
                judges.add(config["qwen_model"])
                csv_path = run_dir / f"{run_name}_qwen_if.csv"
                score, splits = read_run(csv_path, model)
                values.append(score)
                for split, value in splits.items():
                    split_values[split].append(value)
                seed_runs[f"repeat{repeat:02d}"] = {
                    "score_percent": score,
                    "splits_percent": splits,
                    "csv": str(csv_path),
                }
            seed_name = f"seed{seed:03d}"
            model_runs[seed_name] = seed_runs
            model_aggregate[seed_name] = aggregate(values, split_values)
        report_runs[model] = model_runs
        report_aggregate[model] = model_aggregate

    if len(endpoints) != 1 or len(judges) != 1:
        raise ValueError("Qwen endpoint or model differs across reference runs")

    transport_aggregate: dict[str, Any] = {}
    for seed in SEEDS:
        values: list[float] = []
        split_values = {split: [] for split in SPLITS}
        for repeat in range(1, 4):
            run_name = f"batch01_repeat{repeat:02d}"
            csv_path = args.transport_root / run_name / f"{run_name}_qwen_if.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if f"/seed{seed:03d}/" in row["key"]
                ]
            temp_path = args.qwen_root / f".transport_seed{seed:03d}_repeat{repeat:02d}.csv"
            fields = rows[0].keys() if rows else []
            with temp_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            try:
                score, splits = read_run(temp_path, TRANSPORT_MODEL)
            finally:
                temp_path.unlink(missing_ok=True)
            values.append(score)
            for split, value in splits.items():
                split_values[split].append(value)
        transport_aggregate[f"seed{seed:03d}"] = aggregate(values, split_values)

    comparisons = {}
    for seed in SEEDS:
        seed_name = f"seed{seed:03d}"
        pretrain = report_aggregate["pretrain_raw_s000"][seed_name]["mean_percent"]
        sft = report_aggregate["standard_sft_raw_s150"][seed_name]["mean_percent"]
        transport = transport_aggregate[seed_name]["mean_percent"]
        comparisons[seed_name] = {
            "sft_minus_pretrain_pp": sft - pretrain,
            "transport_minus_pretrain_pp": transport - pretrain,
            "transport_minus_sft_pp": transport - sft,
        }

    report = {
        "schema": "eve-reference-seed049-seed062-qwen-three-repeat-v1",
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
        "runs": report_runs,
        "aggregate": report_aggregate,
        "transport_s150_aggregate": transport_aggregate,
        "matched_comparisons": comparisons,
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
