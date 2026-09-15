#!/usr/bin/env python3
"""Validate and aggregate repeated Gemini-IF CFG sweep runs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SPLIT_COUNTS = {"env": 29, "object": 50, "behavior": 47}


def score(rows: list[dict[str, str]]) -> float:
    return 100.0 * sum(int(row["prediction"]) for row in rows) / len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    expected_models = [
        f"step_{step:03d}_{tag}"
        for step in (50, 100, 150, 200, 250, 300)
        for tag in ("cfg_1", "cfg_2p5", "cfg_5", "cfg_7p5")
    ]
    expected_set = set(expected_models)
    runs: list[dict[str, object]] = []
    for run_dir in args.run_dir:
        path = run_dir / "qwen_if.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 3024:
            raise ValueError(f"{path}: expected 3024 rows, got {len(rows)}")
        keys = [row["key"] for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{path}: duplicate keys")
        if {row["model"] for row in rows} != expected_set:
            raise ValueError(f"{path}: model set mismatch")
        if any(row.get("error") for row in rows):
            raise ValueError(f"{path}: contains judge errors")
        if any(row.get("prediction") not in {"0", "1"} for row in rows):
            raise ValueError(f"{path}: non-binary prediction")
        models: dict[str, object] = {}
        for model in expected_models:
            model_rows = [row for row in rows if row["model"] == model]
            by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in model_rows:
                by_split[row["split"]].append(row)
            for split, count in SPLIT_COUNTS.items():
                if len(by_split[split]) != count:
                    raise ValueError(
                        f"{path}: {model}/{split} has {len(by_split[split])}, expected {count}"
                    )
            models[model] = {
                "overall": score(model_rows),
                "splits": {split: score(by_split[split]) for split in SPLIT_COUNTS},
            }
        runs.append({"run": run_dir.name, "csv": str(path.resolve()), "models": models})

    aggregate: dict[str, object] = {}
    table_rows: list[dict[str, object]] = []
    for model in expected_models:
        overall = [float(run["models"][model]["overall"]) for run in runs]
        split_values = {
            split: [float(run["models"][model]["splits"][split]) for run in runs]
            for split in SPLIT_COUNTS
        }
        aggregate[model] = {
            "overall_mean": statistics.mean(overall),
            "overall_sample_sd": statistics.stdev(overall) if len(overall) > 1 else None,
            "split_means": {split: statistics.mean(values) for split, values in split_values.items()},
            "split_sample_sd": {
                split: statistics.stdev(values) if len(values) > 1 else None
                for split, values in split_values.items()
            },
        }
        table_rows.append({
            "model": model,
            **{f"run_{i + 1}_overall": value for i, value in enumerate(overall)},
            "overall_mean": statistics.mean(overall),
            "overall_sample_sd": statistics.stdev(overall) if len(overall) > 1 else None,
            **{f"{split}_mean": statistics.mean(values) for split, values in split_values.items()},
            **{f"{split}_sample_sd": statistics.stdev(values) if len(values) > 1 else None
               for split, values in split_values.items()},
        })

    summary = {
        "protocol": "Gemini-3.6-Flash backend; DreamGenBench Qwen-IF; 49 JPEG frames; temperature 0; low thinking",
        "run_count": len(runs),
        "records_per_run": 3024,
        "models": expected_models,
        "runs": runs,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps({"run_count": len(runs), "models": len(expected_models), "records_per_run": 3024}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
