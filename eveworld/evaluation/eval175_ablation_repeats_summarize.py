#!/usr/bin/env python3
"""Audit and summarize repeated fixed-seed Gemini ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SPLIT_COUNTS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-models", nargs="+", required=True)
    return parser.parse_args()


def percent(rows: list[dict[str, str]]) -> float:
    return 100.0 * sum(int(row["prediction"]) for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    expected_models = list(args.expected_models)
    expected_set = set(expected_models)
    run_results: list[dict[str, object]] = []

    for run_dir in args.run_dir:
        csv_path = run_dir / "qwen_if.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 126 * len(expected_models):
            raise ValueError(f"{csv_path}: unexpected row count {len(rows)}")
        keys = [row["key"] for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{csv_path}: duplicate keys")
        if {row["model"] for row in rows} != expected_set:
            raise ValueError(f"{csv_path}: model set mismatch")
        errors = [row for row in rows if row.get("error")]
        if errors:
            raise ValueError(f"{csv_path}: {len(errors)} judge errors")
        if any(row["prediction"] not in {"0", "1"} for row in rows):
            raise ValueError(f"{csv_path}: non-binary prediction")

        models: dict[str, object] = {}
        for model in expected_models:
            model_rows = [row for row in rows if row["model"] == model]
            by_split = defaultdict(list)
            for row in model_rows:
                by_split[row["split"]].append(row)
            for split, expected in SPLIT_COUNTS.items():
                if len(by_split[split]) != expected:
                    raise ValueError(
                        f"{csv_path}: {model}/{split} has {len(by_split[split])}, "
                        f"expected {expected}"
                    )
            models[model] = {
                "overall": percent(model_rows),
                "splits": {split: percent(by_split[split]) for split in SPLIT_COUNTS},
            }
        run_results.append(
            {"run": run_dir.name, "csv": str(csv_path.resolve()), "models": models}
        )

    aggregate: dict[str, object] = {}
    table_rows = []
    for model in expected_models:
        overall = [run["models"][model]["overall"] for run in run_results]
        split_values = {
            split: [run["models"][model]["splits"][split] for run in run_results]
            for split in SPLIT_COUNTS
        }
        aggregate[model] = {
            "overall_mean": statistics.mean(overall),
            "overall_sample_sd": statistics.stdev(overall) if len(overall) > 1 else None,
            "split_means": {
                split: statistics.mean(values) for split, values in split_values.items()
            },
            "split_sample_sd": {
                split: statistics.stdev(values) if len(values) > 1 else None
                for split, values in split_values.items()
            },
        }
        table_rows.append(
            {
                "model": model,
                **{f"run_{index + 1}": value for index, value in enumerate(overall)},
                "mean": statistics.mean(overall),
                "sample_sd": statistics.stdev(overall) if len(overall) > 1 else None,
                "env_mean": statistics.mean(split_values["gr1_env"]),
                "object_mean": statistics.mean(split_values["gr1_object"]),
                "behavior_mean": statistics.mean(split_values["gr1_behavior"]),
            }
        )

    summary = {
        "protocol": "fixed inference seed004; Gemini DreamGenBench Qwen-IF",
        "run_count": len(run_results),
        "expected_models": expected_models,
        "runs": run_results,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
