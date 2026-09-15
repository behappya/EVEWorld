#!/usr/bin/env python3
"""Validate and summarize the fixed-seed EVEWorld ablation results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SPLIT_COUNTS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-models", nargs="+", required=True)
    parser.add_argument("--mlr-summary", type=Path)
    return parser.parse_args()


def score(rows: list[dict[str, str]]) -> dict[str, object]:
    positives = sum(int(float(row["prediction"])) for row in rows)
    count = len(rows)
    return {
        "positive": positives,
        "count": count,
        "score": positives / count if count else None,
        "percent": 100.0 * positives / count if count else None,
    }


def main() -> None:
    args = parse_args()
    with args.qwen_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_models = list(args.expected_models)
    expected_set = set(expected_models)
    actual_set = {row["model"] for row in rows}
    if actual_set != expected_set:
        raise ValueError(
            f"model mismatch: missing={sorted(expected_set - actual_set)} "
            f"extra={sorted(actual_set - expected_set)}"
        )
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate evaluation keys detected")
    errors = [row for row in rows if row.get("error") not in (None, "", "None")]
    if errors:
        raise ValueError(f"judge errors detected: {len(errors)}")

    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["model"]][row["split"]].append(row)

    mlr = {}
    if args.mlr_summary:
        mlr = json.loads(args.mlr_summary.read_text(encoding="utf-8")).get("summary", {})

    summary = {"protocol": "fixed inference seed004", "models": {}}
    table_rows = []
    for model in expected_models:
        by_split = grouped[model]
        for split, expected in SPLIT_COUNTS.items():
            if len(by_split[split]) != expected:
                raise ValueError(
                    f"{model}/{split}: expected {expected}, found {len(by_split[split])}"
                )
        model_rows = [row for split in SPLIT_COUNTS for row in by_split[split]]
        if len(model_rows) != 126:
            raise ValueError(f"{model}: expected 126 rows, found {len(model_rows)}")
        split_scores = {split: score(by_split[split]) for split in SPLIT_COUNTS}
        overall = score(model_rows)
        mlr_row = mlr.get(model)
        summary["models"][model] = {
            "overall": overall,
            "splits": split_scores,
            "mlr": mlr_row,
        }
        table_rows.append(
            {
                "model": model,
                "qwen_if": overall["percent"],
                "env": split_scores["gr1_env"]["percent"],
                "object": split_scores["gr1_object"]["percent"],
                "behavior": split_scores["gr1_behavior"]["percent"],
                "mlr": 100.0 * mlr_row["dup_rate"] if mlr_row else None,
                "mlr_events": (
                    round(mlr_row["dup_rate"] * (mlr_row["n"] - mlr_row["undetectable"]))
                    if mlr_row
                    else None
                ),
                "mlr_eligible": mlr_row["n"] - mlr_row["undetectable"] if mlr_row else None,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
