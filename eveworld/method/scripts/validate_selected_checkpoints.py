#!/usr/bin/env python3
"""Fail closed unless a selected-checkpoint TSV has exactly the requested runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ("method", "pipeline", "training_seed", "checkpoint_step", "lora_path")
PIPELINES = {"joint_lora": "joint", "frontier_only": "frontier", "eve": "frontier"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--training-seeds", nargs="+", required=True)
    parser.add_argument("--allow-missing-paths", action="store_true")
    args = parser.parse_args()

    with args.selected.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError(f"{args.selected} must have columns: {FIELDS}")
    expected = {(method, seed) for method in args.methods for seed in args.training_seeds}
    actual = {(row["method"], row["training_seed"]) for row in rows}
    if len(actual) != len(rows):
        raise ValueError("selected checkpoint TSV contains duplicate method/training_seed rows")
    if actual != expected:
        raise ValueError(
            f"selected checkpoint matrix mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for row in rows:
        expected_pipeline = PIPELINES.get(row["method"])
        if row["pipeline"] != expected_pipeline:
            raise ValueError(f"bad pipeline for {row['method']}: {row['pipeline']}")
        if not args.allow_missing_paths and not Path(row["lora_path"]).is_dir():
            raise FileNotFoundError(row["lora_path"])
    print(f"validated {len(rows)} selected checkpoints")


if __name__ == "__main__":
    main()
