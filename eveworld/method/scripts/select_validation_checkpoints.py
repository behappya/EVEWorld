#!/usr/bin/env python3
"""Select one checkpoint per method/training seed from frozen validation scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ("method", "pipeline", "training_seed", "checkpoint_step", "lora_path")


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError(f"{path} must have columns: {FIELDS}")
    return rows


def read_scores(path: Path) -> list[tuple[float, float]]:
    scores = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("error") or row.get("parsed_ok") == "0":
                continue
            scores.append((float(row["laziness_severity"]), float(row["process_completeness"])))
    if not scores:
        raise ValueError(f"no valid Judge B rows in {path}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--generation-seeds", nargs="+", required=True)
    parser.add_argument("--expected-per-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidates = read_candidates(args.candidates)
    audit_rows = []
    for row in candidates:
        all_scores = []
        for generation_seed in args.generation_seeds:
            name = (
                f"{row['method']}_train{row['training_seed']}_step{row['checkpoint_step']}"
                f"_gen{generation_seed}_B_laziness.csv"
            )
            scores = read_scores(args.score_root / name)
            if len(scores) != args.expected_per_seed:
                raise ValueError(
                    f"{name}: got {len(scores)} valid rows, expected {args.expected_per_seed}"
                )
            all_scores.extend(scores)
        severity = sum(value[0] for value in all_scores) / len(all_scores)
        completeness = sum(value[1] for value in all_scores) / len(all_scores)
        audit_rows.append(
            {
                **row,
                "n": len(all_scores),
                "mean_severity": severity,
                "mean_completeness": completeness,
            }
        )

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in audit_rows:
        grouped.setdefault((row["method"], row["training_seed"]), []).append(row)
    selected = []
    for key, rows in sorted(grouped.items()):
        # Pre-registered rule: severity first, completeness second, earlier step third.
        selected.append(
            min(
                rows,
                key=lambda row: (
                    row["mean_severity"],
                    -row["mean_completeness"],
                    int(row["checkpoint_step"]),
                ),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in selected)
    audit_path = args.out.with_name("validation_checkpoint_scores.json")
    audit_path.write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
