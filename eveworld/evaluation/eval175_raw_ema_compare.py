#!/usr/bin/env python3
"""Build a paired raw-versus-EMA table from two repeated-run summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SPLITS = ("gr1_env", "gr1_object", "gr1_behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--ema-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_count") != 3:
        raise ValueError(f"{path}: expected exactly three runs")
    if not isinstance(payload.get("aggregate"), dict):
        raise ValueError(f"{path}: missing aggregate")
    return payload


def main() -> None:
    args = parse_args()
    raw = load_summary(args.raw_summary)
    ema = load_summary(args.ema_summary)
    raw_models = raw["expected_models"]
    expected_ema = {f"{model}_ema" for model in raw_models}
    if set(ema["expected_models"]) != expected_ema:
        raise ValueError("EMA model set does not match raw model set plus _ema suffix")

    rows: list[dict[str, Any]] = []
    for raw_model in raw_models:
        ema_model = f"{raw_model}_ema"
        raw_values = raw["aggregate"][raw_model]
        ema_values = ema["aggregate"][ema_model]
        row = {
            "model": raw_model,
            "raw_mean": raw_values["overall_mean"],
            "raw_sample_sd": raw_values["overall_sample_sd"],
            "ema_mean": ema_values["overall_mean"],
            "ema_sample_sd": ema_values["overall_sample_sd"],
            "delta_ema_minus_raw": (
                ema_values["overall_mean"] - raw_values["overall_mean"]
            ),
        }
        for split in SPLITS:
            short = split.removeprefix("gr1_")
            raw_split = raw_values["split_means"][split]
            ema_split = ema_values["split_means"][split]
            row[f"raw_{short}"] = raw_split
            row[f"ema_{short}"] = ema_split
            row[f"delta_{short}"] = ema_split - raw_split
        rows.append(row)

    output = {
        "definition": "delta = EMA three-run mean - raw three-run mean",
        "raw_summary": str(args.raw_summary.resolve()),
        "ema_summary": str(args.ema_summary.resolve()),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
