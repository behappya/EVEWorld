#!/usr/bin/env python3
"""Audit and rank a frozen three-repeat Transport s150 Qwen batch."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SPLITS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}
MODEL = "transport_raw_s150"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_seeds(path: Path) -> list[int]:
    seeds = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not seeds or len(seeds) != len(set(seeds)) or 4 not in seeds:
        raise ValueError("frozen seeds must be unique, non-empty, and include seed004")
    return seeds


def parse_seed(row: dict[str, str]) -> int:
    parts = row["key"].split("/")
    if len(parts) != 4 or parts[0] != MODEL or not parts[1].startswith("seed"):
        raise ValueError(f"unexpected key: {row['key']}")
    return int(parts[1][4:])


def main() -> None:
    args = parse_args()
    seeds = load_seeds(args.seed_file)
    seed_set = set(seeds)
    expected_total = 126 * len(seeds)
    scores: dict[int, list[float]] = {seed: [] for seed in seeds}
    split_scores = {
        seed: {split: [] for split in SPLITS} for seed in seeds
    }
    repeats: dict[str, Any] = {}
    endpoints: set[str] = set()
    models: set[str] = set()

    for repeat in range(1, 4):
        run_name = f"{args.batch}_repeat{repeat:02d}"
        run_dir = args.qwen_root / run_name
        csv_path = run_dir / f"{run_name}_qwen_if.csv"
        config_path = run_dir / f"{run_name}_run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("disable_thinking") is not True:
            raise ValueError(f"{run_name}: thinking is not explicitly disabled")
        endpoints.add(config["qwen_base"])
        models.add(config["qwen_model"])

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_total:
            raise ValueError(
                f"{run_name}: expected {expected_total} rows, got {len(rows)}"
            )
        if len({row["key"] for row in rows}) != expected_total:
            raise ValueError(f"{run_name}: duplicate keys")
        if any(row.get("error") for row in rows):
            raise ValueError(f"{run_name}: Qwen errors remain")
        if any(not row.get("raw_text", "").strip() for row in rows):
            raise ValueError(f"{run_name}: empty model response")
        if {row["model"] for row in rows} != {MODEL}:
            raise ValueError(f"{run_name}: unexpected model label")

        by_seed: dict[int, list[dict[str, str]]] = {seed: [] for seed in seeds}
        for row in rows:
            seed = parse_seed(row)
            if seed not in seed_set:
                raise ValueError(f"{run_name}: unexpected seed {seed}")
            prediction = int(float(row["prediction"]))
            if prediction not in (0, 1) or float(row["prediction"]) != prediction:
                raise ValueError(f"{run_name}: non-binary prediction")
            by_seed[seed].append(row)

        repeat_report = {}
        for seed in seeds:
            seed_rows = by_seed[seed]
            if len(seed_rows) != 126:
                raise ValueError(f"{run_name} seed{seed:03d}: got {len(seed_rows)} rows")
            positives = sum(int(float(row["prediction"])) for row in seed_rows)
            overall = 100.0 * positives / 126
            scores[seed].append(overall)
            split_report = {}
            for split, count in SPLITS.items():
                subset = [row for row in seed_rows if row["split"] == split]
                if len(subset) != count:
                    raise ValueError(
                        f"{run_name} seed{seed:03d} {split}: got {len(subset)} rows"
                    )
                split_score = 100.0 * sum(
                    int(float(row["prediction"])) for row in subset
                ) / count
                split_scores[seed][split].append(split_score)
                split_report[split] = split_score
            repeat_report[f"seed{seed:03d}"] = {
                "positive": positives,
                "score_percent": overall,
                "splits_percent": split_report,
            }
        repeats[f"repeat{repeat:02d}"] = {
            "csv": str(csv_path),
            "seeds": repeat_report,
        }

    if len(endpoints) != 1 or len(models) != 1:
        raise ValueError("Qwen endpoint or model differs across repeats")
    aggregate = {}
    for seed in seeds:
        aggregate[f"seed{seed:03d}"] = {
            "repeat_scores_percent": scores[seed],
            "mean_percent": statistics.mean(scores[seed]),
            "sample_sd_percent": statistics.stdev(scores[seed]),
            "splits": {
                split: {
                    "repeat_scores_percent": values,
                    "mean_percent": statistics.mean(values),
                    "sample_sd_percent": statistics.stdev(values),
                }
                for split, values in split_scores[seed].items()
            },
        }

    baseline = aggregate["seed004"]["mean_percent"]
    ranking = sorted(
        (
            {
                "seed": seed,
                "mean_percent": aggregate[f"seed{seed:03d}"]["mean_percent"],
                "sample_sd_percent": aggregate[f"seed{seed:03d}"]["sample_sd_percent"],
                "delta_vs_seed004_pp": aggregate[f"seed{seed:03d}"]["mean_percent"] - baseline,
            }
            for seed in seeds
        ),
        key=lambda row: (-row["mean_percent"], row["sample_sd_percent"], row["seed"]),
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    report = {
        "schema": "eve-cic-transport-s150-multiseed-qwen-three-repeat-v1",
        "batch": args.batch,
        "protocol": {
            "endpoint": next(iter(endpoints)),
            "model": next(iter(models)),
            "metric": "qwen_if",
            "frame_count": 49,
            "jpeg_quality": 85,
            "temperature": 0,
            "max_tokens": 32000,
            "thinking": False,
            "repeats": 3,
        },
        "frozen_seeds": seeds,
        "expected_per_repeat": expected_total,
        "repeats": repeats,
        "aggregate": aggregate,
        "seed004_mean_percent": baseline,
        "ranking": ranking,
        "better_than_seed004": [row for row in ranking if row["delta_vs_seed004_pp"] > 0],
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
