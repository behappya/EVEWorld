#!/usr/bin/env python3
"""Audit remaining s150 and all s200 Qwen runs, then compare all 70 seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SPLITS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}
LATE_S150_SEEDS = (33, 34, 35, 68, 69, 70)
ALL_SEEDS = tuple(range(1, 71))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--s150-batch01-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(values: list[float], splits: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "repeat_scores_percent": values,
        "mean_percent": statistics.mean(values),
        "sample_sd_percent": statistics.stdev(values),
        "splits": {
            split: {
                "repeat_scores_percent": split_values,
                "mean_percent": statistics.mean(split_values),
                "sample_sd_percent": statistics.stdev(split_values),
            }
            for split, split_values in splits.items()
        },
    }


def audit_batch(
    qwen_root: Path,
    prefix: str,
    model: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    seed_set = set(seeds)
    expected = len(seeds) * 126
    scores = {seed: [] for seed in seeds}
    split_scores = {seed: {split: [] for split in SPLITS} for seed in seeds}
    endpoints: set[str] = set()
    judges: set[str] = set()
    repeats: dict[str, Any] = {}

    for repeat in range(1, 4):
        run_name = f"{prefix}_repeat{repeat:02d}"
        run_dir = qwen_root / run_name
        config = json.loads(
            (run_dir / f"{run_name}_run_config.json").read_text(encoding="utf-8")
        )
        if config.get("disable_thinking") is not True:
            raise ValueError(f"{run_name}: thinking is not disabled")
        endpoints.add(config["qwen_base"])
        judges.add(config["qwen_model"])
        csv_path = run_dir / f"{run_name}_qwen_if.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected or len({row["key"] for row in rows}) != expected:
            raise ValueError(f"{run_name}: incomplete or duplicate rows")
        if {row["model"] for row in rows} != {model}:
            raise ValueError(f"{run_name}: model label drift")
        if any(row.get("error") or not row.get("raw_text", "").strip() for row in rows):
            raise ValueError(f"{run_name}: error or empty response")

        by_seed = {seed: [] for seed in seeds}
        for row in rows:
            parts = row["key"].split("/")
            if len(parts) != 4 or parts[0] != model or not parts[1].startswith("seed"):
                raise ValueError(f"{run_name}: bad key {row['key']}")
            seed = int(parts[1][4:])
            if seed not in seed_set:
                raise ValueError(f"{run_name}: unexpected seed {seed}")
            prediction = int(float(row["prediction"]))
            if prediction not in (0, 1) or float(row["prediction"]) != prediction:
                raise ValueError(f"{run_name}: non-binary prediction")
            by_seed[seed].append(row)

        repeat_seeds = {}
        for seed, seed_rows in by_seed.items():
            if len(seed_rows) != 126:
                raise ValueError(f"{run_name} seed{seed:03d}: incomplete")
            value = 100.0 * sum(int(float(row["prediction"])) for row in seed_rows) / 126
            scores[seed].append(value)
            seed_splits = {}
            for split, count in SPLITS.items():
                subset = [row for row in seed_rows if row["split"] == split]
                if len(subset) != count:
                    raise ValueError(f"{run_name} seed{seed:03d}: {split} count drift")
                split_value = 100.0 * sum(
                    int(float(row["prediction"])) for row in subset
                ) / count
                split_scores[seed][split].append(split_value)
                seed_splits[split] = split_value
            repeat_seeds[f"seed{seed:03d}"] = {
                "score_percent": value,
                "splits_percent": seed_splits,
            }
        repeats[f"repeat{repeat:02d}"] = {
            "csv": str(csv_path),
            "seeds": repeat_seeds,
        }

    if len(endpoints) != 1 or len(judges) != 1:
        raise ValueError(f"{prefix}: endpoint or judge differs across repeats")
    return {
        "protocol": {
            "endpoint": next(iter(endpoints)),
            "judge_model": next(iter(judges)),
        },
        "expected_per_repeat": expected,
        "repeats": repeats,
        "aggregate": {
            f"seed{seed:03d}": aggregate(scores[seed], split_scores[seed])
            for seed in seeds
        },
    }


def rank(aggregate_by_seed: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = aggregate_by_seed["seed004"]["mean_percent"]
    rows = [
        {
            "seed": int(seed_name[4:]),
            "mean_percent": values["mean_percent"],
            "sample_sd_percent": values["sample_sd_percent"],
            "delta_vs_seed004_pp": values["mean_percent"] - baseline,
        }
        for seed_name, values in aggregate_by_seed.items()
    ]
    rows.sort(key=lambda row: (-row["mean_percent"], row["sample_sd_percent"], row["seed"]))
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def main() -> None:
    args = parse_args()
    late_s150 = audit_batch(
        args.qwen_root, "s150_late06", "transport_raw_s150", LATE_S150_SEEDS
    )
    s200 = audit_batch(args.qwen_root, "s200_all70", "transport_raw_s200", ALL_SEEDS)
    batch01 = json.loads(args.s150_batch01_audit.read_text(encoding="utf-8"))
    if batch01.get("ready") is not True or len(batch01.get("aggregate", {})) != 64:
        raise ValueError("s150 batch01 audit is not ready or does not contain 64 seeds")
    if batch01["protocol"]["endpoint"] != late_s150["protocol"]["endpoint"]:
        raise ValueError("s150 endpoint drift between batch01 and late06")
    if batch01["protocol"]["model"] != late_s150["protocol"]["judge_model"]:
        raise ValueError("s150 judge drift between batch01 and late06")
    if late_s150["protocol"] != s200["protocol"]:
        raise ValueError("s150 late06 and s200 protocol drift")

    s150_aggregate = dict(batch01["aggregate"])
    overlap = set(s150_aggregate) & set(late_s150["aggregate"])
    if overlap:
        raise ValueError(f"s150 batch overlap: {sorted(overlap)}")
    s150_aggregate.update(late_s150["aggregate"])
    expected_names = {f"seed{seed:03d}" for seed in ALL_SEEDS}
    if set(s150_aggregate) != expected_names or set(s200["aggregate"]) != expected_names:
        raise ValueError("s150 or s200 does not contain exactly seeds001-070")

    paired = {}
    deltas = []
    for seed in ALL_SEEDS:
        name = f"seed{seed:03d}"
        s150_value = s150_aggregate[name]["mean_percent"]
        s200_value = s200["aggregate"][name]["mean_percent"]
        delta = s200_value - s150_value
        deltas.append(delta)
        paired[name] = {
            "s150_mean_percent": s150_value,
            "s200_mean_percent": s200_value,
            "s200_minus_s150_pp": delta,
        }

    report = {
        "schema": "eve-cic-transport-s150-s200-all70-qwen-three-repeat-v1",
        "protocol": {
            **late_s150["protocol"],
            "metric": "qwen_if",
            "frame_count": 49,
            "jpeg_quality": 85,
            "temperature": 0,
            "max_tokens": 32000,
            "thinking": False,
            "repeats": 3,
        },
        "s150_late06": late_s150,
        "s150_all70": {
            "aggregate": s150_aggregate,
            "ranking": rank(s150_aggregate),
            "mean_across_seeds_percent": statistics.mean(
                values["mean_percent"] for values in s150_aggregate.values()
            ),
        },
        "s200_all70": {
            **s200,
            "ranking": rank(s200["aggregate"]),
            "mean_across_seeds_percent": statistics.mean(
                values["mean_percent"] for values in s200["aggregate"].values()
            ),
        },
        "paired_s200_vs_s150": {
            "per_seed": paired,
            "mean_delta_pp": statistics.mean(deltas),
            "sample_sd_delta_pp": statistics.stdev(deltas),
            "s200_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "s200_losses": sum(delta < 0 for delta in deltas),
        },
        "ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
