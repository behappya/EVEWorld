#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CORE_METRICS = {
    "image_quality": "Image Quality",
    "aesthetic_quality": "Aesthetic Quality",
    "dynamic_degree": "Dynamic Degree",
    "flow_score": "Flow Score",
    "motion_smoothness": "Motion Smoothness",
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "photometric_smoothness": "Photometric Consistency",
}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate WorldArena 1.0 core-8 scores")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_metric(path: Path, metric: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get(metric)
    if not isinstance(result, list) or len(result) < 2 or not isinstance(result[1], list):
        raise ValueError(f"Invalid {metric} result: {path}")
    values = {}
    for item in result[1]:
        request_id = Path(item.get("video_path", "")).stem
        value = item.get("video_results_normalized", item.get("video_results"))
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Out-of-range {metric} value for {request_id}: {value}")
        values[request_id] = value
    return values


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_ids = [row["request_id"] for row in manifest]
    expected_set = set(expected_ids)
    summaries = []
    display_metrics = list(CORE_METRICS.values())

    for model in args.models:
        per_video = {request_id: {} for request_id in expected_ids}
        for metric, display in CORE_METRICS.items():
            values = load_metric(args.eval_root / model / "core" / f"{metric}.json", metric)
            if set(values) != expected_set:
                raise ValueError(
                    f"{model} {metric} coverage mismatch: {len(values)}/{len(expected_ids)}"
                )
            for request_id, value in values.items():
                per_video[request_id][display] = value

        rows = []
        for request_id in expected_ids:
            metrics = per_video[request_id]
            score = 100.0 * sum(metrics.values()) / len(metrics)
            rows.append(
                {
                    "Model": model,
                    "Video_ID": request_id,
                    **{key: round(metrics[key], 6) for key in display_metrics},
                    "EWMScore-local-8": round(score, 4),
                }
            )

        means = {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in display_metrics
        }
        summaries.append(
            {
                "Model": model,
                "coverage": len(rows),
                **{key: round(value, 6) for key, value in means.items()},
                "EWMScore-local-8": round(
                    100.0 * sum(means.values()) / len(means), 4
                ),
            }
        )
        write_csv(
            args.output_dir / f"{model}_per_video.csv",
            rows,
            ["Model", "Video_ID", *display_metrics, "EWMScore-local-8"],
        )

    summaries.sort(key=lambda row: (-row["EWMScore-local-8"], row["Model"]))
    for rank, row in enumerate(summaries, start=1):
        row["Rank"] = rank
    fields = [
        "Rank",
        "Model",
        "coverage",
        *display_metrics,
        "EWMScore-local-8",
    ]
    write_csv(args.output_dir / "model_comparison.csv", summaries, fields)
    atomic_write_json(args.output_dir / "model_comparison.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
