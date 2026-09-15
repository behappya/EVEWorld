#!/usr/bin/env python3
"""Strictly validate and summarize DreamGenBench EVAL-175 scorer outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLIT_COUNTS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}
METRICS = ("qwen_if", "gpt_if", "pa_i", "pa_ii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="", help="Comma-separated model names; default discovers manifests")
    parser.add_argument("--baseline", default="round0")
    parser.add_argument("--require", default="qwen_if,pa_i,pa_ii", help="Metrics required for a successful exit")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_path(results_dir: Path, metric: str) -> Path | None:
    direct = results_dir / f"{metric}.csv"
    if direct.is_file():
        return direct
    candidates = sorted(path for path in results_dir.glob(f"*_{metric}.csv") if not path.name.endswith("_raw.csv"))
    if len(candidates) > 1:
        raise ValueError(f"Multiple PA-II candidates under {results_dir}: {candidates}")
    return candidates[0] if candidates else None


def path_aliases(path_value: str | None) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value).expanduser()
    aliases = {str(path), str(path.absolute())}
    try:
        aliases.add(str(path.resolve()))
    except OSError:
        pass
    return aliases


def load_metric_records(
    metric: str,
    path: Path | None,
    manifest_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    if path is None:
        return {}, [f"missing {metric} CSV"]
    expected_keys = {row["key"] for row in manifest_rows}
    path_to_key: dict[str, str] = {}
    for row in manifest_rows:
        for field in ("video_path", "pa2_video_path"):
            for alias in path_aliases(row.get(field)):
                path_to_key[alias] = row["key"]

    records: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_csv(path), start=2):
        key = row.get("key", "").strip()
        if not key:
            video_path = row.get("video_path") or row.get("videopath")
            aliases = path_aliases(video_path)
            key = next((path_to_key[alias] for alias in aliases if alias in path_to_key), "")
        if not key:
            errors.append(f"{path}:{line_number}: cannot map row to manifest key")
            continue
        if key not in expected_keys:
            errors.append(f"{path}:{line_number}: unexpected key {key}")
            continue
        if key in records:
            errors.append(f"{path}:{line_number}: duplicate key {key}")
            continue
        try:
            prediction = int(float(row.get("prediction", "")))
        except ValueError:
            errors.append(f"{path}:{line_number}: invalid prediction for {key}")
            continue
        if prediction not in (0, 1):
            errors.append(f"{path}:{line_number}: non-binary prediction {prediction} for {key}")
            continue
        records[key] = {
            "prediction": prediction,
            "error": row.get("error", ""),
            "raw_score": float(row["raw_score"]) if row.get("raw_score") not in (None, "") else None,
        }
    return records, errors


def wilson_interval(positive: int, count: int) -> list[float] | None:
    if count <= 0:
        return None
    z = 1.959963984540054
    proportion = positive / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_metric(
    records: dict[str, dict[str, Any]],
    keys: list[str],
    source_path: Path | None,
    parse_errors: list[str],
) -> dict[str, Any]:
    selected = [records[key] for key in keys if key in records]
    positive = sum(row["prediction"] for row in selected)
    row_errors = sum(bool(row["error"]) for row in selected)
    complete = len(selected) == len(keys) and row_errors == 0 and not parse_errors
    raw_scores = [row["raw_score"] for row in selected if row["raw_score"] is not None]
    return {
        "path": str(source_path) if source_path is not None else None,
        "expected_count": len(keys),
        "count": len(selected),
        "positive": positive,
        "error_count": row_errors,
        "complete": complete,
        "score": positive / len(selected) if complete and selected else None,
        "wilson_95": wilson_interval(positive, len(selected)) if complete else None,
        "mean_raw_score": sum(raw_scores) / len(raw_scores) if complete and raw_scores else None,
        "parse_errors": parse_errors,
    }


def format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    args = parse_args()
    manifest_root = args.manifest_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {item.strip() for item in args.require.split(",") if item.strip()}
    unknown_required = sorted(required - set(METRICS))
    if unknown_required:
        raise SystemExit(f"Unknown required metrics: {unknown_required}")
    if args.models.strip():
        models = [item.strip() for item in args.models.split(",") if item.strip()]
    else:
        models = sorted(path.stem for path in manifest_root.glob("*.jsonl") if path.name != "all_models.jsonl")

    payload: dict[str, Any] = {"models": {}, "required_metrics": sorted(required), "errors": []}
    prediction_cache: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    flat_rows: list[dict[str, Any]] = []
    for model in models:
        manifest_path = manifest_root / f"{model}.jsonl"
        if not manifest_path.is_file():
            payload["errors"].append(f"{model}: missing manifest {manifest_path}")
            continue
        manifest_rows = read_jsonl(manifest_path)
        key_to_sample = {row["key"]: row["sample_key"] for row in manifest_rows}
        result_dir = results_root / model
        loaded: dict[str, tuple[dict[str, dict[str, Any]], Path | None, list[str]]] = {}
        for metric in METRICS:
            source_path = metric_path(result_dir, metric)
            records, parse_errors = load_metric_records(metric, source_path, manifest_rows)
            loaded[metric] = (records, source_path, parse_errors)
            prediction_cache[model][metric] = {
                key_to_sample[key]: record for key, record in records.items() if key in key_to_sample
            }

        model_result: dict[str, Any] = {"manifest": str(manifest_path), "splits": {}, "overall": {}}
        all_keys = [row["key"] for row in manifest_rows]
        for split, expected_count in SPLIT_COUNTS.items():
            keys = [row["key"] for row in manifest_rows if row["split"] == split]
            if len(keys) != expected_count:
                payload["errors"].append(f"{model}/{split}: manifest count={len(keys)}, expected {expected_count}")
            metrics: dict[str, Any] = {}
            for metric, (records, source_path, parse_errors) in loaded.items():
                metrics[metric] = summarize_metric(records, keys, source_path, parse_errors)
            pa = None
            if metrics["pa_i"]["score"] is not None and metrics["pa_ii"]["score"] is not None:
                pa = (metrics["pa_i"]["score"] + metrics["pa_ii"]["score"]) / 2.0
            split_result = {"expected_count": expected_count, "metrics": metrics, "pa": pa}
            model_result["splits"][split] = split_result
            flat_rows.append(
                {
                    "model": model,
                    "split": split,
                    "count": expected_count,
                    "qwen_if": metrics["qwen_if"]["score"],
                    "gpt_if": metrics["gpt_if"]["score"],
                    "pa_i": metrics["pa_i"]["score"],
                    "pa_ii": metrics["pa_ii"]["score"],
                    "pa": pa,
                }
            )

        overall_metrics: dict[str, Any] = {}
        for metric, (records, source_path, parse_errors) in loaded.items():
            overall_metrics[metric] = summarize_metric(records, all_keys, source_path, parse_errors)
        overall_pa = None
        if overall_metrics["pa_i"]["score"] is not None and overall_metrics["pa_ii"]["score"] is not None:
            overall_pa = (overall_metrics["pa_i"]["score"] + overall_metrics["pa_ii"]["score"]) / 2.0
        model_result["overall"] = {"metrics": overall_metrics, "pa": overall_pa}
        for metric in required:
            if not overall_metrics[metric]["complete"]:
                payload["errors"].append(f"{model}: required metric {metric} is incomplete")
        payload["models"][model] = model_result

    comparisons: dict[str, Any] = {}
    if args.baseline in payload["models"]:
        for model in models:
            if model == args.baseline or model not in payload["models"]:
                continue
            comparison: dict[str, Any] = {}
            for metric in METRICS:
                baseline_records = prediction_cache[args.baseline].get(metric, {})
                model_records = prediction_cache[model].get(metric, {})
                shared = sorted(set(baseline_records) & set(model_records))
                wins = sum(model_records[key]["prediction"] > baseline_records[key]["prediction"] for key in shared)
                losses = sum(model_records[key]["prediction"] < baseline_records[key]["prediction"] for key in shared)
                comparison[metric] = {"shared": len(shared), "wins": wins, "losses": losses, "ties": len(shared) - wins - losses}
            comparisons[f"{model}-vs-{args.baseline}"] = comparison
    payload["comparisons"] = comparisons
    payload["ready"] = not payload["errors"]

    json_path = output_dir / "eval175_scores.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "eval175_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "split", "count", "qwen_if", "gpt_if", "pa_i", "pa_ii", "pa"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)

    lines = [
        "# DreamGenBench EVAL-175 scores",
        "",
        "| Model | Split | N | Qwen-IF | GPT-IF | PA-I | PA-II | PA |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat_rows:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['count']} | {format_score(row['qwen_if'])} | "
            f"{format_score(row['gpt_if'])} | {format_score(row['pa_i'])} | {format_score(row['pa_ii'])} | "
            f"{format_score(row['pa'])} |"
        )
    if payload["errors"]:
        lines.extend(["", "## Validation errors", ""] + [f"- {error}" for error in payload["errors"]])
    markdown_path = output_dir / "eval175_scores.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if payload["errors"]:
        raise SystemExit(f"Summary is incomplete: {len(payload['errors'])} validation error(s)")


if __name__ == "__main__":
    main()
