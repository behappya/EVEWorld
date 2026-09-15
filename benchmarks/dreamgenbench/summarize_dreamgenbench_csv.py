#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DreamGenBench binary CSV scores.")
    parser.add_argument("--qwen-if-csv", type=Path, default=None)
    parser.add_argument("--gpt-if-csv", type=Path, default=None)
    parser.add_argument("--pa-i-csv", type=Path, default=None)
    parser.add_argument("--pa-ii-csv", type=Path, default=None)
    parser.add_argument("--gpu-peak-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def average_prediction(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    values: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("prediction")
            if raw is None:
                continue
            try:
                values.append(int(float(raw)))
            except ValueError:
                continue
    score = sum(values) / len(values) if values else None
    return {
        "path": str(path),
        "exists": True,
        "count": len(values),
        "positive": sum(values),
        "score": score,
    }


def read_optional_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.setdefault("path", str(path))
        payload.setdefault("exists", True)
        return payload
    return {"path": str(path), "exists": True, "value": payload}


def main() -> None:
    args = parse_args()
    qwen_if = average_prediction(args.qwen_if_csv)
    gpt_if = average_prediction(args.gpt_if_csv)
    pa_i = average_prediction(args.pa_i_csv)
    pa_ii = average_prediction(args.pa_ii_csv)
    gpu_peak = read_optional_json(args.gpu_peak_json)

    pa_score = None
    if pa_i and pa_i.get("score") is not None and pa_ii and pa_ii.get("score") is not None:
        pa_score = (float(pa_i["score"]) + float(pa_ii["score"])) / 2.0

    summary = {
        "qwen_if": qwen_if,
        "gpt_if": gpt_if,
        "pa_i": pa_i,
        "pa_ii": pa_ii,
        "pa": pa_score,
        "gpu_peak": gpu_peak,
        "note": "DreamGen paper reports PA as the average of PA-I and PA-II. This summary only computes PA when both CSVs are provided.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
