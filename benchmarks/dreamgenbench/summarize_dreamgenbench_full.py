#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DreamGenBench IF and PA metrics.")
    parser.add_argument("--qwen-if-csv", type=Path, default=None)
    parser.add_argument("--gpt-if-csv", type=Path, default=None)
    parser.add_argument("--pa-i-csv", type=Path, default=None)
    parser.add_argument("--pa-ii-csv", type=Path, default=None)
    parser.add_argument("--pa-ii-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def read_binary_csv(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    values: list[int] = []
    errors = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("prediction")
            if raw is None:
                continue
            try:
                values.append(int(float(raw)))
            except ValueError:
                values.append(0)
            if row.get("error"):
                errors += 1
    return {
        "path": str(path),
        "exists": True,
        "count": len(values),
        "positive": sum(values),
        "error_count": errors,
        "score": sum(values) / len(values) if values else None,
    }


def read_videophy_csv(path: Path | None, threshold: float) -> dict | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        sample_lower = sample.lower()
        has_header = (
            ("videopath" in sample_lower or "video_path" in sample_lower)
            and ("score" in sample_lower or "entail" in sample_lower or "prediction" in sample_lower)
        )
        if has_header:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = row.get("raw_score") or row.get("score") or row.get("entailment") or row.get("prediction")
                if raw is None:
                    numeric = [value for value in row.values() if value is not None]
                    raw = numeric[-1] if numeric else None
                if raw is not None:
                    scores.append(float(raw))
        else:
            reader = csv.reader(fh)
            for row in reader:
                if row:
                    scores.append(float(row[-1]))
    positives = sum(1 for score in scores if score >= threshold)
    return {
        "path": str(path),
        "exists": True,
        "count": len(scores),
        "positive": positives,
        "threshold": threshold,
        "score": positives / len(scores) if scores else None,
        "mean_raw_score": sum(scores) / len(scores) if scores else None,
    }


def main() -> None:
    args = parse_args()
    qwen_if = read_binary_csv(args.qwen_if_csv)
    gpt_if = read_binary_csv(args.gpt_if_csv)
    pa_i = read_binary_csv(args.pa_i_csv)
    pa_ii = read_videophy_csv(args.pa_ii_csv, args.pa_ii_threshold)
    pa = None
    if pa_i and pa_i.get("score") is not None and pa_ii and pa_ii.get("score") is not None:
        pa = (float(pa_i["score"]) + float(pa_ii["score"])) / 2.0
    payload = {
        "qwen_if": qwen_if,
        "gpt_if": gpt_if,
        "pa_i": pa_i,
        "pa_ii": pa_ii,
        "pa": pa,
        "note": "PA follows DreamGen convention only when PA-I and VideoPhy PA-II are both provided.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
