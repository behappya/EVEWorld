#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw VideoPhy scores to DreamGenBench PA-II CSV.")
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_csv.open("r", encoding="utf-8", newline="") as src, args.output_csv.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["video_path", "prompt", "prediction", "raw_score"])
        writer.writeheader()
        for row in csv.reader(src):
            if not row:
                continue
            score = float(row[-1])
            writer.writerow({
                "video_path": row[0],
                "prompt": row[1] if len(row) > 2 else "",
                "prediction": 1 if score >= args.threshold else 0,
                "raw_score": score,
            })
    print(f"Wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
