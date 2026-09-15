#!/usr/bin/env python3
"""Aggregate checkpoint-screen PSNR/SSIM files into a compact ranking."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.metrics_root.glob("*/psnr_ssim.json")):
        report = json.loads(path.read_text())
        arms = report.get("arms", {})
        if len(arms) != 1:
            raise ValueError(f"expected one variant in {path}, got {list(arms)}")
        tag, metrics = next(iter(arms.items()))
        rows.append({
            "tag": tag,
            "videos": int(metrics["videos"]),
            "psnr_db": float(metrics["mean_psnr_db"]),
            "ssim": float(metrics["mean_ssim"]),
            "median_psnr_db": float(metrics["median_psnr_db"]),
            "source": str(path),
        })
    rows.sort(key=lambda row: (float(row["psnr_db"]), float(row["ssim"])), reverse=True)
    if not rows:
        raise ValueError(f"no metrics found below {args.metrics_root}")

    payload = {"protocol": "flowwam_checkpoint_screen_dev50_v1", "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
