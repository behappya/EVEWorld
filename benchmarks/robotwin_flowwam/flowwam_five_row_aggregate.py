#!/usr/bin/env python3
"""Aggregate the frozen five-row FlowWAM R250 table from audited metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROWS = (
    ("stage1", "FlowWAM Stage-1", False, False),
    ("sft", "Standard SFT", False, False),
    ("igr", "+ IGR", True, False),
    ("tia", "+ TIA", False, True),
    ("eve", "EVEWorld", True, True),
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract(psnr: dict[str, Any], perceptual: dict[str, Any], mlr: dict[str, Any],
            key: str) -> dict[str, Any]:
    p = psnr["arms"][key]
    q = perceptual["arms"][key]
    m = mlr["summary"][key]
    if p["videos"] != 250 or p["truncated_videos"] != 0:
        raise ValueError(f"{key}: invalid PSNR/SSIM coverage: {p}")
    if q["valid"] != 250 or q["errors"] != 0:
        raise ValueError(f"{key}: invalid LPIPS/EPE coverage: {q}")
    if m["records"] != 199 or m["eligible"] != 71 or m["errors"] != 0:
        raise ValueError(f"{key}: invalid MLR coverage: {m}")
    return {
        "psnr_db": p["mean_psnr_db"],
        "ssim": p["mean_ssim"],
        "lpips": q["mean_lpips"],
        "flow_epe": q["mean_flow_epe"],
        "mlr_percent": 100.0 * m["mlr"],
        "mlr_events": m["events"],
        "mlr_eligible": m["eligible"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--campaign-root", type=Path,
        default=Path("/data/datasets/gagi/flowwam/five_row_r250_v1"),
    )
    ap.add_argument(
        "--existing-root", type=Path,
        default=Path("/data/datasets/gagi/flowwam/heldout_r250_v1"),
    )
    args = ap.parse_args()

    results: list[dict[str, Any]] = []
    for key, label, igr, tia in ROWS:
        if key == "eve":
            psnr_path = args.existing_root / "fidelity_psnr_ssim.json"
            perceptual_path = args.existing_root / "lpips_flow_epe_v2/summary.json"
            mlr_path = args.campaign_root / "metrics/eve/mlr/summary.json"
            source_key = "eve"
            psnr_key = perceptual_key = "eve"
        else:
            metric_root = args.campaign_root / "metrics" / key
            psnr_path = metric_root / "psnr_ssim.json"
            perceptual_path = metric_root / "lpips_flow_epe/summary.json"
            mlr_path = metric_root / "mlr/summary.json"
            source_key = psnr_key = perceptual_key = key
        psnr = load(psnr_path)
        perceptual = load(perceptual_path)
        mlr = load(mlr_path)
        values = extract(
            {"arms": {key: psnr["arms"][psnr_key]}},
            {"arms": {key: perceptual["arms"][perceptual_key]}},
            {"summary": {key: mlr["summary"][source_key]}},
            key,
        )
        results.append({
            "key": key, "variant": label, "igr": igr, "tia": tia,
            **values,
            "sources": {
                "psnr_ssim": str(psnr_path),
                "lpips_flow_epe": str(perceptual_path),
                "mlr": str(mlr_path),
            },
        })

    report = {
        "protocol": "flowwam_five_row_heldout_r250_v1",
        "manifest": str(args.campaign_root / "manifest.json"),
        "rows": results,
    }
    output = args.campaign_root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    (output / "table5.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "table5.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Variant", "IGR", "TIA", "PSNR", "SSIM", "LPIPS",
            "Flow-EPE", "MLR (%)", "MLR events", "MLR eligible",
        ])
        for row in results:
            writer.writerow([
                row["variant"], int(row["igr"]), int(row["tia"]),
                f"{row['psnr_db']:.3f}", f"{row['ssim']:.3f}",
                f"{row['lpips']:.3f}", f"{row['flow_epe']:.3f}",
                f"{row['mlr_percent']:.2f}", row["mlr_events"],
                row["mlr_eligible"],
            ])
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
