#!/usr/bin/env python3
"""Compute frame-aligned fidelity metrics for FlowWAM held-out videos."""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def parse_variants(specs: list[str], flow_root: Path | None) -> dict[str, Path]:
    if specs:
        variants: dict[str, Path] = {}
        for spec in specs:
            if "=" not in spec:
                raise ValueError(f"variant must be NAME=VIDEO_DIR, got: {spec}")
            name, directory = spec.split("=", 1)
            if not name or name in variants:
                raise ValueError(f"invalid or duplicate variant name: {name!r}")
            variants[name] = Path(directory)
        return variants
    if flow_root is None:
        raise ValueError("provide --variant or legacy --flow-root")
    return {
        arm: flow_root / f"arm_{arm}_final_robot_only"
        for arm in ("control", "eve")
    }


def read_pair(generated: Path, target: Path, ssim_samples: int) -> dict[str, Any]:
    pred = cv2.VideoCapture(str(generated))
    gt = cv2.VideoCapture(str(target))
    if not pred.isOpened() or not gt.isOpened():
        raise RuntimeError(f"cannot open pair: {generated} / {target}")
    pred_count = int(pred.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    gt_count = int(gt.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total = min(pred_count, gt_count)
    if total <= 0:
        raise RuntimeError(f"empty pair: {generated} / {target}")
    sample_ids = set(np.linspace(0, total - 1, min(ssim_samples, total)).astype(int).tolist())
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    index = 0
    try:
        while index < total:
            ok_p, frame_p = pred.read()
            ok_g, frame_g = gt.read()
            if not ok_p or not ok_g:
                break
            diff = frame_p.astype(np.float32) - frame_g.astype(np.float32)
            mse = float(np.mean(diff * diff))
            psnr_values.append(10.0 * math.log10((255.0 * 255.0) / max(mse, 1e-12)))
            if index in sample_ids:
                ssim_values.append(float(structural_similarity(
                    frame_p, frame_g, channel_axis=2, data_range=255
                )))
            index += 1
    finally:
        pred.release()
        gt.release()
    if not psnr_values:
        raise RuntimeError(f"no decodable frames: {generated}")
    return {
        "generated_frames": pred_count,
        "target_frames": gt_count,
        "aligned_frames": len(psnr_values),
        "psnr_db": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)) if ssim_values else None,
        "truncated": pred_count < gt_count,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--flow-root", type=Path)
    ap.add_argument(
        "--variant", action="append", default=[], metavar="NAME=VIDEO_DIR",
        help="Variant and generated-video directory; repeat for multiple rows.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ssim-samples", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--expected-count", type=int, default=250,
        help="Fail unless the manifest contains this many rows.",
    )
    args = ap.parse_args()
    rows = json.loads(args.manifest.read_text())
    if len(rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} manifest rows, got {len(rows)}"
        )
    variants = parse_variants(args.variant, args.flow_root)
    all_results: dict[str, Any] = {}
    for arm, directory in variants.items():
        pairs = [(row, directory / f"{row['request_id']}.mp4") for row in rows]
        missing = [str(path) for _, path in pairs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{arm}: missing {len(missing)} videos, e.g. {missing[:3]}")

        def one(pair: tuple[dict[str, Any], Path]) -> dict[str, Any]:
            row, path = pair
            result = read_pair(path, Path(row["gt_video"]), args.ssim_samples)
            return {"request_id": row["request_id"], **result}

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            details = list(pool.map(one, pairs))
        all_results[arm] = details

    summary: dict[str, Any] = {"protocol": "flowwam_fidelity_psnr_ssim_v1", "arms": {}}
    for arm, details in all_results.items():
        summary["arms"][arm] = {
            "videos": len(details),
            "mean_psnr_db": float(np.mean([x["psnr_db"] for x in details])),
            "median_psnr_db": float(np.median([x["psnr_db"] for x in details])),
            "mean_ssim": float(np.mean([x["ssim"] for x in details if x["ssim"] is not None])),
            "mean_aligned_frames": float(np.mean([x["aligned_frames"] for x in details])),
            "truncated_videos": sum(bool(x["truncated"]) for x in details),
        }
    summary["details"] = all_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["arms"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
