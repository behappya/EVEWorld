#!/usr/bin/env python3
"""Compute LPIPS and RAFT optical-flow EPE for FlowWAM held-out media."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_variants(specs: list[str], flow_root: Path | None) -> dict[str, Path]:
    if specs:
        variants: dict[str, Path] = {}
        for spec in specs:
            if "=" not in spec:
                raise ValueError(f"variant must be NAME=VIDEO_DIR, got: {spec}")
            name, directory = spec.split("=", 1)
            if not name or not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"invalid variant name: {name!r}")
            if name in variants:
                raise ValueError(f"duplicate variant name: {name}")
            variants[name] = Path(directory)
        return variants
    if flow_root is None:
        raise ValueError("provide --variant or legacy --flow-root")
    return {
        arm: flow_root / f"arm_{arm}_final_robot_only"
        for arm in ("control", "eve")
    }


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    return frames


def sample_indices(total: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, total - 1, min(total, count)).astype(int).tolist()))


def evaluate_pair(generated: Path, target: Path, lpips_model: Any, raft: Any,
                  lpips_samples: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    pred = read_frames(generated)
    gt = read_frames(target)
    target_frames = len(gt)
    total = min(len(pred), target_frames)
    if total < 2:
        raise RuntimeError(f"pair has fewer than 2 frames: {generated}")
    pred = pred[:total]
    gt = gt[:total]

    ids = sample_indices(total, lpips_samples)
    pred_tensor = torch.from_numpy(np.stack([pred[i] for i in ids])).permute(0, 3, 1, 2).float()
    gt_tensor = torch.from_numpy(np.stack([gt[i] for i in ids])).permute(0, 3, 1, 2).float()
    pred_tensor = functional.interpolate(pred_tensor, size=(256, 256), mode="bilinear", align_corners=False)
    gt_tensor = functional.interpolate(gt_tensor, size=(256, 256), mode="bilinear", align_corners=False)
    pred_tensor = pred_tensor.to(raft.device) / 127.5 - 1.0
    gt_tensor = gt_tensor.to(raft.device) / 127.5 - 1.0
    with torch.no_grad():
        lpips_value = float(lpips_model(pred_tensor, gt_tensor).mean().item())

    pred_flows = raft.batch_call(pred, max_batch_size=16)
    gt_flows = raft.batch_call(gt, max_batch_size=16)
    epe = [float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32), axis=-1).mean())
           for a, b in zip(pred_flows, gt_flows)]
    return {
        "generated_frames": len(pred),
        "target_frames": target_frames,
        "aligned_frames": total,
        "lpips": lpips_value,
        "flow_epe": float(np.mean(epe)),
        "flow_pairs": len(epe),
    }


def worker_main(args: argparse.Namespace) -> int:
    import lpips
    from raft_flow_extractor import RAFTFlowExtractor

    rows = json.loads(args.manifest.read_text())
    rows = rows[args.shard_index :: args.num_shards]
    directory = args.video_dir
    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    raft = RAFTFlowExtractor(device=device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(raft.device).eval()
    records: list[dict[str, Any]] = []
    output = args.output_dir / f"{args.variant_name}_part{args.shard_index}.json"
    for index, row in enumerate(rows, start=1):
        generated = directory / f"{row['request_id']}.mp4"
        try:
            metrics = evaluate_pair(generated, Path(row["gt_video"]), lpips_model, raft, args.lpips_samples)
            records.append({"request_id": row["request_id"], "error": None, **metrics})
        except Exception as exc:  # noqa: BLE001
            records.append({"request_id": row["request_id"], "error": f"{type(exc).__name__}: {exc}"})
        if index % 5 == 0 or index == len(rows):
            output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
            print(
                f"variant={args.variant_name} shard={args.shard_index} "
                f"progress={index}/{len(rows)}", flush=True,
            )
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--flow-root", type=Path)
    ap.add_argument(
        "--variant", action="append", default=[], metavar="NAME=VIDEO_DIR",
        help="Variant and generated-video directory; repeat for multiple rows.",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--lpips-samples", type=int, default=8)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--variant-name", default="")
    ap.add_argument("--video-dir", type=Path)
    args = ap.parse_args()
    if args.worker:
        if not args.variant_name or args.video_dir is None:
            raise ValueError("worker requires --variant-name and --video-dir")
        return worker_main(args)
    if args.num_shards != 8:
        raise ValueError("this runner requires eight GPUs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variant, args.flow_root)
    for arm, directory in variants.items():
        processes = []
        for shard in range(args.num_shards):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), PYTHONUNBUFFERED="1")
            processes.append(subprocess.Popen([
                sys.executable, __file__, "--worker", "--manifest", str(args.manifest),
                "--output-dir", str(args.output_dir),
                "--num-shards", str(args.num_shards), "--lpips-samples", str(args.lpips_samples),
                "--shard-index", str(shard), "--variant-name", arm,
                "--video-dir", str(directory),
            ], env=env))
        rc = 0
        for process in processes:
            rc |= process.wait()
        if rc:
            return rc
        details: list[dict[str, Any]] = []
        for shard in range(args.num_shards):
            # Arm-specific part files are renamed after each arm below.
            path = args.output_dir / f"{arm}_part{shard}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            details.extend(json.loads(path.read_text()))
        valid = [row for row in details if not row.get("error")]
        summary = {
            "videos": len(details),
            "valid": len(valid),
            "errors": len(details) - len(valid),
            "mean_lpips": float(np.mean([row["lpips"] for row in valid])) if valid else None,
            "mean_flow_epe": float(np.mean([row["flow_epe"] for row in valid])) if valid else None,
            "mean_aligned_frames": float(np.mean([row["aligned_frames"] for row in valid])) if valid else None,
        }
        (args.output_dir / f"{arm}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output_dir / f"{arm}_details.json").write_text(json.dumps(details, indent=2) + "\n")
    report = {"protocol": "flowwam_lpips_raft_epe_v1", "arms": {}}
    for arm in variants:
        report["arms"][arm] = json.loads((args.output_dir / f"{arm}_summary.json").read_text())
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
