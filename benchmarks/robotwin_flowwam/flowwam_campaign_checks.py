#!/usr/bin/env python3
"""Portable validation helpers for the FlowWAM five-row kjob pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


MLR_ELIGIBLE_COUNT = 71
MLR_ELIGIBLE_SHA256 = "3b23c8c8fa1ec2562a100dbbeee0e57fd40e807b18dc2dbea83fa10d798e8a53"


def read(path: Path):
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest")
    p.add_argument("path", type=Path)

    p = sub.add_parser("protocol")
    p.add_argument("output", type=Path)
    p.add_argument("manifest")
    p.add_argument("stage1")
    p.add_argument("--tia-inject", choices=("on", "off"), default="off")

    p = sub.add_parser("train")
    p.add_argument("path", type=Path)
    p.add_argument("variant")

    p = sub.add_parser("psnr")
    p.add_argument("path", type=Path)
    p.add_argument("variant")

    p = sub.add_parser("lpips")
    p.add_argument("path", type=Path)
    p.add_argument("variant")

    p = sub.add_parser("mlr")
    p.add_argument("path", type=Path)
    p.add_argument("variant")
    args = parser.parse_args()

    if args.command == "manifest":
        rows = read(args.path)
        if len(rows) != 250:
            raise SystemExit(f"expected 250 manifest rows, got {len(rows)}")
        ids = {row["request_id"] for row in rows}
        if len(ids) != 250:
            raise SystemExit(f"expected 250 unique request IDs, got {len(ids)}")
        print(f"MANIFEST_OK rows={len(rows)} unique_ids={len(ids)}")
        return 0

    if args.command == "protocol":
        payload = {
            "campaign": "flowwam_five_row_heldout_r250_v1",
            "manifest": args.manifest,
            "stage1": args.stage1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "train": {
                "split": "episodes 0-44", "epochs": 4, "steps": 1128,
                "flow_mode": "robot_only", "lora_rank": 32, "lr": 0.0001,
            },
            "generation": {
                "split": "episodes 45-49", "videos": 250,
                "flow_condition": "robot_only", "trajectory": "direct",
                "cfg": 5.0, "steps": 40, "seed": 42,
                "tia_inject": args.tia_inject == "on",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"PROTOCOL_OK path={args.output}")
        return 0

    if args.command == "train":
        payload = read(args.path)
        if not (payload.get("arm") == args.variant
                and payload.get("num_epochs") == 4
                and payload.get("flow_mode") == "robot_only"):
            raise SystemExit(
                f"training arguments do not match {args.variant}: {payload}"
            )
        print(f"TRAIN_ARGS_OK variant={args.variant}")
        return 0

    payload = read(args.path)
    if args.command == "psnr":
        arm = payload["arms"][args.variant]
        ok = arm.get("videos") == 250 and arm.get("truncated_videos") == 0
        label = "PSNR_OK" if ok else "invalid PSNR/SSIM coverage"
    elif args.command == "lpips":
        arm = payload["arms"][args.variant]
        ok = arm.get("valid") == 250 and arm.get("errors") == 0
        label = "LPIPS_OK" if ok else "invalid LPIPS/EPE coverage"
    else:
        arm = payload["summary"][args.variant]
        eligible_ids = sorted(
            row["request_id"] for row in payload.get("records", [])
            if row.get("model") == args.variant
            and row.get("eligible") and not row.get("error")
        )
        eligible_sha256 = hashlib.sha256(
            "\n".join(eligible_ids).encode("utf-8")
        ).hexdigest()
        ok = (arm.get("records") == 199
              and arm.get("eligible") == MLR_ELIGIBLE_COUNT
              and arm.get("errors") == 0
              and len(eligible_ids) == MLR_ELIGIBLE_COUNT
              and eligible_sha256 == MLR_ELIGIBLE_SHA256)
        label = "MLR_OK" if ok else "invalid MLR coverage"
    if not ok:
        raise SystemExit(f"{label}: {arm}")
    print(f"{label} variant={args.variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
