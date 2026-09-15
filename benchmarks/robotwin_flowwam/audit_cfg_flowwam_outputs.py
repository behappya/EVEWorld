#!/usr/bin/env python3
"""Audit the completed CFG grid and FlowWAM held-out media outputs."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import av


def load_cfg_rows(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("env", "object", "behavior"):
        rows.extend(json.loads((data_root / f"eval175_gr1_{split}.json").read_text()))
    ids = [str(row.get("request_id", "")) for row in rows]
    if len(rows) != 126 or len(set(ids)) != 126 or any(not value for value in ids):
        raise ValueError(f"invalid CFG manifest: rows={len(rows)} unique={len(set(ids))}")
    return rows


def load_flow_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = json.loads(manifest.read_text())
    ids = [str(row.get("request_id", "")) for row in rows]
    if len(rows) != 250 or len(set(ids)) != 250 or any(not value for value in ids):
        raise ValueError(f"invalid FlowWAM manifest: rows={len(rows)} unique={len(set(ids))}")
    return rows


def probe(path: Path) -> dict[str, Any]:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = sum(1 for _ in container.decode(stream))
            rate = stream.average_rate or stream.base_rate
            return {
                "width": int(stream.codec_context.width),
                "height": int(stream.codec_context.height),
                "frames": frames,
                "fps": float(rate) if rate else None,
                "bytes": path.stat().st_size,
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001
        return {"width": None, "height": None, "frames": None, "fps": None,
                "bytes": path.stat().st_size if path.exists() else 0,
                "error": f"{type(exc).__name__}: {exc}"}


def audit_dir(
    directory: Path,
    expected_ids: set[str],
    *,
    width: int,
    height: int,
    fps: float,
    expected_frames: int | None = None,
    max_frames: dict[str, int] | None = None,
    workers: int,
) -> dict[str, Any]:
    actual_paths = {path.stem: path for path in directory.glob("*.mp4")}
    missing = sorted(expected_ids - set(actual_paths))
    extra = sorted(set(actual_paths) - expected_ids)
    names = sorted(expected_ids & set(actual_paths))

    def inspect(name: str) -> dict[str, Any]:
        path = actual_paths[name]
        info = probe(path)
        errors: list[str] = []
        if info["error"]:
            errors.append(info["error"])
        if info["width"] != width or info["height"] != height:
            errors.append(f"resolution={info['width']}x{info['height']}, expected={width}x{height}")
        if info["fps"] is None or abs(info["fps"] - fps) > 0.01:
            errors.append(f"fps={info['fps']}, expected={fps}")
        if info["bytes"] <= 10_000:
            errors.append(f"bytes={info['bytes']}, expected >10000")
        if expected_frames is not None and info["frames"] != expected_frames:
            errors.append(f"frames={info['frames']}, expected={expected_frames}")
        if max_frames is not None and (info["frames"] is None or info["frames"] > max_frames[name]):
            errors.append(f"frames={info['frames']}, gt_frames={max_frames[name]}")
        return {"id": name, **info, "errors": errors}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        details = list(pool.map(inspect, names))
    invalid = [item for item in details if item["errors"]]
    return {
        "directory": str(directory),
        "expected_count": len(expected_ids),
        "actual_count": len(actual_paths),
        "valid_count": len(details) - len(invalid),
        "missing": missing,
        "extra": extra,
        "invalid_count": len(invalid),
        "invalid": invalid,
        "complete": not missing and not extra and not invalid,
        "details": details,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-data-root", type=Path, required=True)
    ap.add_argument("--cfg-root", type=Path, required=True)
    ap.add_argument("--flow-manifest", type=Path, required=True)
    ap.add_argument("--flow-root", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg_rows = load_cfg_rows(args.cfg_data_root)
    cfg_ids = {str(row["request_id"]) for row in cfg_rows}
    flow_rows = load_flow_rows(args.flow_manifest)
    flow_ids = {str(row["request_id"]) for row in flow_rows}

    cfg: dict[str, Any] = {}
    for step in (50, 100, 150, 200, 250, 300):
        for cfg_value, tag in ((1.0, "cfg_1"), (2.5, "cfg_2p5"),
                               (5.0, "cfg_5"), (7.5, "cfg_7p5")):
            key = f"step_{step:03d}/{tag}"
            cfg[key] = audit_dir(
                args.cfg_root / f"step_{step:03d}" / tag / "generated_only",
                cfg_ids, width=768, height=480, fps=16.0, expected_frames=93,
                workers=args.workers,
            )

    flow: dict[str, Any] = {}
    for arm in ("control", "eve"):
        directory = args.flow_root / f"arm_{arm}_final_robot_only"
        max_frames: dict[str, int] = {}
        for row in flow_rows:
            with av.open(row["gt_video"]) as container:
                max_frames[row["request_id"]] = sum(1 for _ in container.decode(container.streams.video[0]))
        flow[arm] = audit_dir(
            directory, flow_ids, width=640, height=480, fps=24.0,
            max_frames=max_frames, workers=args.workers,
        )

    report = {
        "protocol": "cfg_grid_and_flowwam_media_audit_v1",
        "cfg": cfg,
        "flowwam": flow,
        "complete": all(item["complete"] for item in cfg.values())
        and all(item["complete"] for item in flow.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "cfg_cells": len(cfg),
        "cfg_complete_cells": sum(item["complete"] for item in cfg.values()),
        "cfg_videos": sum(item["valid_count"] for item in cfg.values()),
        "flow_control": flow["control"]["valid_count"],
        "flow_eve": flow["eve"]["valid_count"],
        "complete": report["complete"],
        "report": str(args.report),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
