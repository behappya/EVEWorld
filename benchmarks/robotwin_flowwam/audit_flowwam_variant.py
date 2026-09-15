#!/usr/bin/env python3
"""Validate coverage and media integrity for one FlowWAM R250 variant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-width", type=int, default=640)
    ap.add_argument("--expected-height", type=int, default=480)
    args = ap.parse_args()

    rows = json.loads(args.manifest.read_text())
    expected = {f"{row['request_id']}.mp4" for row in rows}
    actual = {path.name for path in args.video_dir.glob("*.mp4")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    invalid: list[dict[str, object]] = []
    frame_counts: list[int] = []
    for name in sorted(expected & actual):
        path = args.video_dir / name
        cap = cv2.VideoCapture(str(path))
        opened = cap.isOpened()
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read() if opened else (False, None)
        cap.release()
        frame_counts.append(frames)
        if (not opened or not ok or frames <= 0 or width != args.expected_width
                or height != args.expected_height):
            invalid.append({
                "file": name, "opened": opened, "first_frame": bool(ok),
                "frames": frames, "width": width, "height": height,
            })
    report = {
        "manifest_rows": len(rows),
        "unique_expected": len(expected),
        "mp4_files": len(actual),
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
        "min_frames": min(frame_counts) if frame_counts else None,
        "max_frames": max(frame_counts) if frame_counts else None,
        "valid": not missing and not extra and not invalid and len(expected) == 250,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
