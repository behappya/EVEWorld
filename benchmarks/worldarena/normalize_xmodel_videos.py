#!/usr/bin/env python3
"""Normalize external-model videos to the WorldArena 1.0 evaluation contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import imageio_ffmpeg


def probe(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return width, height, frames, fps


def normalize_one(
    source: Path,
    destination: Path,
    *,
    width: int,
    height: int,
    frames: int,
    fps: int,
) -> tuple[str, str, dict[str, float | int]]:
    src_width, src_height, src_frames, src_fps = probe(source)
    source_info = {
        "width": src_width,
        "height": src_height,
        "frames": src_frames,
        "fps": src_fps,
    }
    if src_frames < frames:
        return source.name, f"too-short:{src_frames}<{frames}", source_info
    if destination.exists():
        try:
            if probe(destination) == (width, height, frames, float(fps)):
                return source.name, "skip", source_info
        except RuntimeError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height},trim=end_frame={frames},setpts=N/({fps}*TB)",
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        actual = probe(temporary)
        expected = (width, height, frames, float(fps))
        if actual != expected:
            raise RuntimeError(f"normalized metadata {actual} != {expected}")
        os.replace(temporary, destination)
        return source.name, "ok", source_info
    except Exception as error:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        return source.name, f"error:{type(error).__name__}:{error}", source_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(manifest) != args.expected_count:
        raise ValueError(f"manifest has {len(manifest)} rows, expected {args.expected_count}")
    jobs = []
    for row in manifest:
        name = f"{row['request_id']}.mp4"
        source = args.input_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        jobs.append((source, args.output_dir / name))

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                normalize_one,
                source,
                destination,
                width=args.width,
                height=args.height,
                frames=args.frames,
                fps=args.fps,
            ): source
            for source, destination in jobs
        }
        for future in as_completed(futures):
            name, status, source_info = future.result()
            results.append({"video": name, "status": status, "source": source_info})
            print(f"{name}: {status}", flush=True)

    failures = [row for row in results if row["status"] not in {"ok", "skip"}]
    payload = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "expected_count": args.expected_count,
        "normalized": len(results) - len(failures),
        "failed": len(failures),
        "target": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": args.fps,
        },
        "source_formats": sorted(
            {tuple(row["source"].values()) for row in results}
        ),
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("normalized", "failed", "source_formats")}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
