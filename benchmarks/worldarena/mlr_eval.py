#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


TRACK4GEN = Path(__file__).resolve().parents[2] / "eveworld" / "pipeline"
sys.path.insert(0, str(TRACK4GEN))

from t4g_detect import detect_all  # noqa: E402
from t4g_exam_v2 import count_valid_instances  # noqa: E402
from t4g_gdino import GDinoLocator  # noqa: E402


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sample_frames(path: Path, frame_count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decodable frames: {path}")
    indices = np.linspace(0, len(frames) - 1, frame_count).astype(int)
    return [frames[index] for index in indices]


def count_instances(locator: GDinoLocator, frame: np.ndarray, mover: str) -> int:
    objects = detect_all(locator, frame, mover, topk=6, box_thr=0.35)
    grippers = detect_all(locator, frame, "robot gripper", topk=3, box_thr=0.15)
    return count_valid_instances(objects, grippers, min_dist=60, grip_overlap_thr=0.35)


def evaluate_video(
    locator: GDinoLocator,
    video: Path,
    condition_image: Path,
    mover: str,
    frame_count: int,
    inventory_cache: dict[tuple[str, str], int],
) -> dict[str, Any]:
    cache_key = (str(condition_image), mover)
    if cache_key not in inventory_cache:
        initial = np.asarray(Image.open(condition_image).convert("RGB"))
        inventory_cache[cache_key] = count_instances(locator, initial, mover)
    initial_count = inventory_cache[cache_key]
    if initial_count == 0:
        return {
            "eligible": False,
            "initial_count": 0,
            "counts": [],
            "mlr_event": False,
            "max_count": 0,
        }

    counts = [
        count_instances(locator, frame, mover)
        for frame in sample_frames(video, frame_count)
    ]
    consecutive = 0
    max_consecutive = 0
    for count in counts:
        consecutive = consecutive + 1 if count >= initial_count + 1 else 0
        max_consecutive = max(max_consecutive, consecutive)
    return {
        "eligible": True,
        "initial_count": initial_count,
        "counts": counts,
        "mlr_event": max_consecutive >= 2,
        "max_count": max(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WorldArena 1.0 MLR on one shard")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    jobs = jobs[args.shard_index :: args.num_shards]
    locator = GDinoLocator(device="cuda")
    inventory_cache: dict[tuple[str, str], int] = {}
    records = []
    for index, job in enumerate(jobs, start=1):
        record = {
            "model": job["model"],
            "request_id": job["request_id"],
            "mover": job["mover"],
            "video": job["video"],
            "condition_image": job["condition_image"],
        }
        try:
            record.update(
                evaluate_video(
                    locator,
                    Path(job["video"]),
                    Path(job["condition_image"]),
                    job["mover"],
                    args.frame_count,
                    inventory_cache,
                )
            )
            record["error"] = None
        except Exception as error:  # Preserve item-level failures for the coverage gate.
            record.update(
                {
                    "eligible": False,
                    "initial_count": None,
                    "counts": [],
                    "mlr_event": False,
                    "max_count": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        records.append(record)
        if index % 20 == 0 or index == len(jobs):
            print(f"shard={args.shard_index} progress={index}/{len(jobs)}", flush=True)
    atomic_write_json(
        args.output,
        {
            "metadata": {
                "protocol": "worldarena1_mlr_gdino_v2",
                "frame_count": args.frame_count,
                "object_threshold": 0.35,
                "gripper_threshold": 0.15,
                "gripper_overlap_threshold": 0.35,
                "minimum_center_distance": 60,
                "consecutive_frames": 2,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
