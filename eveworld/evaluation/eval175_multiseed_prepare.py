#!/usr/bin/env python3
"""Audit EVAL-175 seed directories and build one Gemini manifest per seed."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2


SPLIT_COUNTS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}
INDEX_RE = re.compile(r"^(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default="",
        help="Stable model label used in manifest keys; defaults to seedNNN for compatibility.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, default=93)
    parser.add_argument("--expected-width", type=int, default=768)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--expected-fps", type=float, default=16.0)
    parser.add_argument("--no-video-probe", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def index_videos(path: Path) -> tuple[dict[int, Path], list[str]]:
    videos: dict[int, Path] = {}
    errors: list[str] = []
    for video in sorted(path.glob("*.mp4")):
        match = INDEX_RE.match(video.name)
        if match is None:
            errors.append(f"unparseable filename: {video}")
            continue
        index = int(match.group(1))
        if index in videos:
            errors.append(f"duplicate index {index}: {videos[index]} and {video}")
        else:
            videos[index] = video.resolve()
    return videos, errors


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video")
    try:
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    return {
        "frames": frames,
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "size_bytes": path.stat().st_size,
    }


def validate_probe(probe: dict[str, Any], args: argparse.Namespace, label: str) -> list[str]:
    errors: list[str] = []
    for field, expected in (
        ("frames", args.expected_frames),
        ("width", args.expected_width),
        ("height", args.expected_height),
    ):
        if probe[field] != expected:
            errors.append(f"{label}: {field}={probe[field]}, expected {expected}")
    if not math.isclose(probe["fps"], args.expected_fps, rel_tol=0.0, abs_tol=0.05):
        errors.append(f"{label}: fps={probe['fps']}, expected {args.expected_fps}")
    if probe["size_bytes"] <= 0:
        errors.append(f"{label}: empty video")
    return errors


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    generation_root = args.generation_root.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    inputs: dict[str, list[dict[str, Any]]] = {}
    for split, expected in SPLIT_COUNTS.items():
        path = input_root / f"eval175_{split}.json"
        rows = load_json(path)
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError(f"{path}: expected {expected} rows")
        inputs[split] = rows

    report: dict[str, Any] = {
        "generation_root": str(generation_root),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_name": args.model_name or None,
        "seeds": args.seeds,
        "expected_per_seed": 126,
        "expected_total": 126 * len(args.seeds),
        "seed_reports": {},
        "errors": [],
    }
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_name = f"seed{seed:03d}"
        model_name = args.model_name or seed_name
        seed_root = generation_root / seed_name
        seed_errors: list[str] = []
        if not (seed_root / "_COMPLETE.json").is_file():
            seed_errors.append(f"missing completion marker: {seed_root / '_COMPLETE.json'}")
        seed_rows: list[dict[str, Any]] = []
        for split, expected in SPLIT_COUNTS.items():
            video_dir = seed_root / split / "generated_only"
            videos, index_errors = index_videos(video_dir)
            seed_errors.extend(f"{split}: {error}" for error in index_errors)
            expected_indices = set(range(expected))
            actual_indices = set(videos)
            if actual_indices != expected_indices:
                seed_errors.append(
                    f"{split}: indices mismatch; missing={sorted(expected_indices - actual_indices)} "
                    f"extra={sorted(actual_indices - expected_indices)}"
                )
            for index, item in enumerate(inputs[split]):
                video_path = videos.get(index)
                if video_path is None:
                    continue
                probe = None
                if not args.no_video_probe:
                    try:
                        probe = probe_video(video_path)
                        seed_errors.extend(
                            validate_probe(probe, args, f"{seed_name}/{split}/{index}")
                        )
                    except Exception as exc:
                        seed_errors.append(f"{seed_name}/{split}/{index}: probe failed: {exc}")
                key = (
                    f"{model_name}/{seed_name}/{split}/{item['request_id']}"
                    if args.model_name
                    else f"{seed_name}/{split}/{item['request_id']}"
                )
                seed_rows.append(
                    {
                        "key": key,
                        "sample_key": f"{split}/{item['request_id']}",
                        "model": model_name,
                        "inference_seed": seed,
                        "split": split,
                        "index": index,
                        "request_id": item["request_id"],
                        "prompt": item["prompt"],
                        "condition_image": item["image"],
                        "video_path": str(video_path),
                        "video": probe,
                    }
                )
        manifest_path = output_root / "manifests" / f"{seed_name}.jsonl"
        write_jsonl(manifest_path, seed_rows)
        ready = len(seed_rows) == 126 and not seed_errors
        report["seed_reports"][seed_name] = {
            "ready": ready,
            "manifest": str(manifest_path),
            "manifest_count": len(seed_rows),
            "errors": seed_errors,
        }
        report["errors"].extend(f"{seed_name}: {error}" for error in seed_errors)
        all_rows.extend(seed_rows)

    write_jsonl(output_root / "manifests" / "all_seeds.jsonl", all_rows)
    report["manifest_count"] = len(all_rows)
    report["ready"] = not report["errors"] and len(all_rows) == report["expected_total"]
    report_path = output_root / "prepare_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(f"multi-seed preparation failed; see {report_path}")


if __name__ == "__main__":
    main()
