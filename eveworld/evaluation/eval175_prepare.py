#!/usr/bin/env python3
"""Audit EVAL-175 generations and build scorer manifests/staging trees."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2


GAGI_ROOT = Path("/data/datasets/gagi")
DEFAULT_GENERATION_ROOT = GAGI_ROOT / "eve_v2_outputs/eval175_gen"
DEFAULT_INPUT_ROOT = GAGI_ROOT / "gr1_dreamgen_eval/giga_input"
DEFAULT_OUTPUT_ROOT = GAGI_ROOT / "eve_v2_outputs/eval175_eval"
SPLIT_COUNTS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}
INDEX_RE = re.compile(r"^(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, default=DEFAULT_GENERATION_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", default="", help="Comma-separated model dirs; default discovers all complete-looking dirs")
    parser.add_argument("--expected-frames", type=int, default=93)
    parser.add_argument("--expected-width", type=int, default=768)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--expected-fps", type=float, default=16.0)
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--no-video-probe", action="store_true")
    parser.add_argument("--no-stage", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_models(root: Path, requested: str) -> list[str]:
    if requested.strip():
        return [item.strip() for item in requested.split(",") if item.strip()]
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def index_videos(video_dir: Path) -> tuple[dict[int, Path], list[str]]:
    indexed: dict[int, Path] = {}
    errors: list[str] = []
    for path in sorted(video_dir.glob("*.mp4")):
        match = INDEX_RE.match(path.name)
        if match is None:
            errors.append(f"unparseable video filename: {path}")
            continue
        index = int(match.group(1))
        if index in indexed:
            errors.append(f"duplicate index {index}: {indexed[index]} and {path}")
            continue
        indexed[index] = path.resolve()
    return indexed, errors


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
        "duration_sec": round(frames / fps, 6) if fps > 0 else None,
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


def stage_link(stage_root: Path, model: str, split: str, index: int, source: Path) -> Path:
    destination = stage_root / model / f"{split}__{index:03d}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return destination.absolute()
        destination.unlink()
    destination.symlink_to(source)
    return destination.absolute()


def main() -> None:
    args = parse_args()
    generation_root = args.generation_root.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    models = discover_models(generation_root, args.models)
    if not models:
        raise SystemExit(f"No model directories found under {generation_root}")

    report: dict[str, Any] = {
        "generation_root": str(generation_root),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "expected": {
            "splits": SPLIT_COUNTS,
            "total_per_model": sum(SPLIT_COUNTS.values()),
            "frames": args.expected_frames,
            "width": args.expected_width,
            "height": args.expected_height,
            "fps": args.expected_fps,
            "seed": args.expected_seed,
        },
        "models": {},
        "errors": [],
    }

    all_rows: list[dict[str, Any]] = []
    for model in models:
        model_rows: list[dict[str, Any]] = []
        model_errors: list[str] = []
        model_report: dict[str, Any] = {"splits": {}, "errors": model_errors}
        for split, expected_count in SPLIT_COUNTS.items():
            input_path = input_root / f"eval175_{split}.json"
            video_dir = generation_root / model / split / "generated_only"
            summary_path = generation_root / model / split / "generation_summary.json"
            split_errors: list[str] = []
            if not input_path.is_file():
                split_errors.append(f"missing input manifest: {input_path}")
                items: list[dict[str, Any]] = []
            else:
                items = load_json(input_path)
            if len(items) != expected_count:
                split_errors.append(f"input count={len(items)}, expected {expected_count}")
            if not video_dir.is_dir():
                split_errors.append(f"missing generated-only dir: {video_dir}")
                videos: dict[int, Path] = {}
            else:
                videos, filename_errors = index_videos(video_dir)
                split_errors.extend(filename_errors)

            expected_indices = set(range(len(items)))
            actual_indices = set(videos)
            missing = sorted(expected_indices - actual_indices)
            extra = sorted(actual_indices - expected_indices)
            if missing:
                split_errors.append(f"missing video indices: {missing}")
            if extra:
                split_errors.append(f"unexpected video indices: {extra}")

            summary: dict[str, Any] | None = None
            if summary_path.is_file():
                summary = load_json(summary_path)
                if summary.get("count") != expected_count:
                    split_errors.append(f"generation summary count={summary.get('count')}, expected {expected_count}")
                if summary.get("num_frames") != args.expected_frames:
                    split_errors.append(
                        f"generation summary num_frames={summary.get('num_frames')}, expected {args.expected_frames}"
                    )
                if summary.get("seed") != args.expected_seed:
                    split_errors.append(f"generation summary seed={summary.get('seed')}, expected {args.expected_seed}")
                summary_data_path = Path(str(summary.get("data_path", ""))).expanduser()
                if not summary_data_path.is_file() or summary_data_path.resolve() != input_path.resolve():
                    split_errors.append(f"generation summary data_path does not match {input_path}: {summary_data_path}")
            else:
                split_errors.append(f"missing generation summary: {summary_path}")

            split_rows: list[dict[str, Any]] = []
            for index, item in enumerate(items):
                video_path = videos.get(index)
                if video_path is None:
                    continue
                condition_image = Path(item["image"]).expanduser()
                if not condition_image.is_file():
                    split_errors.append(f"{model}/{split}/{index}: missing condition image {condition_image}")
                probe: dict[str, Any] | None = None
                if not args.no_video_probe:
                    try:
                        probe = probe_video(video_path)
                        split_errors.extend(validate_probe(probe, args, f"{model}/{split}/{index}"))
                    except Exception as exc:  # noqa: BLE001
                        split_errors.append(f"{model}/{split}/{index}: probe failed: {exc}")
                pa2_path = None
                if not args.no_stage:
                    pa2_path = stage_link(output_root / "pa2_stage", model, split, index, video_path)
                row = {
                    "key": f"{model}/{split}/{item['request_id']}",
                    "sample_key": f"{split}/{item['request_id']}",
                    "model": model,
                    "split": split,
                    "index": index,
                    "request_id": item["request_id"],
                    "prompt": item["prompt"],
                    "condition_image": str(condition_image),
                    "video_path": str(video_path),
                    "pa2_video_path": str(pa2_path) if pa2_path is not None else None,
                    "video": probe,
                }
                split_rows.append(row)

            split_manifest = output_root / "manifests" / model / f"{split}.jsonl"
            write_jsonl(split_manifest, split_rows)
            model_rows.extend(split_rows)
            model_errors.extend(f"{split}: {error}" for error in split_errors)
            model_report["splits"][split] = {
                "expected_count": expected_count,
                "video_count": len(videos),
                "manifest_count": len(split_rows),
                "manifest": str(split_manifest),
                "generation_summary": summary,
                "errors": split_errors,
            }

        model_manifest = output_root / "manifests" / f"{model}.jsonl"
        write_jsonl(model_manifest, model_rows)
        model_report.update(
            {
                "expected_count": sum(SPLIT_COUNTS.values()),
                "manifest_count": len(model_rows),
                "manifest": str(model_manifest),
                "pa2_stage": str(output_root / "pa2_stage" / model) if not args.no_stage else None,
                "ready": len(model_rows) == sum(SPLIT_COUNTS.values()) and not model_errors,
            }
        )
        report["models"][model] = model_report
        report["errors"].extend(f"{model}: {error}" for error in model_errors)
        all_rows.extend(model_rows)

    write_jsonl(output_root / "manifests" / "all_models.jsonl", all_rows)
    report["ready"] = not report["errors"]
    report["model_count"] = len(models)
    report["manifest_count"] = len(all_rows)
    report_path = output_root / "prepare_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(f"EVAL-175 preparation failed with {len(report['errors'])} error(s); see {report_path}")


if __name__ == "__main__":
    main()
