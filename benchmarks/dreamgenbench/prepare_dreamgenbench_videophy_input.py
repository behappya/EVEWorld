#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_VIDEO_DIR = Path("/data/datasets/gagi/gr1_dreamgen_eval/dreamgenbench_video_dirs/gr1_dreamgen_8gpu_full_20260625_212933")
DEFAULT_OUTPUT_ROOT = Path("/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs/videophy")
PROMPT_PHYSICS = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Does this video follow the physical laws?
AI: """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VideoPhy PA-II input CSVs for DreamGenBench videos.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def prompt_from_video_path(path: Path) -> str:
    stem = path.stem
    prompt = stem.split("_", 1)[1] if "_" in stem else stem
    return prompt.replace("_", " ")


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("**/*.mp4"))
    if args.start_offset:
        videos = videos[args.start_offset :]
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No mp4 files found in {video_dir}")

    base_csv = output_root / f"{args.run_name}_videophy_base.csv"
    physics_csv = output_root / f"{args.run_name}_physics_testing.csv"
    summary_json = output_root / f"{args.run_name}_videophy_prepare_summary.json"

    with base_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["videopath", "caption"])
        writer.writeheader()
        for path in videos:
            writer.writerow({"videopath": str(path), "caption": prompt_from_video_path(path)})

    with physics_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["videopath", "caption"])
        writer.writeheader()
        for path in videos:
            writer.writerow({"videopath": str(path), "caption": PROMPT_PHYSICS})

    payload = {
        "video_dir": str(video_dir),
        "video_count": len(videos),
        "base_csv": str(base_csv),
        "physics_testing_csv": str(physics_csv),
        "note": "physics_testing_csv matches VideoPhy prepare_data.py PA-II physics prompt.",
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
