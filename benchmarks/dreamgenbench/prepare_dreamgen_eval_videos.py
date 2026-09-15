#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg


DEFAULT_EVAL_ROOT = Path("/data/datasets/gagi/gr1_dreamgen_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop generated-only DreamGenBench videos from GigaWorld side-by-side outputs."
    )
    parser.add_argument(
        "--source-video-dir",
        type=Path,
        default=Path(__import__("os").environ.get("SOURCE_VIDEO_DIR", "")) if __import__("os").environ.get("SOURCE_VIDEO_DIR") else None,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def crop_right_half(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    if tmp.exists():
        tmp.unlink()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vf",
        "crop=iw/2:ih:iw/2:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(tmp),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not tmp.exists() or tmp.stat().st_size <= 1024:
        raise RuntimeError(f"ffmpeg did not produce a valid output video: {tmp}")
    tmp.replace(dst)


def main() -> None:
    args = parse_args()
    if args.source_video_dir is None:
        raise SystemExit("Pass --source-video-dir or set SOURCE_VIDEO_DIR.")
    source_dir = args.source_video_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Missing source video dir: {source_dir}")

    eval_root = args.eval_root.expanduser().resolve()
    output_dir = (args.output_dir or eval_root / "dreamgenbench_video_dirs" / source_dir.name).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(source_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files found in {source_dir}")

    for src in videos:
        crop_right_half(src, output_dir / src.name, overwrite=args.overwrite)

    summary = {
        "source_video_dir": str(source_dir),
        "output_dir": str(output_dir),
        "video_count": len(videos),
    }
    summary_path = output_dir.with_name(output_dir.name + ".prepare_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

