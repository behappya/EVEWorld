#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg


DEFAULT_METADATA_JSONL = Path("/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl")
DEFAULT_SOURCE_VIDEO_DIR = Path("/data/datasets/gagi/giga_world_0_outputs/pbench_robot_serving_full_20260612_145541")
DEFAULT_OUTPUT_ROOT = Path("/data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality")

QUALITY_DIMS = [
    "i2v_subject",
    "i2v_background",
    "aesthetic_quality",
    "imaging_quality",
    "background_consistency",
    "motion_smoothness",
    "subject_consistency",
    "overall_consistency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare generated-only PBench Robot videos for VBench quality eval.")
    parser.add_argument("--metadata-jsonl", type=Path, default=DEFAULT_METADATA_JSONL)
    parser.add_argument("--source-video-dir", type=Path, default=DEFAULT_SOURCE_VIDEO_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generated-video-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--resolution-name", default="pbench_robot")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-images", action="store_true", help="Copy input images instead of symlinking them.")
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def crop_right_half_video(src: Path, dst: Path, overwrite: bool) -> None:
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
        raise ValueError(f"ffmpeg did not produce a valid output video: {tmp}")
    tmp.replace(dst)


def link_or_copy_image(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    generated_video_dir = (args.generated_video_dir or output_root / "generated_only_videos").expanduser().resolve()
    work_dir = (args.work_dir or output_root / "vbench_work").expanduser().resolve()
    image_root = work_dir / "vbench2_beta_i2v" / "data" / "crop" / args.resolution_name
    full_info_path = output_root / "pbench_robot_vbench_full_info.json"

    rows = load_samples(args.metadata_jsonl.expanduser().resolve())
    if args.limit > 0:
        rows = rows[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    generated_video_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    full_info: list[dict[str, Any]] = []
    for row in rows:
        pbench_id = str(row["pbench_id"])
        src_video = args.source_video_dir.expanduser().resolve() / f"{pbench_id}.mp4"
        dst_video = generated_video_dir / f"{pbench_id}.mp4"
        if not src_video.exists():
            raise FileNotFoundError(src_video)
        crop_right_half_video(src_video, dst_video, overwrite=args.overwrite)

        image_path = Path(str(row["image"]))
        if not image_path.is_absolute():
            image_path = args.metadata_jsonl.expanduser().resolve().parent / image_path
        image_name = f"{pbench_id}{image_path.suffix or '.jpg'}"
        link_or_copy_image(image_path.resolve(), image_root / image_name, copy_images=args.copy_images)

        full_info.append(
            {
                "prompt_en": row.get("prompt") or "",
                "dimension": QUALITY_DIMS,
                "video_list": [str(dst_video)],
                "image_name": image_name,
                "pbench_id": pbench_id,
            }
        )

    full_info_path.write_text(json.dumps(full_info, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "metadata_jsonl": str(args.metadata_jsonl),
        "source_video_dir": str(args.source_video_dir),
        "generated_video_dir": str(generated_video_dir),
        "work_dir": str(work_dir),
        "image_root": str(image_root),
        "resolution_name": args.resolution_name,
        "full_info_path": str(full_info_path),
        "sample_count": len(full_info),
        "dimensions": QUALITY_DIMS,
    }
    (output_root / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
