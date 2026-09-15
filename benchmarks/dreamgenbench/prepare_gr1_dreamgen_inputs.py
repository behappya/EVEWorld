#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2


DEFAULT_MANIFEST = Path("/data/datasets/gagi/gr1_finetune_data/raw_data/manifest.jsonl")
DEFAULT_OUTPUT_ROOT = Path("/data/datasets/gagi/gr1_dreamgen_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GR1 image-to-video inputs for DreamGenBench-style evaluation."
    )
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite-frames", action="store_true")
    return parser.parse_args()


def clean_prompt_for_filename(prompt: str, max_len: int = 180) -> str:
    text = prompt.strip().rstrip(".")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "instruction"
    return text[:max_len].rstrip("_")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_first_frame(video_path: Path, image_path: Path, overwrite: bool) -> None:
    if image_path.exists() and not overwrite:
        return
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Could not write frame to {image_path}")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_jsonl.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    frames_dir = output_root / "first_frames"
    output_json = (args.output_json or output_root / "giga_input" / "gr1_dreamgen_it2v.json").expanduser().resolve()
    metadata_jsonl = output_json.with_suffix(".metadata.jsonl")

    rows = read_manifest(manifest_path)
    rows = rows[: args.limit] if args.limit > 0 else rows

    output: list[dict[str, Any]] = []
    output_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with metadata_jsonl.open("w", encoding="utf-8") as meta_fh:
        for idx, row in enumerate(rows):
            video_path = Path(str(row["video"])).expanduser().resolve()
            prompt = str(row["text"]).strip()
            stem = Path(str(row.get("file_name") or video_path.name)).stem
            first_frame = frames_dir / f"{stem}.jpg"
            write_first_frame(video_path, first_frame, overwrite=args.overwrite_frames)

            request_id = f"{idx}_{clean_prompt_for_filename(prompt)}"
            item = {
                "request_id": request_id,
                "prompt": prompt,
                "image": str(first_frame),
                "source_video": str(video_path),
                "source_file_name": row.get("file_name") or video_path.name,
            }
            output.append(item)
            meta_fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "manifest_jsonl": str(manifest_path),
        "output_json": str(output_json),
        "metadata_jsonl": str(metadata_jsonl),
        "first_frames_dir": str(frames_dir),
        "sample_count": len(output),
    }
    (output_root / "prepare_gr1_dreamgen_inputs_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

