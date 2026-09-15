#!/usr/bin/env python3
"""Generate a frozen held-out shard with the joint GigaWorld-0 pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eveworld.method.heldout_io import clean_output_id, load_input_image, select_manifest_rows
from giga_datasets import image_utils
from giga_models import GigaWorld0Pipeline


NEGATIVE_PROMPT = (
    "The video captures ugly scenes, static motion, motion blur, low resolution, "
    "choppy or jerky movement, artifacts, unnatural transitions, jump cuts, "
    "visual noise, and flickering. Overall poor quality."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    rows = select_manifest_rows(
        args.data_path, args.split_manifest, args.expected_split, args.limit
    )
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]

    generated_dir = args.save_dir / "generated_only"
    side_by_side_dir = args.save_dir / "side_by_side"
    generated_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    pipeline = GigaWorld0Pipeline.from_pretrained(
        transformer_model_path=str(args.transformer),
        text_encoder_model_path=str(args.text_encoder),
        vae_model_path=str(args.vae),
        lora_model_path=str(args.lora) if args.lora else None,
        lora_fuse=bool(args.lora),
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")

    timings = []
    started = time.time()
    for row in rows:
        output_id = clean_output_id(row)
        generated_path = generated_dir / f"{output_id}.mp4"
        side_by_side_path = side_by_side_dir / f"{output_id}.mp4"
        if args.skip_existing and generated_path.exists() and side_by_side_path.exists():
            continue
        input_image = load_input_image(row, args.data_path, args.width, args.height)
        sample_started = time.time()
        frames = pipeline(
            prompt=row["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            image=input_image,
            num_inference_steps=args.num_inference_steps,
            fps=args.fps,
            num_frames=args.num_frames,
            height=input_image.height,
            width=input_image.width,
            seed=args.seed,
        )[0]
        if len(frames) != args.num_frames:
            raise RuntimeError(f"decoded {len(frames)} frames, expected {args.num_frames}")
        imageio.mimsave(generated_path, list(frames), fps=args.fps)
        side_by_side = [
            image_utils.concat_images_grid([input_image, frame], cols=2, pad=2)
            for frame in frames
        ]
        imageio.mimsave(side_by_side_path, side_by_side, fps=args.fps)
        elapsed = time.time() - sample_started
        timings.append({"output_id": output_id, "elapsed_sec": round(elapsed, 3)})
        print(f"[joint-gen] {output_id} {elapsed:.1f}s", flush=True)

    values = [row["elapsed_sec"] for row in timings]
    summary = {
        "data_path": str(args.data_path),
        "split_manifest": str(args.split_manifest),
        "expected_split": args.expected_split,
        "save_dir": str(args.save_dir),
        "lora": str(args.lora) if args.lora else None,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "count": len(timings),
        "wall_sec": round(time.time() - started, 3),
        "mean_sec": round(statistics.mean(values), 3) if values else None,
        "samples": timings,
    }
    (args.save_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
