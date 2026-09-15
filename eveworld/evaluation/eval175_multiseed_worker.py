#!/usr/bin/env python3
"""Generate all 126 DreamGenBench videos for several seeds on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

SPLIT_COUNTS = {"gr1_env": 29, "gr1_object": 50, "gr1_behavior": 47}
NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated, "
    "poorly lit, washed out colors, choppy, jerky movements, artifacting, unnatural transitions, "
    "fake elements, visual noise, flickering. Overall poor quality."
)


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def save_video_atomic(imageio: Any, path: Path, frames: list[Any], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.partial.mp4")
    if temporary.exists():
        temporary.unlink()
    imageio.mimsave(temporary, frames, fps=fps)
    os.replace(temporary, path)


def output_id(index: int, prompt: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in prompt)[:80]
    return f"{index}_{sanitized}"


def load_items(data_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for split, expected in SPLIT_COUNTS.items():
        path = data_root / f"eval175_{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != expected:
            raise ValueError(f"{path}: expected {expected} rows, found {len(data)}")
        for index, item in enumerate(data):
            prompt = str(item.get("prompt") or "")
            image = Path(str(item.get("image") or ""))
            if not prompt or not image.is_file():
                raise ValueError(f"{path}:{index}: invalid prompt or image {image}")
            items.append(
                {
                    "split": split,
                    "index": index,
                    "prompt": prompt,
                    "image": image,
                    "output_id": output_id(index, prompt),
                }
            )
    if len(items) != 126:
        raise ValueError(f"expected 126 total items, found {len(items)}")
    return items


def output_paths(save_root: Path, seed: int, item: dict[str, Any]) -> tuple[Path, Path]:
    split_root = save_root / f"seed{seed:03d}" / item["split"]
    name = f"{item['output_id']}.mp4"
    return split_root / "generated_only" / name, split_root / "side_by_side" / name


def file_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def seed_output_counts(save_root: Path, seed: int) -> dict[str, dict[str, int]]:
    seed_root = save_root / f"seed{seed:03d}"
    return {
        split: {
            "generated_only": len(list((seed_root / split / "generated_only").glob("*.mp4"))),
            "side_by_side": len(list((seed_root / split / "side_by_side").glob("*.mp4"))),
            "expected": expected,
        }
        for split, expected in SPLIT_COUNTS.items()
    }


def seed_complete(save_root: Path, seed: int, items: list[dict[str, Any]]) -> bool:
    return all(
        file_complete(path)
        for item in items
        for path in output_paths(save_root, seed, item)
    )


def preprocess_image(
    image_path: Path,
    *,
    width: int,
    height: int,
    image_utils: Any,
    image_class: Any,
    interpolation_mode: Any,
    functional: Any,
) -> Any:
    image = image_class.open(image_path).convert("RGB")
    image_width, image_height = image.size
    dst_width, dst_height = image_utils.get_image_size(
        (image_width, image_height), (width, height), mode="area", multiple=16
    )
    if float(dst_height) / image_height < float(dst_width) / image_width:
        new_height = int(round(float(dst_width) / image_width * image_height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / image_height * image_width))
    x1 = (new_width - dst_width) // 2
    y1 = (new_height - dst_height) // 2
    resized = functional.resize(
        image, (new_height, new_width), interpolation_mode.BILINEAR
    )
    return functional.crop(resized, y1, x1, dst_height, dst_width)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("worker seeds must be non-empty and unique")
    if args.task_shard_count < 1:
        raise ValueError("task-shard-count must be positive")
    if not 0 <= args.task_shard_index < args.task_shard_count:
        raise ValueError("task-shard-index must be in [0, task-shard-count)")
    all_items = load_items(args.data_root)
    items = all_items[args.task_shard_index :: args.task_shard_count]
    if not items:
        raise ValueError("task shard is empty")
    pending_seeds = [
        seed
        for seed in args.seeds
        if not (args.skip_existing and seed_complete(args.save_root, seed, items))
    ]
    skipped_seeds = [seed for seed in args.seeds if seed not in pending_seeds]
    started_at = time.time()
    seed_summaries: list[dict[str, Any]] = []

    if pending_seeds:
        import imageio.v2 as imageio
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional

        from eveworld.method.pipeline_eag import EAGGigaWorld0Pipeline
        from giga_datasets import image_utils

        if not torch.cuda.is_available():
            raise RuntimeError("EVAL-175 multi-seed inference requires CUDA")
        print(
            f"[seed-worker] worker={args.worker_id} host={socket.gethostname()} "
            f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES')} assigned={args.seeds} "
            f"pending={pending_seeds} task_shard="
            f"{args.task_shard_index}/{args.task_shard_count} items={len(items)}",
            flush=True,
        )
        pipe = EAGGigaWorld0Pipeline.from_pretrained(
            transformer_model_path=str(args.transformer),
            text_encoder_model_path=str(args.text_encoder),
            vae_model_path=str(args.vae),
            lora_model_path=None,
            lora_fuse=False,
            physics_latent_model_path=None,
        )
        pipe.to("cuda")
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        image_cache: dict[Path, Any] = {}
        for seed in pending_seeds:
            seed_started = time.time()
            generated = 0
            skipped = 0
            timings: list[float] = []
            print(
                f"[seed-worker] worker={args.worker_id} seed={seed} START "
                f"{len(items)} tasks shard={args.task_shard_index}/"
                f"{args.task_shard_count}",
                flush=True,
            )
            for position, item in enumerate(items, start=1):
                generated_path, side_by_side_path = output_paths(args.save_root, seed, item)
                if (
                    args.skip_existing
                    and file_complete(generated_path)
                    and file_complete(side_by_side_path)
                ):
                    skipped += 1
                    continue
                image_path = item["image"]
                input_image = image_cache.get(image_path)
                if input_image is None:
                    input_image = preprocess_image(
                        image_path,
                        width=args.width,
                        height=args.height,
                        image_utils=image_utils,
                        image_class=Image,
                        interpolation_mode=InterpolationMode,
                        functional=functional,
                    )
                    image_cache[image_path] = input_image

                sample_started = time.time()
                output_frames = list(
                    pipe(
                        prompt=item["prompt"],
                        negative_prompt=NEGATIVE_PROMPT,
                        image=input_image,
                        num_inference_steps=args.num_inference_steps,
                        fps=args.fps,
                        num_frames=args.num_frames,
                        height=args.height,
                        width=args.width,
                        seed=seed,
                    )[0]
                )
                save_video_atomic(imageio, generated_path, output_frames, args.fps)
                side_by_side_frames = [
                    image_utils.concat_images_grid([input_image, frame], cols=2, pad=2)
                    for frame in output_frames
                ]
                save_video_atomic(imageio, side_by_side_path, side_by_side_frames, args.fps)
                elapsed = time.time() - sample_started
                timings.append(elapsed)
                generated += 1
                print(
                    f"[seed-worker] worker={args.worker_id} seed={seed} "
                    f"{position}/126 {item['split']}/{item['index']} {elapsed:.1f}s",
                    flush=True,
                )

            if not seed_complete(args.save_root, seed, items):
                raise RuntimeError(
                    f"seed {seed} incomplete after generation: "
                    f"{seed_output_counts(args.save_root, seed)}"
                )
            completed_at = time.time()
            summary = {
                "status": "complete",
                "seed": seed,
                "worker_id": args.worker_id,
                "host": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "task_shard_index": args.task_shard_index,
                "task_shard_count": args.task_shard_count,
                "task_count": len(items),
                "generated_this_run": generated,
                "skipped_this_run": skipped,
                "counts": seed_output_counts(args.save_root, seed),
                "wall_sec": round(completed_at - seed_started, 3),
                "mean_generation_sec": round(statistics.mean(timings), 3)
                if timings
                else None,
                "parameters": {
                    "num_inference_steps": args.num_inference_steps,
                    "num_frames": args.num_frames,
                    "fps": args.fps,
                    "height": args.height,
                    "width": args.width,
                    "eag_weight": 0,
                },
                "model": {
                    "transformer": str(args.transformer.resolve()),
                    "text_encoder": str(args.text_encoder.resolve()),
                    "vae": str(args.vae.resolve()),
                },
                "completed_at_unix": completed_at,
            }
            seed_root = args.save_root / f"seed{seed:03d}"
            if args.task_shard_count == 1:
                atomic_json_dump(seed_root / "generation_summary.json", summary)
                atomic_json_dump(seed_root / "_COMPLETE.json", summary)
            seed_summaries.append(summary)
            print(
                f"[seed-worker] worker={args.worker_id} seed={seed} "
                f"{'COMPLETE' if args.task_shard_count == 1 else 'SHARD_COMPLETE'} "
                f"wall={summary['wall_sec']:.1f}s",
                flush=True,
            )

    worker_summary = {
        "status": "complete",
        "worker_id": args.worker_id,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "task_shard_index": args.task_shard_index,
        "task_shard_count": args.task_shard_count,
        "task_count": len(items),
        "assigned_seeds": args.seeds,
        "generated_seeds": [item["seed"] for item in seed_summaries],
        "skipped_complete_seeds": skipped_seeds,
        "wall_sec": round(time.time() - started_at, 3),
        "seed_summaries": [
            str(args.save_root / f"seed{item['seed']:03d}" / "generation_summary.json")
            for item in seed_summaries
        ],
    }
    atomic_json_dump(args.summary_path, worker_summary)
    print(json.dumps(worker_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
