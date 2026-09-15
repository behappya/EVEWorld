#!/usr/bin/env python3
"""Generate several seeds for one DreamGenBench item on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))


NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated, "
    "poorly lit, washed out colors, choppy, jerky movements, artifacting, unnatural transitions, "
    "fake elements, visual noise, flickering. Overall poor quality."
)


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_single_item(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError(f"{path} must contain a one-item JSON list")
    item = data[0]
    if not item.get("prompt") or not item.get("image"):
        raise ValueError(f"{path} item must contain prompt and image")
    return item


def save_video_atomic(imageio, path: Path, frames: list, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.mp4")
    if temporary.exists():
        temporary.unlink()
    imageio.mimsave(temporary, frames, fps=fps)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("worker seeds must be unique")
    item = load_single_item(args.input_json)
    image_path = Path(item["image"])
    if not image_path.is_file():
        candidate = args.input_json.parent / image_path
        if not candidate.is_file():
            raise FileNotFoundError(image_path)
        image_path = candidate

    generated_dir = args.save_root / "generated_only"
    side_by_side_dir = args.save_root / "side_by_side"
    generated_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = []
    for seed in args.seeds:
        generated_path = generated_dir / f"seed{seed}.mp4"
        side_by_side_path = side_by_side_dir / f"seed{seed}.mp4"
        complete = (
            generated_path.is_file()
            and generated_path.stat().st_size > 0
            and side_by_side_path.is_file()
            and side_by_side_path.stat().st_size > 0
        )
        if args.skip_existing and complete:
            skipped.append(seed)
        else:
            pending.append(seed)

    started_at = time.time()
    timings = []
    if pending:
        import imageio.v2 as imageio
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as functional

        from eveworld.method.pipeline_eag import EAGGigaWorld0Pipeline
        from giga_datasets import image_utils

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("DreamGen multi-seed inference requires CUDA")
        print(
            f"[dreamgen-worker] worker={args.worker_id} seeds={args.seeds} "
            f"pending={pending} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
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
        pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        original_image = Image.open(image_path).convert("RGB")
        image_width, image_height = original_image.size
        dst_width, dst_height = image_utils.get_image_size(
            (image_width, image_height),
            (args.width, args.height),
            mode="area",
            multiple=16,
        )
        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))
        x1 = (new_width - dst_width) // 2
        y1 = (new_height - dst_height) // 2
        input_image = functional.resize(
            original_image, (new_height, new_width), InterpolationMode.BILINEAR
        )
        input_image = functional.crop(input_image, y1, x1, dst_height, dst_width)

        for seed in pending:
            seed_started = time.time()
            print(f"[dreamgen-worker] worker={args.worker_id} seed={seed} START", flush=True)
            output_frames = pipe(
                prompt=item["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                image=input_image,
                num_inference_steps=args.num_inference_steps,
                fps=args.fps,
                num_frames=args.num_frames,
                height=dst_height,
                width=dst_width,
                seed=seed,
            )[0]
            generated_path = generated_dir / f"seed{seed}.mp4"
            side_by_side_path = side_by_side_dir / f"seed{seed}.mp4"
            save_video_atomic(imageio, generated_path, list(output_frames), args.fps)
            side_by_side_frames = [
                image_utils.concat_images_grid([input_image, frame], cols=2, pad=2)
                for frame in output_frames
            ]
            save_video_atomic(imageio, side_by_side_path, side_by_side_frames, args.fps)
            elapsed = round(time.time() - seed_started, 3)
            timings.append(
                {
                    "seed": seed,
                    "elapsed_sec": elapsed,
                    "generated_only": str(generated_path),
                    "side_by_side": str(side_by_side_path),
                }
            )
            print(
                f"[dreamgen-worker] worker={args.worker_id} seed={seed} DONE "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed_values = [item["elapsed_sec"] for item in timings]
    summary = {
        "worker_id": args.worker_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "assigned_seeds": args.seeds,
        "generated_seeds": [item["seed"] for item in timings],
        "skipped_seeds": skipped,
        "prompt": item["prompt"],
        "image": str(image_path),
        "transformer": str(args.transformer.resolve()),
        "text_encoder": str(args.text_encoder.resolve()),
        "vae": str(args.vae.resolve()),
        "num_inference_steps": args.num_inference_steps,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "height": args.height,
        "width": args.width,
        "wall_sec": round(time.time() - started_at, 3),
        "mean_generation_sec": round(statistics.mean(elapsed_values), 3) if elapsed_values else None,
        "outputs": timings,
    }
    atomic_json_dump(args.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
