#!/usr/bin/env python3
"""One-GPU worker for a checkpoint's four CFG values.

The checkpoint is loaded once.  Each worker receives a deterministic task shard
and writes one generated-only mp4 per (CFG, prompt) key.  A parent dispatcher
controls checkpoint-level serial execution.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated, poorly lit, washed out colors, choppy, jerky movements, artifacting, "
    "unnatural transitions, fake elements, visual noise, flickering. Overall poor quality."
)


def complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 10_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--training-step", type=int, required=True)
    ap.add_argument("--cfg-values", nargs="+", type=float, required=True)
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--worker-count", type=int, required=True)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=93)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--pythonpath", default="")
    args = ap.parse_args()

    if args.worker_id < 0 or args.worker_id >= args.worker_count:
        raise ValueError("invalid worker id")
    if not (args.model_dir / "transformer" / "config.json").is_file():
        raise FileNotFoundError(args.model_dir / "transformer" / "config.json")
    if not (args.model_dir / "text_encoder" / "config.json").is_file():
        raise FileNotFoundError(args.model_dir / "text_encoder" / "config.json")
    if not (args.model_dir / "vae" / "config.json").is_file():
        raise FileNotFoundError(args.model_dir / "vae" / "config.json")
    if args.pythonpath:
        sys.path[:0] = [p for p in args.pythonpath.split(os.pathsep) if p]

    import imageio.v2 as imageio
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F
    from giga_datasets import image_utils
    from eveworld.method.pipeline_eag import EAGGigaWorld0Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CFG generation requires CUDA")
    if args.data_root.is_dir():
        rows = []
        for split in ("env", "object", "behavior"):
            path = args.data_root / f"eval175_gr1_{split}.json"
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    else:
        rows = json.loads(args.data_root.read_text(encoding="utf-8"))
    if len(rows) != 126:
        raise ValueError(f"expected 126 DreamGen rows, got {len(rows)}")
    rows = rows[args.worker_id :: args.worker_count]
    pipe = EAGGigaWorld0Pipeline.from_pretrained(
        transformer_model_path=str(args.model_dir / "transformer"),
        text_encoder_model_path=str(args.model_dir / "text_encoder"),
        vae_model_path=str(args.model_dir / "vae"),
        lora_model_path=None,
        lora_fuse=False,
        physics_latent_model_path=None,
    )
    pipe.to("cuda")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    image_cache: dict[str, Image.Image] = {}
    print(f"CFG_WORKER_START id={args.worker_id}/{args.worker_count} "
          f"rows={len(rows)} step={args.training_step} cfg={args.cfg_values}", flush=True)
    for cfg in args.cfg_values:
        cfg_tag = f"cfg_{cfg:g}".replace(".", "p")
        out_dir = args.out_root / f"step_{args.training_step:03d}" / cfg_tag / "generated_only"
        out_dir.mkdir(parents=True, exist_ok=True)
        timings: list[float] = []
        for item in rows:
            output_id = str(item.get("request_id") or item.get("id") or "")
            if not output_id:
                raise ValueError(f"missing request id: {item}")
            out_path = out_dir / f"{output_id}.mp4"
            if complete(out_path):
                continue
            image_path = Path(item["image"])
            if not image_path.is_file():
                image_path = args.data_root.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            key = str(image_path)
            if key not in image_cache:
                image = Image.open(image_path).convert("RGB")
                dw, dh = image_utils.get_image_size(
                    (image.width, image.height), (args.width, args.height), mode="area", multiple=16
                )
                if float(dh) / image.height < float(dw) / image.width:
                    nh, nw = int(round(float(dw) / image.width * image.height)), dw
                else:
                    nh, nw = dh, int(round(float(dh) / image.height * image.width))
                x1, y1 = (nw - dw) // 2, (nh - dh) // 2
                image = F.crop(F.resize(image, (nh, nw), InterpolationMode.BILINEAR), y1, x1, dh, dw)
                image_cache[key] = image
            started = time.time()
            frames = pipe(
                prompt=item["prompt"], negative_prompt=NEGATIVE_PROMPT,
                image=image_cache[key], guidance_scale=cfg,
                num_inference_steps=args.steps, fps=args.fps,
                num_frames=args.frames, height=args.height, width=args.width,
                seed=args.seed,
            )[0]
            # Keep the video suffix so imageio selects the ffmpeg writer.
            tmp = out_path.with_name(out_path.name + ".tmp.mp4")
            imageio.mimsave(tmp, list(frames), fps=args.fps)
            os.replace(tmp, out_path)
            if not complete(out_path):
                raise IOError(f"empty output: {out_path}")
            timings.append(time.time() - started)
        summary = {
            "training_step": args.training_step, "cfg": cfg,
            "worker_id": args.worker_id, "worker_count": args.worker_count,
            "seed": args.seed, "steps": args.steps, "frames": args.frames,
            "expected_shard": len(rows),
            "complete_shard": sum(complete(p) for p in out_dir.glob("*.mp4")),
            "mean_sec_new": statistics.mean(timings) if timings else None,
        }
        (out_dir.parent / f"worker_{args.worker_id}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    print(f"CFG_WORKER_DONE id={args.worker_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
