#!/usr/bin/env python3
"""Evaluate checkpoint-level immediate-next preference.

This evaluator deliberately uses the same fixed sigma/noise and the same
packed train rows for every checkpoint.  It measures the model's next/future
EDM score gap and subtracts the frozen base gap, so a checkpoint is credited
only for preference beyond ordinary temporal continuity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKLWan
from einops import rearrange
from giga_datasets import load_dataset
from giga_models import GigaWorld0Transformer3DModel, LoRAPeftWrapper
from peft import LoraConfig

from eveworld.method.eve_frontier_loss import (
    FrontierConfig,
    fixed_sigma_frontier_batches,
    frontier_score_from_prediction,
    future_candidate_target,
    prepare_frontier_batch,
    prepare_frontier_candidate_batch,
    terminal_candidate_target,
)
from eveworld.alternatives.physlatent.transforms import PhysLatentGigaWorld0Transform


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(path)


def _load_latents(
    dataset,
    indices: list[int],
    vae,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    transform: PhysLatentGigaWorld0Transform,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    output: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index in indices:
        record = transform(dataset[index])
        images = record["images"].unsqueeze(0).to(device=device, dtype=vae.dtype)
        prompt_embeds = record["prompt_embeds"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            image_tensor = rearrange(images, "b t c h w -> b c t h w")
            latents = vae.encode(image_tensor).latent_dist.mode()
        latents = (latents - latents_mean) * latents_std
        output[index] = (latents, prompt_embeds)
    return output


def _forward_candidate(
    model,
    batch,
    prompt_embeds: torch.Tensor,
    padding_mask: torch.Tensor,
    fps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    input_masks = batch.condition_mask.repeat(
        1, 1, 1, batch.model_input.shape[-2], batch.model_input.shape[-1]
    )
    model_input = torch.cat([batch.model_input, input_masks], dim=1).to(dtype)
    return model(
        x=model_input,
        timesteps=batch.timesteps.to(dtype),
        crossattn_emb=prompt_embeds,
        padding_mask=padding_mask,
        fps=fps,
    )


def _adapter_disabled(model):
    """Return the GW-0 adapter-disable context for a wrapped transformer."""

    inner = getattr(model, "model", model)
    disable = getattr(inner, "disable_adapters", None)
    enable = getattr(inner, "enable_adapters", None)
    if disable is None or enable is None:
        raise RuntimeError(f"model has no adapter-disable API: {type(inner).__name__}")

    class _Context:
        def __enter__(self):
            disable()
            return inner

        def __exit__(self, exc_type, exc, tb):
            enable()
            return False

    return _Context()


def evaluate_checkpoint(
    model: LoRAPeftWrapper,
    checkpoint: Path,
    latent_records: dict[int, tuple[torch.Tensor, torch.Tensor]],
    config: FrontierConfig,
    sigma: float,
    seed: int,
    fps: int,
    dtype: torch.dtype,
    device: torch.device,
    negative_mode: str,
) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    for data_index, (clean_latents, prompt_embeds) in latent_records.items():
        fixed_batches = fixed_sigma_frontier_batches(clean_latents, sigma=sigma, config=config, seed=seed)
        for batch in fixed_batches:
            if batch.frontier.index >= len(fixed_batches) - 1:
                continue
            if negative_mode == "terminal":
                future_target = terminal_candidate_target(clean_latents, batch.frontier, config=config)
            else:
                future_target = future_candidate_target(clean_latents, batch.frontier, config=config, offset_blocks=1)
            if future_target is None:
                continue
            future_batch = prepare_frontier_candidate_batch(
                clean_latents,
                batch.frontier,
                active_target=future_target,
                sigma=sigma,
                noise=(batch.raw_noisy_active - batch.active_target())
                / batch.sigma.reshape((1, 1, 1, 1, 1)),
                config=config,
            )
            padding_mask = torch.zeros(
                (1, 1, clean_latents.shape[-2] * 8, clean_latents.shape[-1] * 8),
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                next_prediction = _forward_candidate(model, batch, prompt_embeds, padding_mask, fps, dtype)
                future_prediction = _forward_candidate(model, future_batch, prompt_embeds, padding_mask, fps, dtype)
                next_score = frontier_score_from_prediction(batch, next_prediction.float(), config)
                future_score = frontier_score_from_prediction(future_batch, future_prediction.float(), config)
                with _adapter_disabled(model):
                    base_next_prediction = _forward_candidate(model, batch, prompt_embeds, padding_mask, fps, dtype)
                    base_future_prediction = _forward_candidate(model, future_batch, prompt_embeds, padding_mask, fps, dtype)
                    base_next_score = frontier_score_from_prediction(batch, base_next_prediction.float(), config)
                    base_future_score = frontier_score_from_prediction(future_batch, base_future_prediction.float(), config)
            raw_gap = (next_score - future_score).item()
            base_gap = (base_next_score - base_future_score).item()
            rows.append(
                {
                    "data_index": data_index,
                    "frontier": batch.frontier.index,
                    "active_start": batch.frontier.active.start,
                    "active_stop": batch.frontier.active.stop,
                    "raw_gap": raw_gap,
                    "base_gap": base_gap,
                    "centered_gap": raw_gap - base_gap,
                    "next_score": next_score.item(),
                    "future_score": future_score.item(),
                }
            )

    centered = [row["centered_gap"] for row in rows]
    raw = [row["raw_gap"] for row in rows]
    next_scores = [row["next_score"] for row in rows]
    future_scores = [row["future_score"] for row in rows]
    return {
        "num_rows": len(rows),
        "mean_raw_gap": sum(raw) / max(1, len(raw)),
        "mean_centered_gap": sum(centered) / max(1, len(centered)),
        "mean_next_score": sum(next_scores) / max(1, len(next_scores)),
        "mean_future_score": sum(future_scores) / max(1, len(future_scores)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", default="91,36,57,82,79,23,5,16")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--condition-latents", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--negative-mode", choices=("adjacent", "terminal"), default="adjacent")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    dataset = load_dataset(args.dataset)
    dataset.open()
    transform = PhysLatentGigaWorld0Transform(
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        random_crop=False,
        image_cfg={"mask_generator": {"max_ref_frames": 1, "start": 1, "factor": 4}},
    )
    vae = AutoencoderKLWan.from_pretrained(args.vae).to(device=device, dtype=dtype).eval()
    vae.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=dtype)).view(1, vae.config.z_dim, 1, 1, 1)
    latent_records = _load_latents(dataset, indices, vae, latents_mean, latents_std, transform, device)

    transformer = GigaWorld0Transformer3DModel.from_pretrained(args.transformer).to(device=device, dtype=dtype)
    transformer.requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights=True,
            target_modules=["to_q.0", "to_k.0", "to_v.0", "to_out.0"],
        )
    )
    model = LoRAPeftWrapper(transformer)
    config = FrontierConfig(block_size=args.block_size, condition_latents=args.condition_latents)
    results: dict[str, Any] = {
        "sigma": args.sigma,
        "seed": args.seed,
        "indices": indices,
        "checkpoints": {},
    }
    for name, checkpoint in args.checkpoint:
        results["checkpoints"][name] = evaluate_checkpoint(
            model,
            checkpoint,
            latent_records,
            config,
            args.sigma,
            args.seed,
            args.fps,
            dtype,
            device,
            args.negative_mode,
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    key: value[key]
                    for key in ("mean_centered_gap", "mean_next_score", "mean_future_score")
                }
                for name, value in results["checkpoints"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
