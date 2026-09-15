"""Sequential committed-frontier inference for GigaWorld-0."""

from __future__ import annotations

from typing import Optional

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor

from giga_models import GigaWorld0Pipeline

from .eve_frontier_loss import (
    CommittedHistory,
    FrontierConfig,
    boundary_commit_guard,
    latent_frames_from_pixels,
    prepare_frontier_sampling_inputs,
)


class FrontierGigaWorld0Pipeline(GigaWorld0Pipeline):
    """Generate one active latent block at a time and freeze every commit.

    The implementation intentionally keeps the MVP boundary: no full-token
    causal mask, K/V cache, or overlap guard.  A model forward contains only
    the clean committed prefix and the currently denoised block.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: PipelineImageInput = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        fps: int = 16,
        num_frames: int = 93,
        height: int = 480,
        width: int = 768,
        seed: int = -1,
        augment_sigma: float = 0.001,
        sigma_max: float = 80.0,
        output_type: Optional[str] = "pil",
        block_size: int = 4,
        condition_latents: int = 1,
        history_sigma: float = 1e-4,
        boundary_guard_strength: float = 0.0,
        boundary_guard_velocity_scale: float = 0.5,
        boundary_guard_decay: float = 1.0,
    ):
        if image is None:
            raise ValueError("Frontier image-to-video sampling requires a conditioning image")
        if augment_sigma != 0.001:
            raise ValueError("augment_sigma is not used by committed-frontier sampling")

        self._guidance_scale = guidance_scale
        batch_size = 1
        device = self._execution_device
        dtype = self.transformer.dtype
        total_latents = latent_frames_from_pixels(num_frames, self.vae_scale_factor_temporal)
        config = FrontierConfig(
            block_size=block_size,
            condition_latents=condition_latents,
            history_sigma=history_sigma,
            sigma_data=float(self.scheduler.config.sigma_data),
            use_flow=self.scheduler.config.prediction_type == "rf",
        )

        generator = None
        if seed > 0:
            generator = torch.Generator(device=device).manual_seed(seed)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        cond_latents, _ = self.prepare_cond_latents(
            image,
            num_frames,
            height,
            width,
            device=device,
            dtype=torch.float32,
        )
        condition = cond_latents[:, :, :condition_latents].to(dtype=dtype)
        if condition.shape[2] != condition_latents:
            raise ValueError(
                f"conditioning image produced {condition.shape[2]} latent tokens; "
                f"expected {condition_latents}"
            )

        latent_template = torch.empty(
            (
                batch_size,
                self.latent_channels,
                total_latents,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            ),
            device=device,
            dtype=dtype,
        )
        latent_template[:, :, :condition_latents] = condition
        state = CommittedHistory.from_first_latent(latent_template, config)
        initial_condition = state.committed
        padding_mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
        if self.do_classifier_free_guidance:
            padding_mask = torch.cat([padding_mask, padding_mask], dim=0)

        self.scheduler.config.sigma_max = sigma_max
        trace = []
        while not state.complete:
            frontier = state.next_frontier
            if frontier is None:
                raise RuntimeError("frontier rollout stopped before all latent tokens were committed")
            committed_before = state.committed
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            active_shape = (
                batch_size,
                self.latent_channels,
                frontier.active_length,
                latent_template.shape[-2],
                latent_template.shape[-1],
            )
            active = randn_tensor(active_shape, generator=generator, device=device, dtype=dtype)
            active = active * self.scheduler.init_noise_sigma

            for timestep in timesteps:
                scaled_active = self.scheduler.scale_model_input(active, timestep)
                model_input, model_timesteps, _ = prepare_frontier_sampling_inputs(
                    committed_before,
                    scaled_active,
                    timestep,
                    config,
                )
                if self.do_classifier_free_guidance:
                    model_input = torch.cat([model_input, model_input], dim=0)
                    model_timesteps = torch.cat([model_timesteps, model_timesteps], dim=0)

                prediction = self.transformer(
                    x=model_input.to(dtype),
                    timesteps=model_timesteps.to(dtype),
                    crossattn_emb=prompt_embeds,
                    fps=fps,
                    padding_mask=padding_mask,
                )
                active_prediction = prediction[:, :, frontier.history_length :, :, :]
                if self.do_classifier_free_guidance:
                    pred_uncond, pred_text = active_prediction.chunk(2)
                    active_prediction = pred_text + self.guidance_scale * (pred_text - pred_uncond)
                active = self.scheduler.step(
                    active_prediction.float(),
                    timestep=timestep,
                    sample=active,
                    return_dict=False,
                )[0]

            active = boundary_commit_guard(
                committed_before,
                active,
                strength=boundary_guard_strength,
                velocity_scale=boundary_guard_velocity_scale,
                decay=boundary_guard_decay,
            )
            if not torch.equal(state.committed, committed_before):
                raise RuntimeError("committed history changed while denoising an active block")
            state = state.commit(active)
            if not torch.equal(state.committed[:, :, : frontier.history_length], committed_before):
                raise RuntimeError("a commit rewrote an earlier latent token")
            trace.append(
                {
                    "frontier": frontier.index,
                    "history_length": frontier.history_length,
                    "active_length": frontier.active_length,
                    "committed_length": state.committed.shape[2],
                    "boundary_guard_strength": boundary_guard_strength,
                }
            )

        latents = state.committed
        if latents.shape[2] != total_latents:
            raise RuntimeError(f"frontier rollout returned {latents.shape[2]} latents, expected {total_latents}")
        if not torch.equal(latents[:, :, :condition_latents], initial_condition):
            raise RuntimeError("first-frame conditioning latents drifted during frontier rollout")
        self.last_frontier_trace = tuple(trace)

        if output_type != "latent":
            video = self.decode(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents
        self.maybe_free_model_hooks()
        return video


__all__ = ["FrontierGigaWorld0Pipeline"]
