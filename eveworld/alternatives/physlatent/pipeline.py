from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from giga_models import GigaWorld0Pipeline

from .modules import PhysicsLatentEncoder, append_physics_tokens


def _resolve_physics_latent_weights(model_path: str | os.PathLike[str]) -> Path:
    path = Path(model_path)
    if path.is_file():
        return path

    candidates = [
        path / 'diffusion_pytorch_model.bin',
        path / 'physics_latent_encoder' / 'diffusion_pytorch_model.bin',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = '\n'.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f'Could not find physics latent weights. Checked:\n{checked}')


def _infer_encoder_kwargs(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    token_queries = state_dict['token_queries']
    latent_proj_weight = state_dict['latent_proj.1.weight']
    prompt_proj_weight = state_dict['prompt_proj.1.weight']
    traj_bias = state_dict['traj_head.bias']
    contact_bias = state_dict['contact_head.bias']
    phase_bias = state_dict.get('phase_head.bias')
    num_tokens = int(token_queries.shape[0])
    traj_points = int(traj_bias.shape[0] // 2) if traj_bias.numel() > 2 else num_tokens

    return {
        'num_tokens': num_tokens,
        'prompt_dim': int(prompt_proj_weight.shape[1]),
        'latent_channels': int(latent_proj_weight.shape[1]),
        'token_dim': int(token_queries.shape[1]),
        'hidden_dim': int(prompt_proj_weight.shape[0]),
        'num_contact_classes': int(contact_bias.shape[0]),
        'num_phase_classes': int(phase_bias.shape[0]) if phase_bias is not None else 8,
        'num_traj_points': traj_points,
        'use_stop_token': 'stop_query' in state_dict,
    }


def load_physics_latent_encoder(
    model_path: str | os.PathLike[str],
    dtype: torch.dtype | None = None,
) -> PhysicsLatentEncoder:
    """Load a PhysicsLatentEncoder saved by the GigaTrain checkpoint hook."""
    weights_path = _resolve_physics_latent_weights(model_path)
    state_dict = torch.load(weights_path, map_location='cpu')
    if not isinstance(state_dict, dict):
        raise TypeError(f'Expected a state_dict dict from {weights_path}, got {type(state_dict)!r}')

    encoder = PhysicsLatentEncoder(**_infer_encoder_kwargs(state_dict))
    model_state = encoder.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state_dict) - set(compatible_state))
    missing, unexpected = encoder.load_state_dict(compatible_state, strict=False)
    if skipped or missing or unexpected:
        warnings.warn(
            'Loaded PhysLatent encoder with partial state_dict compatibility. '
            f'skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}',
            RuntimeWarning,
        )
    encoder.eval()
    if dtype is not None:
        encoder.to(dtype=dtype)
    return encoder


@torch.no_grad()
def build_physlatent_crossattn(
    encoder: PhysicsLatentEncoder,
    prompt_embeds: torch.Tensor,
    ref_latents: torch.Tensor,
    ref_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Utility for inference wrappers: append physics tokens to text context."""
    physics_tokens = encoder(ref_latents=ref_latents, prompt_embeds=prompt_embeds, ref_masks=ref_masks)
    return append_physics_tokens(prompt_embeds, physics_tokens)


class PhysLatentGigaWorld0Pipeline(GigaWorld0Pipeline):
    """GigaWorld-0 inference pipeline with PhysLatent condition tokens."""

    @classmethod
    def from_pretrained(
        cls,
        transformer_model_path,
        text_encoder_model_path=None,
        vae_model_path=None,
        lora_model_path=None,
        lora_fuse=False,
        fp8_eval=True,
        physics_latent_model_path: str | os.PathLike[str] | None = None,
        physlatent_uncond_mode: str = 'shared',
    ):
        pipe = super().from_pretrained(
            transformer_model_path=transformer_model_path,
            text_encoder_model_path=text_encoder_model_path,
            vae_model_path=vae_model_path,
            lora_model_path=lora_model_path,
            lora_fuse=lora_fuse,
            fp8_eval=fp8_eval,
        )
        pipe.physics_latent_encoder = None
        pipe.physlatent_uncond_mode = physlatent_uncond_mode
        if physics_latent_model_path is not None:
            pipe.physics_latent_encoder = load_physics_latent_encoder(
                physics_latent_model_path,
                dtype=pipe.transformer.dtype,
            )
        return pipe

    def to(self, *args, **kwargs):
        pipe = super().to(*args, **kwargs)
        if getattr(self, 'physics_latent_encoder', None) is not None:
            self.physics_latent_encoder.to(self._execution_device)
        return pipe

    @property
    def physics_latent_enabled(self) -> bool:
        return getattr(self, 'physics_latent_encoder', None) is not None

    def _empty_ref_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (
            batch_size,
            self.latent_channels,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        ref_latents = torch.zeros(shape, device=device, dtype=dtype)
        ref_masks = torch.zeros((batch_size, 1, shape[2], 1, 1), device=device, dtype=dtype)
        return ref_latents, ref_masks

    def _append_physlatent_for_cfg(
        self,
        positive_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        ref_latents: torch.Tensor,
        ref_masks: torch.Tensor | None,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.physics_latent_enabled:
            return positive_prompt_embeds, negative_prompt_embeds

        encoder = self.physics_latent_encoder
        encoder_dtype = next(encoder.parameters()).dtype
        encoder_device = next(encoder.parameters()).device
        ref_latents_for_encoder = ref_latents.to(device=encoder_device, dtype=encoder_dtype)
        ref_masks_for_encoder = None if ref_masks is None else ref_masks.to(device=encoder_device, dtype=encoder_dtype)
        positive_for_encoder = positive_prompt_embeds.to(device=encoder_device, dtype=encoder_dtype)

        physics_tokens = encoder(
            ref_latents=ref_latents_for_encoder,
            prompt_embeds=positive_for_encoder,
            ref_masks=ref_masks_for_encoder,
        ).to(device=positive_prompt_embeds.device, dtype=target_dtype)
        positive_prompt_embeds = append_physics_tokens(positive_prompt_embeds, physics_tokens)

        if negative_prompt_embeds is None:
            return positive_prompt_embeds, negative_prompt_embeds

        mode = getattr(self, 'physlatent_uncond_mode', 'shared')
        if mode == 'shared':
            negative_tokens = physics_tokens
        elif mode == 'zero':
            negative_tokens = torch.zeros_like(physics_tokens)
        elif mode == 'separate':
            negative_for_encoder = negative_prompt_embeds.to(device=encoder_device, dtype=encoder_dtype)
            negative_tokens = encoder(
                ref_latents=ref_latents_for_encoder,
                prompt_embeds=negative_for_encoder,
                ref_masks=ref_masks_for_encoder,
            ).to(device=negative_prompt_embeds.device, dtype=target_dtype)
        else:
            raise ValueError(f'Unsupported physlatent_uncond_mode: {mode!r}')

        negative_prompt_embeds = append_physics_tokens(negative_prompt_embeds, negative_tokens)
        return positive_prompt_embeds, negative_prompt_embeds

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
        output_type: Optional[str] = 'pil',
    ):
        self._guidance_scale = guidance_scale
        batch_size = 1
        device = self._execution_device
        dtype = self.transformer.dtype

        generator = None
        if seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        positive_prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        positive_prompt_embeds = positive_prompt_embeds.to(device=device, dtype=dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        self.scheduler.config.sigma_max = sigma_max
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas.to(device)

        shape = (
            batch_size,
            self.latent_channels,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        if image is not None:
            cond_latents, cond_masks = self.prepare_cond_latents(
                image,
                num_frames,
                height,
                width,
                device=device,
                dtype=torch.float32,
            )
            cond_sigma = torch.tensor([augment_sigma], device=device, dtype=cond_latents.dtype)
            masks_input = cond_masks.repeat(batch_size, 1, 1, cond_latents.shape[-2], cond_latents.shape[-1])
            ref_latents, ref_masks = cond_latents, cond_masks
        else:
            cond_latents = cond_masks = None
            ref_latents, ref_masks = self._empty_ref_latents(batch_size, num_frames, height, width, device, torch.float32)

        positive_prompt_embeds, negative_prompt_embeds = self._append_physlatent_for_cfg(
            positive_prompt_embeds=positive_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds if self.do_classifier_free_guidance else None,
            ref_latents=ref_latents,
            ref_masks=ref_masks,
            target_dtype=dtype,
        )
        prompt_embeds = positive_prompt_embeds
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds], dim=0)

        padding_mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        if self.do_classifier_free_guidance:
            padding_mask = torch.cat([padding_mask, padding_mask], dim=0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents
                if cond_latents is not None:
                    cond_noise = torch.randn(
                        cond_latents.shape,
                        device=device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    latent_model_input = self.scheduler.add_condition_inputs(
                        latent_model_input,
                        sigmas[i],
                        cond_sample=cond_latents,
                        cond_mask=cond_masks,
                        cond_noise=cond_noise,
                        cond_sigma=cond_sigma,
                    )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                timestep = t
                if cond_masks is not None:
                    latent_model_input = torch.cat([latent_model_input, masks_input], dim=1)
                    timestep = timestep.view(1, 1, 1, 1, 1).expand(latents.size(0), -1, latents.size(2), -1, -1)
                    t_conditioning = cond_sigma / (cond_sigma + 1)
                    cond_timestep = cond_masks * t_conditioning + (1 - cond_masks) * timestep
                    timestep = cond_timestep.to(dtype)
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                latent_model_input = latent_model_input.to(dtype)

                timestep = timestep.expand(latent_model_input.shape[0], -1, -1, -1, -1)
                timestep = timestep.to(dtype)

                noise_pred = self.transformer(
                    x=latent_model_input,
                    timesteps=timestep,
                    crossattn_emb=prompt_embeds,
                    fps=fps,
                    padding_mask=padding_mask,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = noise_pred.float()

                latents = self.scheduler.step(
                    noise_pred,
                    timestep=t,
                    sample=latents,
                    cond_sample=cond_latents,
                    cond_mask=cond_masks,
                    return_dict=False,
                )[0]

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if not output_type == 'latent':
            video = self.decode(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        return video
