from __future__ import annotations

import functools
from typing import Any

import torch
from diffusers.models import AutoencoderKLWan
from einops import rearrange
from giga_train import ModuleDict, Trainer
from peft import LoraConfig

from giga_models import GigaWorld0Transformer3DModel, LoRAPeftWrapper
from giga_models.nn import EDMLoss

from .losses import PhysicsAuxiliaryLoss
from .modules import PhysicsLatentConfig, PhysicsLatentEncoder, append_physics_tokens, cfg_get


class PhysicsLatentGigaWorld0Trainer(Trainer):
    """GigaWorld trainer with a lightweight physics-latent condition path."""

    def get_models(self, model_config: Any):
        model = dict()

        vae_dtype = cfg_get(model_config, 'vae_dtype', self.dtype)
        vae = AutoencoderKLWan.from_pretrained(model_config.vae_model_path)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)

        physics_config = cfg_get(model_config, 'physics_latent', {})
        self.physics_config = PhysicsLatentConfig.from_config(physics_config)
        self.physics_latent_enabled = self.physics_config.enabled
        self.physics_aux_loss = PhysicsAuxiliaryLoss(cfg_get(physics_config, 'loss_weights', {}))
        self.terminal_denoise_weight = float(cfg_get(physics_config, 'terminal_denoise_weight', 0.0))
        self.terminal_quality_threshold = float(cfg_get(physics_config, 'terminal_quality_threshold', 0.6))

        transformer = GigaWorld0Transformer3DModel.from_pretrained(model_config.transformer_model_path)
        self.train_mode = cfg_get(model_config, 'train_mode', 'full')
        if self.train_mode == 'lora':
            transformer.requires_grad_(False)
            lora_rank = cfg_get(model_config, 'lora_rank', 64)
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
                target_modules=['to_q.0', 'to_k.0', 'to_v.0', 'to_out.0'],
            )
            transformer.add_adapter(lora_config)
            transformer = LoRAPeftWrapper(transformer)
        elif not self.physics_config.train_backbone:
            transformer.requires_grad_(False)
        model.update(transformer=transformer)

        if self.physics_latent_enabled:
            model.update(
                physics_latent_encoder=PhysicsLatentEncoder(
                    num_tokens=self.physics_config.num_tokens,
                    prompt_dim=self.physics_config.prompt_dim,
                    latent_channels=self.physics_config.latent_channels,
                    token_dim=self.physics_config.token_dim,
                    hidden_dim=self.physics_config.hidden_dim,
                    num_contact_classes=self.physics_config.num_contact_classes,
                    num_phase_classes=self.physics_config.num_phase_classes,
                    num_traj_points=self.physics_config.num_traj_points,
                    num_attention_heads=self.physics_config.num_attention_heads,
                    spatial_pool_size=self.physics_config.spatial_pool_size,
                    dropout=self.physics_config.dropout,
                    use_stop_token=self.physics_config.use_stop_token,
                )
            )

        self.edm_loss = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)

        model = ModuleDict(model)
        model.to(self.dtype)
        model.train()
        if self.train_mode == 'lora':
            model.load_state_dict_mode = 'each'
        if self.mixed_precision == 'fp8':
            for model_name in model.keys():
                model[model_name].to_fp8(ignore_modules=model_config.fp8_ignore_modules)
        return model

    def forward_step(self, batch_dict: dict[str, Any]):
        transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']
        batch_size = images.shape[0]

        padding_mask = torch.zeros((batch_size, 1, images.shape[-2], images.shape[-1]), dtype=self.dtype, device=self.device)
        fps = batch_dict['fps'][0]

        latents = self.forward_vae(images)
        input_latents, timesteps = self.edm_loss.add_noise(latents)

        ref_images = batch_dict['ref_images']
        ref_masks = batch_dict['ref_masks'].to(self.dtype)
        ref_latents = self.forward_vae(ref_images)

        augment_sigma = torch.tensor([0.0001], device=ref_latents.device, dtype=latents.dtype)
        while len(augment_sigma.shape) < len(ref_latents.shape):
            augment_sigma = augment_sigma.unsqueeze(-1)

        input_latents = ref_masks * ref_latents + (1 - ref_masks) * input_latents
        input_masks = ref_masks.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])
        input_latents = torch.cat([input_latents, input_masks], dim=1)
        timesteps = timesteps.view(1, 1, 1, 1, 1).expand(latents.size(0), -1, latents.size(2), -1, -1)
        t_conditioning = augment_sigma / (augment_sigma + 1)
        timesteps = ref_masks * t_conditioning + (1 - ref_masks) * timesteps

        input_latents = input_latents.to(self.dtype)
        timesteps = timesteps.to(self.dtype)
        prompt_embeds = prompt_embeds.to(self.dtype)

        aux_outputs: dict[str, torch.Tensor] = {}
        if self.physics_latent_enabled:
            physics_encoder = functools.partial(self.model, 'physics_latent_encoder')
            physics_tokens, aux_outputs = physics_encoder(
                ref_latents=ref_latents.to(self.dtype),
                prompt_embeds=prompt_embeds,
                ref_masks=ref_masks,
                return_aux=True,
            )
            prompt_embeds = append_physics_tokens(prompt_embeds, physics_tokens)

        if self.train_mode == 'lora':
            input_latents.requires_grad_(True)
        model_pred = transformer(
            x=input_latents,
            timesteps=timesteps,
            crossattn_emb=prompt_embeds,
            padding_mask=padding_mask,
            fps=fps,
        )
        denoised_latents = self.edm_loss.denoise(model_pred.float())
        if 'ref_images' in batch_dict:
            denoised_latents = ref_masks * ref_latents + (1 - ref_masks) * denoised_latents
        edm_loss = self.edm_loss.compute_loss(denoised_latents)
        losses = {'edm': edm_loss}
        terminal_denoise_loss = self._terminal_denoise_loss(denoised_latents, latents, batch_dict)
        if terminal_denoise_loss is not None:
            losses['phys_terminal_denoise'] = terminal_denoise_loss

        aux_losses = self.physics_aux_loss(aux_outputs, batch_dict)
        for key, value in aux_losses.items():
            losses[f'phys_{key}'] = value
        return losses

    def _terminal_denoise_loss(
        self,
        denoised_latents: torch.Tensor,
        target_latents: torch.Tensor,
        batch_dict: dict[str, Any],
    ) -> torch.Tensor | None:
        if self.terminal_denoise_weight <= 0 or 'phys_terminal_mask' not in batch_dict:
            return None

        mask = batch_dict['phys_terminal_mask'].to(device=denoised_latents.device, dtype=denoised_latents.dtype)
        if mask.ndim == 1:
            mask = mask[None, :]
        quality = batch_dict.get('phys_label_quality')
        if quality is not None:
            quality = quality.to(device=denoised_latents.device, dtype=denoised_latents.dtype).view(-1, 1)
            mask = mask * (quality >= self.terminal_quality_threshold).to(mask.dtype)

        latent_t = denoised_latents.shape[2]
        mask = torch.nn.functional.interpolate(mask[:, None, :], size=latent_t, mode='nearest')[:, 0]

        value = (denoised_latents.float() - target_latents.float()).square().mean(dim=(1, 3, 4))
        mask = mask.to(value.dtype)
        if mask.sum() <= 0:
            loss = value.sum() * 0.0
        else:
            loss = (value * mask).sum() / mask.sum().clamp(min=1)
        return loss * self.terminal_denoise_weight

    def forward_vae(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.sample()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents
