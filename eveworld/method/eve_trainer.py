"""EVE · 因果过程忠实 Trainer(继承 physlatent,复用其真实前向)。

在父类 EDM 去噪损失之外叠加 CPC:真实过程去噪误差 < 偷懒负样本(latent 空间构造)。
runner 名: 'eveworld.EveCausalTrainer'(见 eveworld/__init__.py)。

实现按 eveworld/alternatives/physlatent/trainer.py 的真实 forward_step 精确复刻:
  transformer(x=, timesteps=, crossattn_emb=, padding_mask=, fps=)
  add_noise(latents)->(input,timesteps); denoise(pred); compute_loss(denoised)->标量
"""
from __future__ import annotations
import functools
from typing import Any
import torch
import torch.nn.functional as F

from eveworld.alternatives.physlatent.trainer import PhysicsLatentGigaWorld0Trainer
from eveworld.alternatives.physlatent.modules import append_physics_tokens, cfg_get
from .losses.lazy_negatives import make_negatives, random_control


class EveCausalTrainer(PhysicsLatentGigaWorld0Trainer):
    def get_models(self, model_config: Any):
        model = super().get_models(model_config)
        causal = cfg_get(model_config, 'causal', {})
        self._w_cpc = float(cfg_get(causal, 'w_cpc', 0.5))
        self._neg_kinds = list(cfg_get(causal, 'neg_kinds', ['teleport', 'excision', 'freeze_jump']))
        self._margin = float(cfg_get(causal, 'margin', 0.1))
        self._use_random_neg = bool(cfg_get(causal, 'use_random_neg', False))
        return model

    def _denoise_loss_from_latents(self, latents, ref_latents, ref_masks, prompt_embeds,
                                   padding_mask, fps):
        """复刻父类去噪路径,输入任意 clean latent 序列,返回标量去噪误差。"""
        transformer = functools.partial(self.model, 'transformer')
        input_latents, timesteps = self.edm_loss.add_noise(latents)
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
        model_pred = transformer(x=input_latents, timesteps=timesteps,
                                 crossattn_emb=prompt_embeds.to(self.dtype),
                                 padding_mask=padding_mask, fps=fps)
        denoised = self.edm_loss.denoise(model_pred.float())
        denoised = ref_masks * ref_latents + (1 - ref_masks) * denoised
        return self.edm_loss.compute_loss(denoised)

    def forward_step(self, batch_dict: dict[str, Any]):
        losses = super().forward_step(batch_dict)      # 主去噪 'edm'(+ physics 项,已置0)
        if self._w_cpc <= 0:
            return losses
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']
        fps = batch_dict['fps'][0]
        bs = images.shape[0]
        padding_mask = torch.zeros((bs, 1, images.shape[-2], images.shape[-1]),
                                   dtype=self.dtype, device=self.device)
        latents = self.forward_vae(images)
        ref_latents = self.forward_vae(batch_dict['ref_images'])
        ref_masks = batch_dict['ref_masks'].to(self.dtype)
        # 若启用 physics token,负样本比较也用同一 prompt_embeds(条件一致)
        pe = prompt_embeds
        if getattr(self, 'physics_latent_enabled', False):
            physics_encoder = functools.partial(self.model, 'physics_latent_encoder')
            physics_tokens, _ = physics_encoder(ref_latents=ref_latents.to(self.dtype),
                                                prompt_embeds=prompt_embeds.to(self.dtype),
                                                ref_masks=ref_masks, return_aux=True)
            pe = append_physics_tokens(prompt_embeds.to(self.dtype), physics_tokens)

        e_pos = self._denoise_loss_from_latents(latents, ref_latents, ref_masks, pe, padding_mask, fps)
        negs = {'random': random_control(latents)} if self._use_random_neg else make_negatives(latents, self._neg_kinds)
        for k, zneg in negs.items():
            e_neg = self._denoise_loss_from_latents(zneg, ref_latents, ref_masks, pe, padding_mask, fps)
            losses[f'cpc_{k}'] = self._w_cpc * F.relu(self._margin - (e_neg - e_pos))
            # rank0 可视分项(便于监控 CPC margin;e_neg 应 > e_pos)
            if int(getattr(self, 'process_index', 0)) == 0:
                print(f"[EVE cpc] {k}: e_pos={float(e_pos):.4f} e_neg={float(e_neg):.4f} "
                      f"margin_loss={float(losses[f'cpc_{k}']):.4f}", flush=True)
        return losses
