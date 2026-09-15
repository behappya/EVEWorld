#!/usr/bin/env python3
"""EVE · EAG 采样 pipeline(方案27 §四 I2)。

继承 PhysLatentGigaWorld0Pipeline, 只在采样循环里插一处 EAG 引导:
  noise_pred 算好后 -> 用 scheduler.precondition_outputs 拿 x0 预测 ẑ0
  -> EAG 用 LAD 能量梯度修正 ẑ0(往转移合法推) -> 反解回修正后的 noise_pred
  -> scheduler.step 照常。
不改父类, 不重训 backbone。只有 eag_guidance.weight>0 时才生效(否则 == 原始采样)。

用法:
  from eveworld.method.pipeline_eag import EAGGigaWorld0Pipeline
  pipe = EAGGigaWorld0Pipeline.from_pretrained(..., lam_path=..., eag_weight=0.03)
  video = pipe(prompt=..., image=..., ...)
"""
from __future__ import annotations
import os, sys
from typing import Optional

import torch
from diffusers.utils.torch_utils import randn_tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eveworld.alternatives.physlatent.pipeline import PhysLatentGigaWorld0Pipeline
from eveworld.method.lam.latent_action_model import LatentActionModel
from eveworld.method.eag import EAGGuidance


class EAGGigaWorld0Pipeline(PhysLatentGigaWorld0Pipeline):
    """在父类采样上加 EAG 可执行性引导。"""

    def attach_eag(self, lam_path: str, weight: float = 0.03, topk: int = 3, tau: float = 0.5):
        dev = self._execution_device
        ck = torch.load(lam_path, map_location=dev)
        lam = LatentActionModel(latent_ch=ck["z_dim"], action_dim=ck["action_dim"], codebook=ck["codebook"]).to(dev)
        lam.load_state_dict(ck["state_dict"])
        lam.eval()
        self.eag_guidance = EAGGuidance(lam, topk=topk, tau=tau, weight=weight)
        self.eag_lam_z_dim = ck["z_dim"]
        print(f"[EAG] attached LAD(z_dim={ck['z_dim']}) weight={weight} topk={topk}", flush=True)
        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image=None,
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
    ):
        eag = getattr(self, "eag_guidance", None)
        if eag is None or eag.weight <= 0:
            # 无引导 -> 完全走父类(等价原始采样)
            return super().__call__(
                prompt=prompt, negative_prompt=negative_prompt, image=image,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                fps=fps, num_frames=num_frames, height=height, width=width, seed=seed,
                augment_sigma=augment_sigma, sigma_max=sigma_max, output_type=output_type)

        # ---- 复刻父类采样循环, 仅在 x0 处插 EAG(其余逐行同 physlatent pipeline) ----
        self._guidance_scale = guidance_scale
        device = self._execution_device
        dtype = self.transformer.dtype
        generator = None
        if seed > 0:
            generator = torch.Generator(device=device); generator.manual_seed(seed)

        pos, neg = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        pos = pos.to(device=device, dtype=dtype)
        if neg is not None:
            neg = neg.to(device=device, dtype=dtype)

        self.scheduler.config.sigma_max = sigma_max
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas.to(device)

        shape = (1, self.latent_channels,
                 (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                 int(height) // self.vae_scale_factor_spatial,
                 int(width) // self.vae_scale_factor_spatial)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        if image is not None:
            cond_latents, cond_masks = self.prepare_cond_latents(image, num_frames, height, width, device=device, dtype=torch.float32)
            cond_sigma = torch.tensor([augment_sigma], device=device, dtype=cond_latents.dtype)
            masks_input = cond_masks.repeat(1, 1, 1, cond_latents.shape[-2], cond_latents.shape[-1])
            ref_latents, ref_masks = cond_latents, cond_masks
        else:
            cond_latents = cond_masks = None
            ref_latents, ref_masks = self._empty_ref_latents(1, num_frames, height, width, device, torch.float32)

        pos, neg = self._append_physlatent_for_cfg(
            positive_prompt_embeds=pos,
            negative_prompt_embeds=neg if self.do_classifier_free_guidance else None,
            ref_latents=ref_latents, ref_masks=ref_masks, target_dtype=dtype)
        prompt_embeds = pos
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([neg, pos], dim=0)

        padding_mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        if self.do_classifier_free_guidance:
            padding_mask = torch.cat([padding_mask, padding_mask], dim=0)

        eag_log = []
        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents
                if cond_latents is not None:
                    cond_noise = torch.randn(cond_latents.shape, device=device, dtype=torch.float32, generator=generator)
                    latent_model_input = self.scheduler.add_condition_inputs(
                        latent_model_input, sigmas[i], cond_sample=cond_latents,
                        cond_mask=cond_masks, cond_noise=cond_noise, cond_sigma=cond_sigma)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                timestep = t
                if cond_masks is not None:
                    latent_model_input = torch.cat([latent_model_input, masks_input], dim=1)
                    timestep = timestep.view(1, 1, 1, 1, 1).expand(latents.size(0), -1, latents.size(2), -1, -1)
                    t_conditioning = cond_sigma / (cond_sigma + 1)
                    timestep = (cond_masks * t_conditioning + (1 - cond_masks) * timestep).to(dtype)
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                latent_model_input = latent_model_input.to(dtype)
                timestep = timestep.expand(latent_model_input.shape[0], -1, -1, -1, -1).to(dtype)

                noise_pred = self.transformer(x=latent_model_input, timesteps=timestep,
                                              crossattn_emb=prompt_embeds, fps=fps, padding_mask=padding_mask)
                if self.do_classifier_free_guidance:
                    nu, nt = noise_pred.chunk(2)
                    noise_pred = nt + self.guidance_scale * (nt - nu)
                noise_pred = noise_pred.float()

                # ---------- EAG 引导: 在 x0 空间修正, 反解回 noise_pred ----------
                sigma = float(sigmas[i])
                # x0 预测(scheduler 现成的 precondition_outputs)
                x0 = self.scheduler.precondition_outputs(latents.float(), noise_pred, sigmas[i])
                w = eag.sigma_weight(sigma, sigma_max)
                if w > 0 and x0.shape[1] == self.eag_lam_z_dim:
                    e_before = eag.energy(x0).item()
                    grad = eag.gradient(x0)
                    B = grad.shape[0]
                    gn = (grad.reshape(B, -1) / (grad.reshape(B, -1).norm(dim=1, keepdim=True) + 1e-8)).reshape_as(grad)
                    x0n = x0.reshape(B, -1).norm(dim=1).view(B, 1, 1, 1, 1)
                    x0_new = x0 - w * x0n * gn
                    # 反解 noise_pred: x0 = c_skip*sample + c_out*model_out
                    #   => model_out = (x0 - c_skip*sample)/c_out ; 用 Δx0 更新 noise_pred
                    #   c_out 依 sigma, 用两次 precondition 的线性关系直接按比例回填:
                    #   x0 对 model_output 线性(斜率 c_out), 故 Δnoise = Δx0 / c_out
                    # 取 c_out: 由 precondition_outputs 内部公式(epsilon 预测)
                    sd = self.scheduler.config.sigma_data
                    ptype = self.scheduler.config.prediction_type
                    if ptype == "epsilon":
                        c_out = sigma * sd / (sigma ** 2 + sd ** 2) ** 0.5
                    elif ptype == "rf":
                        c_out = -(sigma / (1 + sigma))
                    elif ptype == "v_prediction":
                        c_out = -sigma * sd / (sigma ** 2 + sd ** 2) ** 0.5
                    else:
                        c_out = 1.0
                    noise_pred = noise_pred + (x0_new - x0) / (c_out if abs(c_out) > 1e-6 else 1.0)
                    if i % 5 == 0:
                        e_after = eag.energy(self.scheduler.precondition_outputs(latents.float(), noise_pred, sigmas[i])).item()
                        eag_log.append((i, sigma, e_before, e_after))
                        print(f"[EAG] step{i} sigma={sigma:.2f} w={w:.4f} E {e_before:.4f}->{e_after:.4f}", flush=True)

                latents = self.scheduler.step(noise_pred, timestep=t, sample=latents,
                                              cond_sample=cond_latents, cond_mask=cond_masks, return_dict=False)[0]
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    pbar.update()

        if output_type != "latent":
            video = self.decode(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents
        self.maybe_free_model_hooks()
        return video
