"""EVE · LAD-LoRA 训练级反偷懒(方案27 §十七/§二十 正路: EAG 的训练时版本)。

动机(§二十): EAG 采样引导对 GW-0 陡峭能量地形强度天然不足(负结果)。训练级不受单步
步长限制 —— 把"转移可执行性"蒸馏进 LoRA 权重, 使不加采样引导也更忠实。这是模型侧创新。

机制:
  冻结预训 LAD -> backbone(LoRA)去噪出 x0 预测 -> 对 x0 算 transition_error 的
  soft-top-k(盯非法跳变尖峰, 与 §十四 MAX 聚合发现一致) -> 作正则压低。
  关键: 按 sigma 门控。EDM 每步采样单个随机 sigma; sigma 大时 x0 基本是噪声, 对其算
  transition_error 无意义 -> 用 1/(1+sigma)(flow 模式下即 c_skip, "x0 中有多少是信号")
  加权, 只有低噪声步真正贡献正则。这也呼应 EAG 实测"大 sigma 无效、小 sigma 才有效"。

runner 名: 'eveworld.EveLadLoraTrainer'(见 eveworld/__init__.py)。
主损失仍是父类 EDM 去噪(保证画质不塌); LAD 正则是辅助项 lad_reg。
"""
from __future__ import annotations
import functools
import math
from typing import Any
import torch
from accelerate import DistributedType

from eveworld.alternatives.physlatent.trainer import PhysicsLatentGigaWorld0Trainer
from eveworld.alternatives.physlatent.modules import cfg_get
from .lam.latent_action_model import LatentActionModel


class EveLadLoraTrainer(PhysicsLatentGigaWorld0Trainer):
    """在 EDM 去噪主损失上叠加"冻结 LAD 的 x0 转移误差"正则(sigma 门控), 蒸馏进 LoRA。"""

    def get_models(self, model_config: Any):
        model = super().get_models(model_config)
        lad_cfg = cfg_get(model_config, 'lad_lora', {})
        self._w_lad = float(cfg_get(lad_cfg, 'w_lad', 0.1))
        self._lad_topk = int(cfg_get(lad_cfg, 'topk', 3))
        self._lad_tau = float(cfg_get(lad_cfg, 'tau', 0.5))
        self._lad_sigma_max = float(cfg_get(lad_cfg, 'sigma_max', 0.0))  # >0: 硬门控, sigma>此值不计正则
        self._lad_balance_max_scale = float(cfg_get(lad_cfg, 'balance_max_scale', 1.0))
        ckpt_path = cfg_get(lad_cfg, 'lam_ckpt', None)
        self._lad = None
        if self._w_lad > 0 and ckpt_path:
            self._lad = self._load_frozen_lad(ckpt_path)
            if int(getattr(self, 'process_index', 0)) == 0:
                print(f"[EVE lad-lora] frozen LAD loaded <- {ckpt_path}; "
                      f"w_lad={self._w_lad} topk={self._lad_topk} tau={self._lad_tau} "
                      f"sigma_max={self._lad_sigma_max} "
                      f"balance_max_scale={self._lad_balance_max_scale}", flush=True)
        return model

    def _load_frozen_lad(self, ckpt_path: str) -> LatentActionModel:
        blob = torch.load(ckpt_path, map_location='cpu')
        lad = LatentActionModel(latent_ch=blob.get('z_dim', 16),
                                action_dim=blob.get('action_dim', 32),
                                codebook=blob.get('codebook', 64))
        lad.load_state_dict(blob['state_dict'])
        lad.to(self.device, dtype=torch.float32)
        for p in lad.parameters():
            p.requires_grad_(False)
        lad.eval()
        return lad

    def _lad_energy(self, z0: torch.Tensor) -> torch.Tensor:
        """soft-top-k 转移可执行性能量(与 eag.py 一致, 盯非法跳变尖峰; 可微)。
        z0:(B,C,T,H,W) -> 标量(B 平均)。梯度经 z0 回流到 backbone/LoRA; LAD 冻结。"""
        te = self._lad.transition_error(z0)              # (B, T-1)
        k = min(self._lad_topk, te.shape[1])
        topv, _ = te.topk(k, dim=1)                      # (B,k)
        w = torch.softmax(topv / self._lad_tau, dim=1)
        return (w * topv).sum(dim=1)                     # (B,)  soft-max per sample

    def forward_step(self, batch_dict: dict[str, Any]):
        # 复刻父类去噪路径, 捕获 denoised_latents(x0 预测)与 sigma, 单次前向同时算 EDM+LAD 正则。
        transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds'].to(self.dtype)
        bs = images.shape[0]
        padding_mask = torch.zeros((bs, 1, images.shape[-2], images.shape[-1]),
                                   dtype=self.dtype, device=self.device)
        fps = batch_dict['fps'][0]

        latents = self.forward_vae(images)
        input_latents, timesteps = self.edm_loss.add_noise(latents)
        ref_latents = self.forward_vae(batch_dict['ref_images'])
        ref_masks = batch_dict['ref_masks'].to(self.dtype)

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
        if self.train_mode == 'lora':      # 与父类一致: 梯度检查点需输入 requires_grad
            input_latents.requires_grad_(True)

        model_pred = transformer(x=input_latents, timesteps=timesteps,
                                 crossattn_emb=prompt_embeds, padding_mask=padding_mask, fps=fps)
        denoised_latents = self.edm_loss.denoise(model_pred.float())
        denoised_latents = ref_masks * ref_latents + (1 - ref_masks) * denoised_latents
        edm_loss = self.edm_loss.compute_loss(denoised_latents)
        losses = {'edm': edm_loss}

        if self._lad is not None and self._w_lad > 0:
            losses['lad_reg'] = self._lad_regularizer(denoised_latents, edm_loss)
        return losses

    def print_before_train(self) -> None:
        """Fail fast unless DeepSpeed's real clipping path is configured."""
        if self.distributed_type != DistributedType.DEEPSPEED:
            raise RuntimeError('EVE LAD-LoRA requires DeepSpeed for the verified 8-GPU training path')
        configured_clip = float(self.accelerator.deepspeed_config.get('gradient_clipping', 0.0))
        expected_clip = float(self.kwargs.get('max_grad_norm', 0.0))
        if configured_clip <= 0 or not math.isclose(configured_clip, expected_clip):
            raise RuntimeError(
                'DeepSpeed gradient clipping is not active or mismatched: '
                f'ds={configured_clip}, expected={expected_clip}'
            )
        super().print_before_train()
        if self.is_main_process:
            self.logger.info(
                '[EVE lad-lora] verified DeepSpeed gradient_clipping=%.4f',
                configured_clip,
            )

    def parse_losses(self, losses: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        loss = super().parse_losses(losses)
        diagnostics = getattr(self, '_lad_diagnostics', None)
        self._lad_diagnostics = None
        if diagnostics:
            for name, value in diagnostics.items():
                reduced = self.accelerator.gather(value.detach().float().reshape(1)).mean()
                self._record_scalar(name, reduced)
        return loss

    def backward_step(self, loss: torch.Tensor) -> None:
        """Record the norm DeepSpeed computed while applying its configured clip."""
        sync_gradients = self.accelerator.sync_gradients
        super().backward_step(loss)
        if sync_gradients:
            engine = self.accelerator.deepspeed_engine_wrapped
            grad_norm = engine.get_global_grad_norm() if engine is not None else None
            if grad_norm is not None:
                self._record_scalar('grad_norm', grad_norm)

    def _record_scalar(self, name: str, value: Any) -> None:
        if not self.is_main_process:
            return
        scalar = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        stats = self._outputs.setdefault(name, {'sum': 0.0, 'num': 0})
        stats['sum'] += scalar
        stats['num'] += 1

    def _lad_regularizer(
        self,
        denoised_latents: torch.Tensor,
        edm_loss: torch.Tensor,
    ) -> torch.Tensor:
        """反偷懒正则 = sigma 门控的 soft-top-k transition_error(EAG 的训练时版本)。

        为何按 sigma 门控: EDM 每步采单个随机 sigma。sigma 大时 x0 预测本身是模糊噪声,
        transition_error 会普遍偏高但与"偷懒"无关(是信噪比问题), 直接罚会注入噪声梯度。
        用 gate = 1/(1+sigma)(= flow 的 c_skip, x0 中真实信号占比)加权: 大 sigma→0(近乎不罚),
        小 sigma→1(x0 接近干净, 此时罚非法跳变才有意义)。与 eag.sigma_weight 调度同源。
        """
        e_per = self._lad_energy(denoised_latents.float())    # (B,)
        sigma = self.edm_loss.sigma.detach().reshape(-1).float()  # (B,) 本步每样本 sigma
        gate = 1.0 / (1.0 + sigma)                            # (B,) in (0,1]
        if self._lad_sigma_max > 0:                           # 可选硬截断: 超阈 sigma 完全不罚
            gate = gate * (sigma <= self._lad_sigma_max).float()
        # Do not normalize by gate.sum(): with batch_size_per_gpu=1 that cancels
        # the gate exactly and gives high-sigma samples full LAD strength.
        weighted_energy = gate * e_per

        # LAD's loss value is small but its x0 gradient is orders of magnitude
        # larger than EDM's. Balance at their shared output so w_lad denotes the
        # desired LAD/EDM gradient ratio rather than an arbitrary loss ratio.
        edm_grad = torch.autograd.grad(
            edm_loss.mean(), denoised_latents, retain_graph=True, create_graph=False
        )[0].detach().float()
        lad_grad = torch.autograd.grad(
            weighted_energy.mean(), denoised_latents, retain_graph=True, create_graph=False
        )[0].detach().float()
        edm_grad_norm = edm_grad.norm()
        lad_grad_norm = lad_grad.norm()
        balance_scale = (edm_grad_norm / lad_grad_norm.clamp(min=1e-12)).clamp(
            max=self._lad_balance_max_scale
        )
        multiplier = self._w_lad * balance_scale.detach()
        grad_cosine = (edm_grad * lad_grad).sum() / (
            edm_grad_norm * lad_grad_norm
        ).clamp(min=1e-12)

        self._lad_diagnostics = {
            'lad_sigma': sigma.mean(),
            'lad_gate': gate.mean(),
            'lad_energy_raw': e_per.mean(),
            'lad_energy_weighted': weighted_energy.mean(),
            'edm_z0_grad_log10': edm_grad_norm.clamp(min=1e-30).log10(),
            'lad_z0_grad_log10': lad_grad_norm.clamp(min=1e-30).log10(),
            'lad_balance_log10': balance_scale.clamp(min=1e-30).log10(),
            'lad_multiplier_ppm': multiplier * 1e6,
            'lad_grad_cosine': grad_cosine,
        }
        return multiplier * weighted_energy.mean()
