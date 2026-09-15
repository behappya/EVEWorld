#!/usr/bin/env python3
"""EVE · 联合 trainer: L_id (E4/corr 探针验证) + 合成复制增广 (E10/E11 验证)。44 号阶段2。

分工 (金标签两大主病, 互补不重叠):
  L_id   治瞬移/形变 —— block22 前后帧接力 CE, 仅 sigma∈[0.2,0.5], λ warmup 0→0.5;
  增广   治复制/重生 —— 投毒输入+干净目标, 全 sigma, 贴入区 w_paste 上权重。
组合方式: 继承 T4GCorrTrainer (拿 hook/L_id/_compute_corr/哨兵), forward_step 换成
增广式前向 (双 VAE 编码 + 手算干净目标 loss), 之后照旧调 _compute_corr。
L_change λ=0 (E7 机制真但训练时无罪可罚, 只留诊断打印)。

对抗性检查点 (写码时已核):
  - 贴入安全区豁免走廊 -> 假货不进 L_id 追踪窗口中心; 目标格恒为 GDINO 真实格;
  - 两侧 stats gather 均无条件执行 (collective 对称性);
  - _ensure_hook 在前向最先调用 (corr 的惰性 hook)。
哨兵: [aug-sentinel] + [t4g-sentinel] 双打印。
"""
from __future__ import annotations

import functools
import os

import torch

from .t4g_corr_trainer import T4GCorrTrainer
from .t4g_aug_trainer import T4GAugTransform  # noqa: F401  (注册 transform, config 引用)


class T4GJointTrainer(T4GCorrTrainer):

    def get_models(self, model_config):
        model = super().get_models(model_config)      # corr 全套初始化 (annos/hook/λ/哨兵)
        self.w_paste = float(os.environ.get('T4G_W_PASTE', model_config.get('t4g_w_paste', 4.0)))
        levels = os.environ.get('T4G_REGION_LEVELS', model_config.get(
            't4g_region_levels', '0.5,2.0,3.0,4.0,6.0'))
        names = os.environ.get('T4G_REGION_NAMES', model_config.get(
            't4g_region_names', 'bg0.5x,empty2x,gripB3x,obj4x,trans6x'))
        self.region_levels = tuple(float(x) for x in str(levels).split(','))
        self.region_names = tuple(x.strip() for x in str(names).split(','))
        assert len(self.region_levels) == len(self.region_names) > 0, \
            't4g_region_levels and t4g_region_names must have equal non-zero length'
        expected_samples = int(model_config.get('t4g_expected_samples', 0))
        if expected_samples:
            anno_vids = set(self.annos)
            mapped_vids = set(self.idx2vid.values())
            assert len(self.annos) == expected_samples, \
                f'expected {expected_samples} annos, got {len(self.annos)}'
            assert len(self.idx2vid) == expected_samples, \
                f'expected {expected_samples} idx2vid rows, got {len(self.idx2vid)}'
            assert anno_vids == mapped_vids, \
                f'anno/idx2vid mismatch: anno_only={sorted(anno_vids - mapped_vids)} ' \
                f'map_only={sorted(mapped_vids - anno_vids)}'
        self._aug_stat_buffer = []
        self.print('[t4g-joint] w_paste=%.1f regions=%s wmap=%s + '
                   'L_id(band=[%.2f,%.2f], λ=%.2f) + L_change λ=%.2f(诊断)'
                   % (self.w_paste,
                      ','.join(f'{n}:{v:g}x' for n, v in zip(self.region_names,
                                                             self.region_levels)),
                      model_config.get('t4g_wmap_dir', os.environ.get('T4G_WMAP_DIR', '<transform-config>')),
                      self.sigma_lo, self.sigma_hi, self.lambda_base[0], self.lambda_base[1]))
        return model

    # ---------------------------- forward ---------------------------------
    def forward_step(self, batch_dict):
        self._ensure_hook()
        transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images']                              # (B,T,C,H,W) 干净
        prompt_embeds = batch_dict['prompt_embeds']
        B = images.shape[0]
        padding_mask = torch.zeros((B, 1, images.shape[-2], images.shape[-1]),
                                   dtype=self.dtype, device=self.device)
        fps = batch_dict['fps'][0]

        has_paste = batch_dict.get('has_paste', [0] * B)
        hp = [int(x) for x in has_paste]
        latents_clean = self.forward_vae(images)

        if any(hp):
            images_aug = images.clone()
            for b in range(B):
                if not hp[b]:
                    continue
                p0, p1 = [int(v) for v in batch_dict['paste_frames'][b]]
                a = float(batch_dict['paste_alpha'][b])
                patch = batch_dict['paste_patch'][b].to(images_aug.device, images_aug.dtype)
                boxes = batch_dict['paste_boxes'][b]
                for pp in range(p0, p1):
                    y0, x0, y1, x1 = [int(v) for v in boxes[pp]]
                    images_aug[b, pp, :, y0:y1, x0:x1] = (
                        a * patch + (1 - a) * images_aug[b, pp, :, y0:y1, x0:x1])
            latents_in = self.forward_vae(images_aug)
        else:
            latents_in = latents_clean

        input_latents, timesteps = self.edm_loss.add_noise(latents_in)
        sigma = self.edm_loss.sigma.detach().reshape(-1)           # (B,) 供 L_id 时间档

        ref_images = batch_dict['ref_images']
        ref_masks = batch_dict['ref_masks'].to(self.dtype)
        ref_latents = self.forward_vae(ref_images)

        augment_sigma = torch.tensor([0.0001], device=ref_latents.device, dtype=latents_clean.dtype)
        while len(augment_sigma.shape) < len(ref_latents.shape):
            augment_sigma = augment_sigma.unsqueeze(-1)

        input_latents = ref_masks * ref_latents + (1 - ref_masks) * input_latents
        input_masks = ref_masks.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])
        input_latents = torch.cat([input_latents, input_masks], dim=1)
        timesteps = timesteps.view(1, 1, 1, 1, 1).expand(B, -1, latents_clean.size(2), -1, -1)
        t_conditioning = augment_sigma / (augment_sigma + 1)
        timesteps = ref_masks * t_conditioning + (1 - ref_masks) * timesteps

        self._id_block_out = None
        self._change_block_out = None
        model_pred = transformer(
            x=input_latents.to(self.dtype),
            timesteps=timesteps.to(self.dtype),
            crossattn_emb=prompt_embeds.to(self.dtype),
            padding_mask=padding_mask,
            fps=fps,
        )
        denoised = self.edm_loss.denoise(model_pred.float()).reshape(latents_clean.shape)
        denoised = ref_masks * ref_latents + (1 - ref_masks) * denoised

        # ---- 增广侧: 手算 EDM loss, 目标=干净 latent, 合同权重图 + 贴入区上权重 ----
        weight = self.edm_loss.get_loss_weight().reshape(B).float()
        err = (denoised.float() - latents_clean.float()) ** 2   # (B,z,T,60,96)
        # 合同权重图 (B,24,30,48) -> 空间 2x 上采样 -> (B,24,60,96) -> 广播 z 通道
        cwmap = batch_dict['weight_map'].to(err.device, err.dtype)
        cwmap = cwmap.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        assert cwmap.shape[0] == B and cwmap.shape[1:] == err.shape[2:], \
            ('weight_map 尺寸不符', cwmap.shape, err.shape)
        w_map = cwmap.unsqueeze(1).expand(-1, err.shape[1], -1, -1, -1).contiguous()
        # astat: [0-6]=原七项 + 每个配置权重档的 (加权loss和, 裸误差和, 格数)
        astat = torch.zeros(7 + 3 * len(self.region_levels), device=self.device)
        for b in range(B):
            astat[5] += 1
            if hp[b]:
                lt0, lt1, ly0, ly1, lx0, lx1 = [int(v) for v in batch_dict['paste_lat'][b]]
                # 贴入区取 max(合同权重, w_paste): 不降低合同已给的高权重
                w_map[b, :, lt0:lt1, ly0:ly1, lx0:lx1].clamp_(min=self.w_paste)
                reg = err[b, :, lt0:lt1, ly0:ly1, lx0:lx1]
                delta = ((latents_in - latents_clean).float() ** 2)[b, :, lt0:lt1, ly0:ly1, lx0:lx1]
                l_p = float(reg.detach().mean()); d_p = float(delta.detach().mean())
                astat[0] += 1; astat[1] += l_p; astat[2] += d_p; astat[3] += 1
                if d_p > 1e-8:
                    astat[6] += min(2.0, (l_p / d_p) ** 0.5)
                m = torch.ones_like(err[b], dtype=torch.bool)
                m[:, lt0:lt1, ly0:ly1, lx0:lx1] = False
                astat[4] += float(err[b].detach()[m].mean())
            else:
                astat[4] += float(err[b].detach().mean())
        # 记录贴入区 clamp 后的真实档位，再逐样本归一化到均值 1。
        region_map = w_map[:, 0].detach()
        w_map = w_map / w_map.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
        l_diff = (weight.view(B, 1, 1, 1, 1) * err * w_map).mean(dim=(1, 2, 3, 4))

        # ---- 分区账本: 按合同档位实测 (加权loss份额 + 裸误差), mask 来自归一化前的档位值 ----
        with torch.no_grad():
            err_w = err * w_map                                     # 实际参与梯度的加权误差
            for i, lev in enumerate(self.region_levels):
                m = ((region_map - lev).abs() < 1e-3).unsqueeze(1)  # 广播 z 通道
                astat[7 + i * 3 + 0] += float((err_w * m).sum())
                astat[7 + i * 3 + 1] += float((err * m).sum())
                astat[7 + i * 3 + 2] += float(m.expand_as(err).sum())

        gathered = self.accelerator.gather(astat.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._aug_stat_buffer.append(gathered)

        # ---- L_id 侧 (corr 机制原样; L_change λ=0 只出诊断) ----
        corr = self._compute_corr(batch_dict, sigma)
        losses = dict(l_diff=l_diff)
        losses.update(corr)
        return losses

    # --------------------------- sentinels --------------------------------
    def print_step(self):
        super().print_step()                                       # 基类 + [t4g-sentinel]
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        stat_size = 7 + 3 * len(self.region_levels)
        agg = (torch.stack(self._aug_stat_buffer).sum(0)
               if self._aug_stat_buffer else torch.zeros(stat_size))
        self._aug_stat_buffer = []

        def _r(n, d):
            return (agg[n] / agg[d]).item() if agg[d] > 0 else float('nan')

        msg = ('[aug-sentinel] step=%d applied=%d/%d | l_paste=%.4e delta=%.4e ret_proxy=%.3f '
               '| l_rest=%.4e | w_paste=%.1f'
               % (self.cur_step, int(agg[0]), int(agg[5]), _r(1, 3), _r(2, 3), _r(6, 3),
                  _r(4, 5), self.w_paste))
        self.logger.info(msg)
        print(msg, flush=True)

        # ---- 分区账本打印: 每档 加权loss份额% + 裸单格误差 ----
        if agg.numel() >= stat_size:
            tot_w = sum(float(agg[7 + i * 3]) for i in range(len(self.region_levels)))
            parts = []
            for i, (nm, lev) in enumerate(zip(self.region_names, self.region_levels)):
                ws, es, cs = (float(agg[7 + i * 3 + k]) for k in range(3))
                share = ws / tot_w * 100 if tot_w > 0 else float('nan')
                merr = es / cs if cs > 0 else float('nan')
                parts.append(f'{nm}{lev:g}x:{share:4.1f}%/err{merr:.4f}')
            msg2 = f'[region-sentinel] step={self.cur_step} 加权份额/裸误差 | ' + ' | '.join(parts)
            self.logger.info(msg2)
            print(msg2, flush=True)
