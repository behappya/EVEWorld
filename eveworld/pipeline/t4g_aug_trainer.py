#!/usr/bin/env python3
"""EVE · 合成复制增广 trainer (44 号阶段1', E10 机制修复: 教"删除"技能)。

机制
----
输入投毒: 把帧0 目标物 patch 贴到"确定该空"的时空区 (E8 安全掩码), 加噪后喂模型;
**loss 目标 = 干净原视频** —— 修复出贴入物即吃满重建误差, 只有擦掉才得分。
E10 实测初始 retention 0.92-0.99 → 初始梯度信号最大化。

与 T4GCorrTrainer 的差异 (骨架同源):
  1. T4GAugTransform: 解析 vid (同 corr) + 读 aug_assets 采样贴入计划 (patch/位置/窗口/alpha,
     采样约束: 贴入 patch 覆盖的所有格在整个窗口内 zones>=0; 帧0 永不贴; K 次重试失败则放弃)。
     patch 的缩放/模糊在 transform 完成 (CPU), 混合在 trainer GPU 上做。
  2. forward_step: 双 VAE 编码 (干净 latent 作 loss 目标, 投毒 latent 走加噪+前向);
     手算 EDM loss (对照干净 latent, 贴入区上权重 w_paste); 干净样本与父类逐位等价。
  3. 哨兵: [aug-sentinel] applied / l_paste(贴入区对干净目标的裸误差, 应降) /
     delta_paste(贴入幅度, 恒定参照) / ret_proxy=sqrt(l_paste/delta)(≈retention, 应从~0.95降) /
     l_rest(非贴入区, 应平=不换病) / param_disp。
对抗性封堵 (44 号 §四): 贴点 40%B/20%A/40%bg 防位置捷径; α∈[0.4,1]+缩放+模糊防"看贴图痕迹";
帧0 干净=清单锚; 50% 样本不投毒 + l_rest 哨兵防"见啥删啥"。
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import torch
from giga_train import TRANSFORMS

from giga_world_0.giga_world_0_trainer import GigaWorld0Trainer
from giga_world_0.giga_world_0_transforms import GigaWorld0Transform

from .t4g_corr_trainer import ParamDisplacementProbe

DEFAULT_ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
DEFAULT_ASSETS = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets'
from .t4g_aug_paste import (sample_paste_plan, prep_patch, paste_lat_region,
                            sample_follow_plan, build_follow_boxes, static_boxes,
                            T_LAT, H_LAT, W_LAT, NF, HPIX, WPIX, CELL_PX, VAE_SP)


# ====================================================================================
#  transform: vid 解析 (同 corr) + 贴入计划
# ====================================================================================
@TRANSFORMS.register
class T4GAugTransform(GigaWorld0Transform):
    """输出追加: vid / has_paste / paste_patch(3,hp,wp 归一化) / paste_box / paste_frames
    (像素帧窗口) / paste_lat(6 ints) / paste_alpha。无资产或采样失败 -> has_paste=0。"""

    def __init__(self, num_frames, height, width, image_cfg, fps=16,
                 idx2vid_path=os.path.join(DEFAULT_ANNO_DIR, '_packidx2vid.json'),
                 assets_dir=DEFAULT_ASSETS, p_aug=0.5, zone_bias='0.4,0.2,0.4', seed=20260721,
                 p_corridor=0.0, p_alien=0.0, anno_dir=DEFAULT_ANNO_DIR,
                 wmap_dir=None, wmap_values=None, strict_mapping=False,
                 strict_assets=False, strict_wmap=False):
        super().__init__(num_frames=num_frames, height=height, width=width,
                         image_cfg=image_cfg, fps=fps)
        self.idx2vid = {}
        if idx2vid_path and os.path.exists(idx2vid_path):
            self.idx2vid = {int(k): str(v) for k, v in json.load(open(idx2vid_path)).items()}
        self.strict_mapping = bool(strict_mapping)
        self.strict_assets = bool(strict_assets)
        self.strict_wmap = bool(strict_wmap)
        if self.strict_mapping and not self.idx2vid:
            raise FileNotFoundError(f'idx2vid mapping missing or empty: {idx2vid_path}')
        self.assets_dir = assets_dir
        self.p_aug = float(p_aug)
        self.zone_bias = tuple(float(x) for x in str(zone_bias).split(','))
        # v2: 护送贴 (关闭夹爪出生通道) + 外来物种贴 (治换物); 默认 0 = v1 行为
        self.p_corridor = float(p_corridor)
        self.p_alien = float(p_alien)
        self.anno_dir = anno_dir
        self.alien_bank = sorted(f[:-4] for f in os.listdir(assets_dir)
                                 if f.endswith('.npz')) if os.path.isdir(assets_dir) else []
        default_wmap = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/weightmap_cache'
        self.wmap_dir = os.environ.get('T4G_WMAP_DIR', wmap_dir or default_wmap)
        self.wmap_values = None
        if wmap_values is not None:
            values = str(wmap_values).split(',') if isinstance(wmap_values, str) else wmap_values
            self.wmap_values = {float(x) for x in values}
        self.rng = np.random.default_rng(seed + int(os.environ.get('RANK', '0')))
        self._cache = {}
        self._anno_cache = {}
        self._wmap_cache = {}
        self.rep = np.linspace(0, NF - 1, T_LAT).astype(int)

    def _target_cells(self, vid):
        if vid not in self._anno_cache:
            fp = os.path.join(self.anno_dir, f'{vid}.json')
            cells = None
            if os.path.exists(fp):
                a = json.load(open(fp))
                cells = [fr.get('target_cell') for fr in a.get('per_lat_frame', [])]
                if len(cells) != T_LAT:
                    cells = None
            self._anno_cache[vid] = cells
        return self._anno_cache[vid]

    def _asset(self, vid):
        if vid not in self._cache:
            fp = os.path.join(self.assets_dir, f'{vid}.npz')
            if os.path.exists(fp):
                d = np.load(fp)
                self._cache[vid] = dict(zones=d['zones'],
                                        patch=d['patch'] if 'patch' in d else None)
            else:
                if self.strict_assets:
                    raise FileNotFoundError(f'augmentation asset missing for vid={vid}: {fp}')
                self._cache[vid] = None
        return self._cache[vid]

    def _weightmap(self, vid):
        """读预计算权重图 (24,30,48); 缺失回退全 1 (均匀)。"""
        if vid not in self._wmap_cache:
            fp = os.path.join(self.wmap_dir, f'{vid}.npy')
            if os.path.exists(fp):
                wm = np.load(fp)
                if wm.shape != (T_LAT, H_LAT, W_LAT):
                    raise ValueError(f'weightmap shape invalid for vid={vid}: {wm.shape} at {fp}')
                if not np.isfinite(wm).all():
                    raise ValueError(f'weightmap contains non-finite values for vid={vid}: {fp}')
                if self.wmap_values is not None:
                    actual = {float(x) for x in np.unique(wm)}
                    if not actual.issubset(self.wmap_values):
                        raise ValueError(
                            f'weightmap values invalid for vid={vid}: {sorted(actual)}; '
                            f'expected subset of {sorted(self.wmap_values)} at {fp}')
                self._wmap_cache[vid] = wm
            else:
                if self.strict_wmap:
                    raise FileNotFoundError(f'weightmap missing for vid={vid}: {fp}')
                self._wmap_cache[vid] = None
        wm = self._wmap_cache[vid]
        if wm is None:
            return torch.ones(T_LAT, H_LAT, W_LAT)
        return torch.from_numpy(wm.astype(np.float32))

    def __call__(self, data_dict):
        out = super().__call__(data_dict)
        di = data_dict.get('data_index', None)
        vid = self.idx2vid.get(int(di)) if di is not None else None
        if self.strict_mapping and vid is None:
            raise KeyError(f'unmapped data_index={di!r}; idx2vid has {len(self.idx2vid)} entries')
        out['vid'] = str(vid) if vid is not None else '-1'
        out['has_paste'] = 0
        out['paste_patch'] = torch.zeros(3, 1, 1)
        out['paste_lat'] = torch.zeros(6, dtype=torch.long)
        out['paste_boxes'] = torch.zeros(NF, 4, dtype=torch.long)
        out['paste_frames'] = torch.zeros(2, dtype=torch.long)
        out['paste_alpha'] = torch.zeros(1)
        out['weight_map'] = self._weightmap(out['vid'])            # (24,30,48) 合同权重
        asset = self._asset(out['vid']) if vid is not None else None
        if asset is None or asset['patch'] is None or self.rng.random() >= self.p_aug:
            return out
        # 素材: 本尊复制品 或 外来物种 (v2, 治换物)
        patch_src = asset['patch']
        if self.p_alien > 0 and self.alien_bank and self.rng.random() < self.p_alien:
            other = self.alien_bank[int(self.rng.integers(len(self.alien_bank)))]
            if other != out['vid']:
                oa = self._asset(other)
                if oa is not None and oa['patch'] is not None:
                    patch_src = oa['patch']
        # 通道: 护送贴 (v2, 关闭夹爪出生通道) 或 静态贴 (v1)
        plan, boxes, lat, p0, p1 = None, None, None, 0, 0
        if self.p_corridor > 0 and self.rng.random() < self.p_corridor:
            tc = self._target_cells(out['vid'])
            if tc:
                plan = sample_follow_plan(tc, patch_src.shape[:2], self.rng)
                if plan is not None:
                    boxes, lat, p0, p1 = build_follow_boxes(tc, plan, self.rep)
                    if boxes is None:
                        plan = None
        if plan is None:
            plan = sample_paste_plan(asset['zones'], patch_src.shape[:2], self.rng,
                                     zone_bias=self.zone_bias)
            if plan is None:
                return out
            boxes, p0, p1 = static_boxes(plan, self.rep)
            lat = paste_lat_region(plan)
        p = prep_patch(patch_src, plan)                            # uint8 (hp,wp,3)
        patch_norm = torch.from_numpy(p.astype(np.float32)).permute(2, 0, 1) / 255.0
        patch_norm = (patch_norm - 0.5) / 0.5
        out['has_paste'] = 1
        out['paste_patch'] = patch_norm
        out['paste_boxes'] = torch.from_numpy(boxes).long()
        out['paste_frames'] = torch.tensor([p0, p1], dtype=torch.long)
        out['paste_lat'] = torch.tensor([lat['lt_lo'], lat['lt_hi'], lat['ly0'], lat['ly1'],
                                         lat['lx0'], lat['lx1']], dtype=torch.long)
        out['paste_alpha'] = torch.tensor([plan['alpha']])
        return out


# ====================================================================================
#  trainer
# ====================================================================================
class T4GAugTrainer(GigaWorld0Trainer):
    """合成复制增广: 投毒输入 + 干净目标 + 贴入区上权重 + 五量哨兵。"""

    def get_models(self, model_config):
        model = super().get_models(model_config)
        assert self.train_mode == 'full', 'aug 训练用全参 (与 corr/round0 同规格)'
        self.w_paste = float(os.environ.get('T4G_W_PASTE', model_config.get('t4g_w_paste', 4.0)))
        self.checkpoint_start_step = int(os.environ.get('T4G_CHECKPOINT_START_STEP',
                                                        self.kwargs.get('checkpoint_start_step', 0)))
        self.param_probe = ParamDisplacementProbe(model['transformer'])
        self._stat_buffer = []
        self._logged_step = -1
        self.print('[t4g-aug] w_paste=%.1f ckpt_start=%d probe_scalars=%d'
                   % (self.w_paste, self.checkpoint_start_step, self.param_probe.num_scalars))
        return model

    # ---------------------------- forward ---------------------------------
    def forward_step(self, batch_dict):
        import functools
        transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images']                              # (B,T,C,H,W) 干净
        prompt_embeds = batch_dict['prompt_embeds']
        B = images.shape[0]
        padding_mask = torch.zeros((B, 1, images.shape[-2], images.shape[-1]),
                                   dtype=self.dtype, device=self.device)
        fps = batch_dict['fps'][0]

        has_paste = batch_dict['has_paste']                        # list/tensor (B,)
        hp = [int(x) for x in has_paste]
        latents_clean = self.forward_vae(images)                   # loss 目标

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

        ref_images = batch_dict['ref_images']                      # 干净 (帧0 未投毒)
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

        model_pred = transformer(
            x=input_latents.to(self.dtype),
            timesteps=timesteps.to(self.dtype),
            crossattn_emb=prompt_embeds.to(self.dtype),
            padding_mask=padding_mask,
            fps=fps,
        )
        denoised = self.edm_loss.denoise(model_pred.float()).reshape(latents_clean.shape)
        denoised = ref_masks * ref_latents + (1 - ref_masks) * denoised

        # ---- 手算 EDM loss: 对照 **干净** latent + 贴入区上权重 ----
        # (干净样本与父类 compute_loss 逐位等价: weight*(denoised-clean)^2 全元素均值)
        weight = self.edm_loss.get_loss_weight().reshape(B).float()          # (B,)
        err = (denoised.float() - latents_clean.float()) ** 2                # (B,z,T,h,w)
        w_map = torch.ones_like(err)
        stats = torch.zeros(7, device=self.device)
        # stats: [0]applied [1]sum_lpaste [2]sum_delta [3]cnt_paste [4]sum_lrest [5]cnt_all [6]sum_ret
        for b in range(B):
            stats[5] += 1
            if hp[b]:
                lt0, lt1, ly0, ly1, lx0, lx1 = [int(v) for v in batch_dict['paste_lat'][b]]
                w_map[b, :, lt0:lt1, ly0:ly1, lx0:lx1] = self.w_paste
                reg = err[b, :, lt0:lt1, ly0:ly1, lx0:lx1]
                delta = ((latents_in - latents_clean).float() ** 2)[b, :, lt0:lt1, ly0:ly1, lx0:lx1]
                l_p = float(reg.detach().mean()); d_p = float(delta.detach().mean())
                stats[0] += 1; stats[1] += l_p; stats[2] += d_p; stats[3] += 1
                if d_p > 1e-8:
                    stats[6] += min(2.0, (l_p / d_p) ** 0.5)
                m = torch.ones_like(err[b], dtype=torch.bool)
                m[:, lt0:lt1, ly0:ly1, lx0:lx1] = False
                stats[4] += float(err[b].detach()[m].mean())
            else:
                stats[4] += float(err[b].detach().mean())
        l_diff = (weight.view(B, 1, 1, 1, 1) * err * w_map).mean(dim=(1, 2, 3, 4))   # (B,)

        gathered = self.accelerator.gather(stats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._stat_buffer.append(gathered)
        return dict(l_diff=l_diff)

    # --------------------------- sentinels --------------------------------
    def print_step(self):
        super().print_step()
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        if self.cur_step == self._logged_step:
            return
        self._logged_step = self.cur_step
        agg = torch.stack(self._stat_buffer).sum(0) if self._stat_buffer else torch.zeros(7)
        self._stat_buffer = []

        def _r(n, d):
            return (agg[n] / agg[d]).item() if agg[d] > 0 else float('nan')

        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        disp = self.param_probe.displacement(model['transformer'])
        msg = ('[aug-sentinel] step=%d applied=%d/%d | l_paste=%.4e delta=%.4e ret_proxy=%.3f '
               '| l_rest=%.4e | param_disp_l2=%.4e | w_paste=%.1f'
               % (self.cur_step, int(agg[0]), int(agg[5]), _r(1, 3), _r(2, 3), _r(6, 3),
                  _r(4, 5), disp, self.w_paste))
        self.logger.info(msg)
        print(msg, flush=True)

    def save_checkpoint_step(self):
        if self.cur_step < self.checkpoint_start_step:
            return
        super().save_checkpoint_step()
