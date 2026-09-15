#!/usr/bin/env python3
"""SELF-CASE 训练 (70 号 A1): 自病例修复 —— 病例源从"人工贴入"换成"模型自己犯的错"。

机制 (与 CP 探针唯一差异 = 病例来源):
  images      = 病例修补版 (rollout 复制区时域回填)  <- loss 干净目标
  images_aug  = images + 贴回复制品裁块 (alpha=1)   ≈ 原 rollout <- 加噪输入
  非病例样本  = 干净 GT 照常 SFT (p_case 控制混合)。

设计要点 (对抗检查后):
  - p_aug 强制 0: 本臂绝不混人工 CP 贴入 (纯净对照)。
  - weight_map 对病例样本强制均匀 1: GT 合同权重图与 rollout 轨迹错位, 不可复用;
    复制区上权重仍由 trainer 的 w_paste 承担。
  - ref_images 随 base 重建 (帧0=rollout 首帧≈条件帧), 保持 pair 内部自洽。
  - 病例区外 target=模型自身 rollout -> 梯度近零 (自蒸馏中性), 唯一强信号=删除复制品。

trainer 完全复用 T4GAugTrainer (双 VAE 编码/干净目标/w_paste/aug-sentinel)。
"""
from __future__ import annotations

import os

import numpy as np
import torch
from giga_train import TRANSFORMS

from .t4g_aug_paste import T_LAT, H_LAT
from .t4g_aug_trainer import T4GAugTrainer, T4GAugTransform

W_LAT = 48

DEFAULT_CASE_DIR = '/data/datasets/gagi/eve_v2_outputs/selfcase/mine_round0/case_bank'


@TRANSFORMS.register
class T4GSelfCaseTransform(T4GAugTransform):
    """病例可用时按 p_case 用 (修补版 rollout + 固定贴回计划) 替换样本; 否则干净 GT。"""

    def __init__(self, *args, case_dir=DEFAULT_CASE_DIR, p_case=0.5, case_cache=3, **kw):
        kw['p_aug'] = 0.0                      # 关闭人工 CP 通道 (铁律)
        super().__init__(*args, **kw)
        self.case_dir = case_dir
        self.p_case = float(os.environ.get('T4G_P_CASE', p_case))
        self.case_cache_cap = int(case_cache)
        self.case_index = {}
        if os.path.isdir(case_dir):
            for f in sorted(os.listdir(case_dir)):
                if f.endswith('.npz'):
                    self.case_index.setdefault(f.split('__')[0], []).append(
                        os.path.join(case_dir, f))
        self._case_data = {}
        if int(os.environ.get('RANK', '0')) == 0:
            n = sum(len(v) for v in self.case_index.values())
            print(f'[selfcase] case bank: {n} cases / {len(self.case_index)} vids '
                  f'@ {case_dir} | p_case={self.p_case}', flush=True)

    def _load_case(self, fp):
        if fp not in self._case_data:
            if len(self._case_data) >= self.case_cache_cap:
                self._case_data.pop(next(iter(self._case_data)))
            d = np.load(fp)
            self._case_data[fp] = dict(base=d['base'], patch=d['patch'], boxes=d['boxes'],
                                       frames=d['frames'], lat=d['lat'],
                                       alpha=float(d['alpha']))
        return self._case_data[fp]

    @staticmethod
    def _norm(u8):
        # 与 GigaWorld0Transform 一致: /255 -> Normalize(0.5,0.5) => x/127.5-1
        return torch.from_numpy(u8.astype(np.float32)).div_(127.5).sub_(1.0)

    def __call__(self, data_dict):
        out = super().__call__(data_dict)      # p_aug=0 -> 干净 GT + vid + weight_map
        paths = self.case_index.get(out['vid'])
        if not paths or self.rng.random() >= self.p_case:
            return out
        d = self._load_case(paths[int(self.rng.integers(len(paths)))])
        base = self._norm(d['base']).permute(0, 3, 1, 2).contiguous()   # (T,C,H,W)
        pix_mask = (out['ref_images'].abs().sum(dim=(1, 2, 3)) > 0)     # 参考帧位形
        m = pix_mask.view(-1, 1, 1, 1).to(base.dtype)
        out['images'] = base
        out['ref_images'] = base * m
        out['paste_patch'] = self._norm(d['patch']).permute(2, 0, 1).contiguous()
        out['paste_boxes'] = torch.from_numpy(d['boxes']).long()
        out['paste_frames'] = torch.from_numpy(d['frames']).long()
        out['paste_lat'] = torch.from_numpy(d['lat']).long()
        out['paste_alpha'] = torch.tensor([d['alpha']], dtype=torch.float32)
        out['has_paste'] = 1
        out['weight_map'] = torch.ones(T_LAT, H_LAT, W_LAT)   # 合同图错位 -> 均匀
        return out


class T4GSelfCaseTrainer(T4GAugTrainer):
    """行为与 T4GAugTrainer 完全一致; 独立类名仅为 config runners 指向本模块。"""
