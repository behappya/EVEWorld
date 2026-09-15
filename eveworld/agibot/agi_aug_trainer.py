#!/usr/bin/env python3
"""AgiBot skill 条件化增广 transform + trainer 壳 (方案 Phase 4.2)。

AgiAugTransform(T4GAugTransform) 的 per-sample 差异 (读 anno 的 skill/category):
- TRANSFER 按 skill 定 zone_bias(B,A,bg):
    Pick (391条, 多数无 B) -> (0.0,0.5,0.5)  偷懒教案=A 原位残留 (物已在手原位还有)
    Place/HandOver/Insert  -> (0.5,0.2,0.3)  偷懒教案=提前出现在目的地
    Push/Pull              -> (0.2,0.4,0.4)
- STATE (Pour/Open/Close/Hold...): 提前终态无法用贴块定义 -> 物体贴禁用,
  仅 p_aug_state=0.15 的纯背景贴 (保留"别幻觉多余物"信号), 教案由 W_STATE+L_id 承载。
- 双臂走廊已在 zones 端挖为 -1 (agi_aug_prep), 贴块永不上臂。

AgiJointTrainer = T4GJointTrainer 原样 (grid-agnostic); 子类仅为 config runners
指到本模块, import 时完成 AgiAugTransform 注册。
"""
from __future__ import annotations

import json
import os

from giga_train import TRANSFORMS

from ..pipeline.t4g_aug_trainer import T4GAugTransform
from ..pipeline.t4g_joint_trainer import T4GJointTrainer

SKILL_ZONE_BIAS = {
    'Pick': (0.0, 0.5, 0.5),
    'Place': (0.5, 0.2, 0.3),
    'HandOver': (0.5, 0.2, 0.3),
    'Insert': (0.5, 0.2, 0.3),
    'Push': (0.2, 0.4, 0.4),
    'Pull': (0.2, 0.4, 0.4),
}


@TRANSFORMS.register
class AgiAugTransform(T4GAugTransform):

    def __init__(self, *args, p_aug_state=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_aug_state = float(p_aug_state)
        self._skill_cache = {}

    def _skill_meta(self, vid):
        if vid not in self._skill_cache:
            fp = os.path.join(self.anno_dir, f'{vid}.json')
            meta = None
            if os.path.exists(fp):
                a = json.load(open(fp))
                meta = {'skill': a.get('skill'), 'category': a.get('category', 'TRANSFER')}
            self._skill_cache[vid] = meta
        return self._skill_cache[vid]

    def __call__(self, data_dict):
        di = data_dict.get('data_index', None)
        vid = self.idx2vid.get(int(di)) if di is not None else None
        meta = self._skill_meta(str(vid)) if vid is not None else None
        # dataloader worker 内单线程, 临时改 self 再还原是安全的
        orig_p, orig_zb = self.p_aug, self.zone_bias
        if meta is not None:
            if meta['category'] == 'STATE':
                self.p_aug = self.p_aug_state
                self.zone_bias = (0.0, 0.0, 1.0)
            else:
                self.zone_bias = SKILL_ZONE_BIAS.get(meta['skill'], orig_zb)
        try:
            return super().__call__(data_dict)
        finally:
            self.p_aug, self.zone_bias = orig_p, orig_zb


class AgiJointTrainer(T4GJointTrainer):
    """与 T4GJointTrainer 逐位等价; 存在的意义是 runners 指向本模块触发注册。"""
