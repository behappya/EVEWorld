#!/usr/bin/env python3
"""zones origin-A 补丁 (审阅教案可付性后的修正)。

问题: 原 build_zones 的 A 区依赖 source 容器检测 + t_depart 判定链, AgiBot 上
Pick 只有 31/391 条付得出 A 原位教案 — 而 Pick 的偷懒教案主打 A 原位残留。
修正: A 原位改锚**目标自身初始格** target_cell_0 — 目标离开初始格 (位移>3 格,
连续 2 帧确认) 后, 其半径 2 邻域中 safe 背景格 (zones==0) 升为 zone 2。
只升 0→2 (贴入偏好标签), 不动 -1/1, 安全性不变。仅 TRANSFER 类。
本机 CPU 重算, 保留 npz 的 patch/box。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agi_aug_prep as APP  # noqa: E402  (完成全部 monkey-patch + agi_build_zones)
import t4g_probe as P  # noqa: E402
from t4g_ghost_probe import cell_motion  # noqa: E402

CLEAN = APP.CLEAN
ANNO = APP.ANNO
OUT = APP.OUT
T_LAT, H_LAT, W_LAT = 24, 30, 40
NF, HPIX, WPIX = 93, 480, 640
LEAVE_DIST, ORIGIN_R, SETTLE = 3, 2, 2


def origin_patch(zones, anno):
    c0 = anno.get('target_cell_0')
    if anno.get('category') != 'TRANSFER' or not c0:
        return zones, 0
    per = anno['per_lat_frame']
    t_leave = None
    for t in range(1, T_LAT - 1):
        c, c2 = per[t].get('target_cell'), per[t + 1].get('target_cell')
        if c and c2 and max(abs(c[0] - c0[0]), abs(c[1] - c0[1])) > LEAVE_DIST \
                and max(abs(c2[0] - c0[0]), abs(c2[1] - c0[1])) > LEAVE_DIST:
            t_leave = t
            break
    if t_leave is None:
        return zones, 0
    n = 0
    gy0, gx0 = int(c0[0]), int(c0[1])
    for t in range(min(t_leave + SETTLE, T_LAT), T_LAT):
        for gy in range(max(0, gy0 - ORIGIN_R), min(H_LAT, gy0 + ORIGIN_R + 1)):
            for gx in range(max(0, gx0 - ORIGIN_R), min(W_LAT, gx0 + ORIGIN_R + 1)):
                if zones[t, gy, gx] == 0:
                    zones[t, gy, gx] = 2
                    n += 1
    return zones, n


def one(name):
    try:
        anno = json.load(open(f'{ANNO}/{name}.json'))
        fp = f'{OUT}/{name}.npz'
        # 先物化旧数组再覆盖写同一文件 (np.load 是惰性的)
        with np.load(fp) as d:
            keep = {k: d[k].copy() for k in ('patch', 'box') if k in d}
        rep = np.linspace(0, NF - 1, T_LAT).astype(int)
        frames = P.sample_frames_like_training(f'{CLEAN}/{name}.mp4', NF, HPIX, WPIX)
        zones = APP.agi_build_zones(anno, cell_motion(frames, rep))
        zones, n = origin_patch(zones, anno)
        np.savez_compressed(fp, zones=zones, **keep)
        return name, anno.get('skill', '?'), int((zones == 2).any()), n, 'ok'
    except Exception as e:  # noqa: BLE001
        return name, '?', 0, 0, f'FAIL {type(e).__name__}: {str(e)[:80]}'


def main():
    from collections import Counter
    from multiprocessing import Pool
    names = sorted(f[:-5] for f in os.listdir(ANNO)
                   if f.endswith('.json') and not f.startswith('_'))
    fixed, fails = 0, 0
    has_a = Counter()
    with Pool(16) as pool:
        for i, (name, sk, ha, n, st) in enumerate(pool.imap_unordered(one, names)):
            if st != 'ok':
                fails += 1
                print(f'  {name}: {st}', flush=True)
                continue
            has_a[sk] += ha
            fixed += int(n > 0)
            if (i + 1) % 150 == 0:
                print(f'  {i + 1}/{len(names)} origin_fixed={fixed}', flush=True)
    print(f'[originfix] done: {len(names)} clips, origin-A added in {fixed}, fail={fails}')
    print('hasA by skill:', dict(has_a))


if __name__ == '__main__':
    main()
