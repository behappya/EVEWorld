#!/usr/bin/env python3
"""T4G-AUG-PREP: 合成复制增广的离线资产 (44 号阶段1', 每视频一次, 训练时零 GDINO)。

产出 aug_assets/<vid>.npz:
  patch  (hp,wp,3) uint8   帧0 目标物 GDINO 框裁剪 (贴入素材, 同视频同光照)
  box    (4,) int          帧0 目标物框 (y0,x0,y1,x1)
  zones  (24,30,48) int8   贴入安全区标注: -1 不安全 / 0 背景 / 1 B区(到达前) / 2 A原位(拿走后)
zones 规则 (E8 定档 + E9 走廊):
  安全 = 窗口静止(theta=3, Delta=2) 且 非 GDINO 走廊(目标轨迹 ±2 帧膨胀 2 格);
  B 格仅 t < 0.75*t_arrival 标 1, 之后一律 -1 (临近/已到达, 贴入会与真物冲突);
  A 原位 t >= t_depart+2 标 2 (respawn 教案位)。
帧 0 是清单锚, 训练侧永不贴 (采样约束 t_s>=2, 双保险 zones[0]=-1)。
用法 (kjob 8 卡分片, 同 t4g_detect): python t4g_aug_prep.py --shard-index i --num-shards n
"""
import argparse
import json
import os

import numpy as np

import t4g_probe as P
from t4g_ghost_probe import cell_motion, T_LAT, H_LAT, W_LAT, HPIX, WPIX, NF, CELL_PX, ANNO_DIR, VIDEO_ROOT
from t4g_empty_map import window_static

OUT_DIR_DEFAULT = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets'
THETA, DELTA = 3.0, 2
CORR_TFRAMES, CORR_R = 2, 2          # 走廊: 访问时刻 ±2 帧, 空间膨胀 2 格


def build_zones(anno, motion):
    pm = window_static(motion, THETA, DELTA)                       # (T,30,48) bool
    per = anno['per_lat_frame']
    t_arr = anno['t_arrival'] if anno.get('gate_enabled') else None
    # 走廊: 时刻 t 豁免 [t-2,t+2] 内目标访问过的格 (膨胀 r=2)
    visited_at = [set() for _ in range(T_LAT)]
    for t, fr in enumerate(per):
        if fr['target_cell']:
            visited_at[t].add(tuple(fr['target_cell']))
    corridor = []
    for t in range(T_LAT):
        cells = set()
        for t2 in range(max(0, t - CORR_TFRAMES), min(T_LAT, t + CORR_TFRAMES + 1)):
            cells |= visited_at[t2]
        cor = np.zeros((H_LAT, W_LAT), bool)
        for (gy, gx) in cells:
            cor[max(0, gy - CORR_R):gy + CORR_R + 1, max(0, gx - CORR_R):gx + CORR_R + 1] = True
        corridor.append(cor)
    # A 原位起点
    a_cells = set(map(tuple, anno.get('a_cells', [])))
    t_depart = None
    was_in = False
    for t, fr in enumerate(per):
        c = tuple(fr['target_cell']) if fr['target_cell'] else None
        if c and c in a_cells:
            was_in = True
        elif c and was_in:
            t_depart = t
            break

    zones = np.full((T_LAT, H_LAT, W_LAT), -1, np.int8)
    for t in range(1, T_LAT):
        b_set = set(map(tuple, per[t]['b_cells']))
        safe = pm[t] & ~corridor[t]
        for gy in range(H_LAT):
            for gx in range(W_LAT):
                if not safe[gy, gx]:
                    continue
                c = (gy, gx)
                if t_arr is not None and c in b_set:
                    zones[t, gy, gx] = 1 if t < 0.75 * t_arr else -1
                elif t_depart is not None and t >= t_depart + 2 and c in a_cells:
                    zones[t, gy, gx] = 2
                else:
                    zones[t, gy, gx] = 0
    zones[0] = -1                                                   # 帧0 清单锚, 永不贴
    return zones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--anno-dir', default=ANNO_DIR)
    ap.add_argument('--video-root', default=VIDEO_ROOT)
    ap.add_argument('--vids', nargs='*', default=None,
                    help='Optional explicit video IDs; default is every numeric anno JSON.')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    vids = (sorted({str(v) for v in a.vids}, key=lambda v: (0, int(v)) if v.isdigit() else (1, v)) if a.vids else
            sorted([f[:-5] for f in os.listdir(a.anno_dir) if f[0].isdigit()], key=int))
    vids = vids[a.shard_index::a.num_shards]
    if not vids:
        print(f'[aug-prep] shard {a.shard_index}/{a.num_shards} empty', flush=True)
        return
    from t4g_gdino import GDinoLocator
    from t4g_detect import detect_all
    loc = GDinoLocator(device=a.device)
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    for vid in vids:
        try:
            anno = json.load(open(os.path.join(a.anno_dir, f'{vid}.json')))
            frames = P.sample_frames_like_training(os.path.join(a.video_root, f'{vid}.mp4'), NF, HPIX, WPIX)
            zones = build_zones(anno, cell_motion(frames, rep))
            dets = detect_all(loc, frames[0], anno['target_name'], topk=3)
            if not dets:
                print(f'  {vid}: GDINO 帧0 无检出, 只存 zones (无 patch)', flush=True)
                np.savez_compressed(os.path.join(a.out_dir, f'{vid}.npz'), zones=zones)
                continue
            x0, y0, x1, y1 = [max(0, int(v)) for v in dets[0][2]]
            x1, y1 = min(WPIX, x1), min(HPIX, y1)
            patch = frames[0][y0:y1, x0:x1].copy()
            if patch.shape[0] < 12 or patch.shape[1] < 12:
                np.savez_compressed(os.path.join(a.out_dir, f'{vid}.npz'), zones=zones)
                print(f'  {vid}: patch 过小 {patch.shape}, 只存 zones', flush=True)
                continue
            np.savez_compressed(os.path.join(a.out_dir, f'{vid}.npz'),
                                zones=zones, patch=patch, box=np.array([y0, x0, y1, x1]))
            nz = {z: int((zones == z).sum()) for z in (0, 1, 2)}
            print(f'  {vid}: patch={patch.shape[:2]} 安全格 bg={nz[0]} B={nz[1]} A={nz[2]}', flush=True)
        except Exception as e:
            print(f'  {vid} FAIL: {str(e)[:150]}', flush=True)
    print(f'[aug-prep] shard {a.shard_index}/{a.num_shards} done', flush=True)


if __name__ == '__main__':
    main()
