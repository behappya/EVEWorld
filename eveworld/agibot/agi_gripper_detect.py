#!/usr/bin/env python3
"""AgiBot 777 条双臂夹爪检测 (方案 Phase 2.3): t4g_gripper_detect + left/right side。

patch 要点: t4g_detect 常量 (px_to_cell/box_to_cells 在其命名空间) 与
t4g_gripper_detect 常量 (面积过滤) 都要改到 640x480/30x40。
side 按夹爪中心格 gx < W_LAT/2 分 left/right (双臂 weightmap 走廊所需)。
产出: agibot_t4g_probe/gripper_anno/<name>.json
用法: python agi_gripper_detect.py --shard-index i --num-shards n  (giga_world1 env)
"""
import argparse
import json
import os
import sys

TRACK4GEN = 'eveworld/pipeline'
sys.path.insert(0, TRACK4GEN)

CLEAN = '/data/datasets/gagi/agibot_ewm_clean'
OUT_DEFAULT = '/data/datasets/gagi/eve_v2_outputs/agibot_t4g_probe/gripper_anno'

import t4g_detect as D  # noqa: E402
import t4g_gripper_detect as G  # noqa: E402
import numpy as np  # noqa: E402

for mod in (D, G):
    mod.WIMG = 640
    mod.HIMG = 480
D.W_LAT = 40
D.H_LAT = 30
G.VIDEO_ROOT = CLEAN
W_LAT = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from t4g_gdino import GDinoLocator

    names = sorted(f[:-4] for f in os.listdir(CLEAN)
                   if f.endswith('.mp4') and not f.endswith('_trans.mp4'))
    names = names[a.shard_index::a.num_shards]
    if a.limit:
        names = names[:a.limit]
    loc = GDinoLocator(device=a.device)
    print(f'[agi-gripper] shard {a.shard_index}/{a.num_shards}: {len(names)} clips', flush=True)
    for name in names:
        out = f'{a.out_dir}/{name}.json'
        if os.path.exists(out):
            continue
        try:
            anno = G.process(loc, name)
            for per in anno['per_lat_gripper']:
                for g in per:
                    g['side'] = 'left' if g['center'][1] < W_LAT // 2 else 'right'
            json.dump(anno, open(out, 'w'))
            avg = np.mean([len(p) for p in anno['per_lat_gripper']])
            print(f'  {name}: det={anno["n_det_frames"]}/24 avg={avg:.1f}', flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'  {name} FAIL: {str(e)[:120]}', flush=True)
    print('[agi-gripper] shard done', flush=True)


if __name__ == '__main__':
    main()
