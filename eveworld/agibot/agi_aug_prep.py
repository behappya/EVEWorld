#!/usr/bin/env python3
"""AgiBot Copy-Paste 增广资产 (方案 Phase 4.1, 仿 w6 monkey-patch 线)。

与 GR1 线差异:
- 先 patch t4g_ghost_probe 网格/路径 (640x480 -> W_LAT 40) 再 import t4g_aug_prep
- build_zones 扩展: 双臂夹爪走廊 (±2 帧, dilate 2) 挖成不安全区 -1
  → 副本绝不贴到手臂上/旁 (贴臂正是加剧后段臂畸变的元凶)
产出: agibot_t4g_probe/aug_assets/<name>.npz (patch/box/zones)
用法: python agi_aug_prep.py [--shard-index i --num-shards n]  (GDINO 需 giga_world1)
"""
import json
import os
import sys

TRACK4GEN = 'eveworld/pipeline'
sys.path.insert(0, TRACK4GEN)

CLEAN = '/data/datasets/gagi/agibot_ewm_clean'
PROBE = '/data/datasets/gagi/eve_v2_outputs/agibot_t4g_probe'
ANNO = f'{PROBE}/t4g_anno'
GRIP = f'{PROBE}/gripper_anno'
OUT = f'{PROBE}/aug_assets'

os.environ.setdefault('T4G_W_LAT', '40')
os.environ.setdefault('T4G_WPIX', '640')

import numpy as np  # noqa: E402
import t4g_ghost_probe as G  # noqa: E402

G.W_LAT = 40
G.WPIX = 640
G.VIDEO_ROOT = CLEAN
G.ANNO_DIR = ANNO

import t4g_aug_prep as AP  # noqa: E402  (from-import 此时读 patched 值)

T_LAT, H_LAT, W_LAT = 24, 30, 40
GRIP_TFRAMES, GRIP_R = 2, 2

_orig_build_zones = AP.build_zones


def agi_build_zones(anno, motion):
    """原 zones 逻辑 + 双臂夹爪走廊排除 (检测洞 last-center 延续)。"""
    zones = _orig_build_zones(anno, motion)
    fp = f'{GRIP}/{anno["vid"]}.json'
    if not os.path.exists(fp):
        return zones
    per = json.load(open(fp))['per_lat_gripper']
    centers_at = [[] for _ in range(T_LAT)]
    last = {}
    for t in range(T_LAT):
        for g in (per[t] if t < len(per) else []):
            side = g.get('side') or ('left' if g['center'][1] < W_LAT // 2 else 'right')
            last[side] = tuple(g['center'])
        centers_at[t] = list(last.values())
    for t in range(T_LAT):
        cells = set()
        for t2 in range(max(0, t - GRIP_TFRAMES), min(T_LAT, t + GRIP_TFRAMES + 1)):
            cells.update(centers_at[t2])
        for gy, gx in cells:
            zones[t, max(0, gy - GRIP_R):gy + GRIP_R + 1,
                  max(0, gx - GRIP_R):gx + GRIP_R + 1] = -1
    return zones


AP.build_zones = agi_build_zones


def main():
    names = sorted(f[:-5] for f in os.listdir(ANNO)
                   if f.endswith('.json') and not f.startswith('_'))
    names = [n for n in names if not os.path.exists(f'{OUT}/{n}.npz')]
    if not names:
        print('[agi-aug-prep] 全部资产已存在')
        return
    argv = ['agi_aug_prep',
            '--out-dir', OUT,
            '--anno-dir', ANNO,
            '--video-root', CLEAN,
            '--vids', *names]
    for k in ('--shard-index', '--num-shards', '--device'):
        if k in sys.argv:
            i = sys.argv.index(k)
            argv += [k, sys.argv[i + 1]]
    sys.argv = argv
    AP.main()


if __name__ == '__main__':
    main()
