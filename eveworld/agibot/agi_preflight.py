#!/usr/bin/env python3
"""AgiBot 配方 preflight (方案 Phase 6.3): 训练提交前工件全量校验。

校验项:
- clean 目录 777 条且无 *_trans.mp4 (最高风险: _trans 混入打包会翻倍错位)
- packed data_size == 777
- idx2vid 键 0..776 连续, 值 == sorted(clean stems)
- t4g_anno / aug_assets / weightmap / gripper_anno id 集合 == 777 stems
- anno per_lat_frame 长 24; zones (24,30,40); wmap (24,30,40) 且值 ⊆ 档位集
- T4G_W_LAT/T4G_WPIX env 下 t4g_aug_paste 网格 == (24,30,40)
用法: python agi_preflight.py [--wmap motionauto|skilltiered] [--skip-gripper]
"""
import argparse
import json
import os
import random
import sys

import numpy as np

GAGI = '/data/datasets/gagi'
CLEAN = f'{GAGI}/agibot_ewm_clean'
PACKED = f'{GAGI}/agibot_ewm_packed'
PROBE = f'{GAGI}/eve_v2_outputs/agibot_t4g_probe'
ANNO = f'{PROBE}/t4g_anno'
ASSETS = f'{PROBE}/aug_assets'
GRIP = f'{PROBE}/gripper_anno'
WMAP_VALUES = {'motionauto': {1.0, 3.0},
               'skilltiered': {1.0, 2.0, 2.5, 3.0}}
T_LAT, H_LAT, W_LAT = 24, 30, 40

fails = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f': {detail}' if detail else ''))
    if not ok:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wmap', choices=['motionauto', 'skilltiered'], default='motionauto')
    ap.add_argument('--expected', type=int, default=777)
    ap.add_argument('--sample', type=int, default=60)
    ap.add_argument('--skip-gripper', action='store_true')
    a = ap.parse_args()
    wmap_dir = f'{PROBE}/weightmap_cache_{a.wmap}'

    stems = sorted(f[:-4] for f in os.listdir(CLEAN)
                   if f.endswith('.mp4') and not f.endswith('_trans.mp4'))
    trans = [f for f in os.listdir(CLEAN) if f.endswith('_trans.mp4')]
    check('clean 数量', len(stems) == a.expected, f'{len(stems)}/{a.expected}')
    check('clean 无 _trans', not trans, f'{len(trans)} 个 _trans')
    txt = [s for s in stems if os.path.exists(f'{CLEAN}/{s}.txt')]
    check('txt 齐全', len(txt) == len(stems), f'{len(txt)}/{len(stems)}')

    packed_n = len([f for f in os.listdir(f'{PACKED}/videos/data') if f.endswith('.mp4')])
    check('packed data_size', packed_n == a.expected, f'{packed_n}')

    i2v = json.load(open(f'{ANNO}/_packidx2vid.json'))
    keys_ok = sorted(int(k) for k in i2v) == list(range(a.expected))
    vals_ok = [i2v[str(i)] for i in range(a.expected)] == stems
    check('idx2vid 键连续', keys_ok)
    check('idx2vid 值=sorted stems', vals_ok)

    stem_set = set(stems)
    for label, d, ext in (('anno', ANNO, '.json'), ('assets', ASSETS, '.npz'),
                          ('wmap', wmap_dir, '.npy')):
        ids = {f[:-len(ext)] for f in os.listdir(d)
               if f.endswith(ext) and not f.startswith('_')} if os.path.isdir(d) else set()
        check(f'{label} id 集合', ids == stem_set,
              f'{len(ids)}/{a.expected} 缺={len(stem_set - ids)} 多={len(ids - stem_set)}')
    if not a.skip_gripper:
        gids = {f[:-5] for f in os.listdir(GRIP) if f.endswith('.json')} if os.path.isdir(GRIP) else set()
        check('gripper id 集合', gids == stem_set, f'{len(gids)}/{a.expected}')

    rng = random.Random(0)
    sample = rng.sample(stems, min(a.sample, len(stems)))
    shape_ok, zone_ok, wmap_ok, val_ok, cov = True, True, True, True, []
    allowed = WMAP_VALUES[a.wmap]
    for s in sample:
        anno = json.load(open(f'{ANNO}/{s}.json'))
        if len(anno.get('per_lat_frame', [])) != T_LAT or anno.get('W_lat') != W_LAT:
            shape_ok = False
        z = np.load(f'{ASSETS}/{s}.npz')['zones']
        if z.shape != (T_LAT, H_LAT, W_LAT):
            zone_ok = False
        wm = np.load(f'{wmap_dir}/{s}.npy')
        if wm.shape != (T_LAT, H_LAT, W_LAT):
            wmap_ok = False
        if not {float(x) for x in np.unique(wm)}.issubset(allowed):
            val_ok = False
        cov.append(float((wm > 1.0).mean()))
    check('anno 网格 (24 帧, W_lat=40)', shape_ok)
    check('zones shape (24,30,40)', zone_ok)
    check(f'wmap shape (24,30,40) [{a.wmap}]', wmap_ok)
    check(f'wmap 值 ⊆ {sorted(allowed)}', val_ok)
    cs = sorted(cov)
    cov_ok = cs[-1] <= 0.40
    check('wmap 覆盖率 <= 0.40', cov_ok,
          f'min/med/max={cs[0]:.3f}/{cs[len(cs) // 2]:.3f}/{cs[-1]:.3f}')

    os.environ['T4G_W_LAT'] = '40'
    os.environ['T4G_WPIX'] = '640'
    sys.path.insert(0, 'eveworld/pipeline')
    import t4g_aug_paste as ap_mod
    check('t4g_aug_paste 网格 (env)', (ap_mod.T_LAT, ap_mod.H_LAT, ap_mod.W_LAT) == (24, 30, 40),
          f'{(ap_mod.T_LAT, ap_mod.H_LAT, ap_mod.W_LAT)}')

    if fails:
        raise SystemExit(f'PREFLIGHT FAIL: {fails}')
    print('PREFLIGHT PASS')


if __name__ == '__main__':
    main()
