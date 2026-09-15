#!/usr/bin/env python3
"""AgiBot 双臂 Contractual Weightmap (方案 Phase 3, 核心重设计)。

针对 GR1 配方在 AgiBot 域的失败模式 (过标 → 归一化压低背景 → scene/臂漂移) 重造:
- 全自动 (光流运动 + GDINO 检测), 无手工线; 白色前臂靠运动掩码覆盖
  (dark-foreground 启发式对白臂失明, 明令禁用)
- 双臂: 左右各一条锚定走廊 (gripper 检测带 side), 检测洞用 last-center 延续
- 护 scene 硬约束: W_BG=1.0 / 倍数<=3x / 覆盖率<=0.33 (超则自适应升运动分位
  → 去膨胀 → 降臂档) / CAP=3.0

两版缓存 (值离散, 过 trainer wmap_values 校验):
  motionauto  {1.0,3.0}          运动掩码 ∪ 夹爪框 ∪ 对象格 (最稳, 先训)
  skilltiered {1.0,2.0,2.5,3.0}  2.0=运动/臂走廊 2.5=夹爪/对象/落点(TRANSFER,t<t_arr)
                                  3.0=状态区(STATE, 全时段)
输出 (24,30,40) float32 -> weightmap_cache_{motionauto,skilltiered}/<name>.npy
用法: python agi_weightmap.py [--variant both] [--viz-n 30]
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

CLEAN = '/data/datasets/gagi/agibot_ewm_clean'
PROBE = '/data/datasets/gagi/eve_v2_outputs/agibot_t4g_probe'
ANNO = f'{PROBE}/t4g_anno'
GRIP = f'{PROBE}/gripper_anno'
OUT_MOTION = f'{PROBE}/weightmap_cache_motionauto'
OUT_SKILL = f'{PROBE}/weightmap_cache_skilltiered'
VIZ = f'{PROBE}/weightmap_viz'

T_LAT, H_LAT, W_LAT = 24, 30, 40
NF, H, W = 93, 480, 640
MOTION_MIN = 0.3
COVER_CAP = 0.33
W_ARM, W_GRIP, W_OBJ, W_DEST, W_STATE, CAP = 2.0, 2.5, 2.5, 2.5, 3.0, 3.0


def grid_pool(field, gh, gw):
    return field.reshape(gh, field.shape[0] // gh, gw, field.shape[1] // gw).mean(axis=(1, 3))


def dilate(mask, r):
    if r <= 0:
        return mask
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def stamp_line(mask, a, b, thickness=3):
    m = mask.astype(np.uint8)
    cv2.line(m, (a[1], a[0]), (b[1], b[0]), 1, thickness)
    mask |= m.astype(bool)


def stamp_box(mask, cy, cx, hy, hx):
    mask[max(0, cy - hy):cy + hy + 1, max(0, cx - hx):cx + hx + 1] = True


def read_frames_gray(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


def motion_masks(frames, pct, dil):
    """逐 latent 帧运动掩码 (DIS 光流, 帧 4t->4t+1), 自动覆盖双臂(含白臂)+动目标。"""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    masks = []
    for t in range(T_LAT):
        p = min(4 * t, len(frames) - 2)
        flow = dis.calc(frames[p], frames[p + 1], None)
        g = grid_pool(np.linalg.norm(flow, axis=2), H_LAT, W_LAT)
        thr = max(np.percentile(g, pct), MOTION_MIN)
        masks.append(dilate(g >= thr, dil))
    return masks


def gripper_tracks(name):
    """每帧左右夹爪中心 (检测洞用 last-center 延续, 后段白臂走廊不掉线)。"""
    fp = f'{GRIP}/{name}.json'
    per = json.load(open(fp))['per_lat_gripper'] if os.path.exists(fp) else [[]] * T_LAT
    tracks = {'left': [None] * T_LAT, 'right': [None] * T_LAT}
    last = {}
    for t in range(T_LAT):
        for g in (per[t] if t < len(per) else []):
            side = g.get('side') or ('left' if g['center'][1] < W_LAT // 2 else 'right')
            last[side] = tuple(g['center'])
        for side, c in last.items():
            tracks[side][t] = c
    return tracks


def corridor_masks(tracks):
    """双臂走廊: 臂从画面侧边伸入 (几何不变量), 从该侧边缘同行点水平连到爪心。
    固定底部锚点在 AgiBot (臂自两侧中部入画) 上路径不符, 白色前臂悬停段会漏。"""
    per_t = []
    for t in range(T_LAT):
        m = np.zeros((H_LAT, W_LAT), bool)
        for side in ('left', 'right'):
            c = tracks[side][t]
            if c is not None:
                edge = (c[0], 0) if side == 'left' else (c[0], W_LAT - 1)
                stamp_line(m, edge, c, thickness=4)
                stamp_box(m, c[0], c[1], 3, 4)
        per_t.append(m)
    return per_t


def build_one(name, variant, pct=75, dil=1, arm_on=True):
    anno = json.load(open(f'{ANNO}/{name}.json'))
    frames = read_frames_gray(f'{CLEAN}/{name}.mp4')
    if len(frames) < NF:
        raise RuntimeError(f'frames={len(frames)}')
    mot = motion_masks(frames, pct, dil)
    tracks = gripper_tracks(name)
    corr = corridor_masks(tracks)
    per = anno.get('per_lat_frame') or [{}] * T_LAT
    cat = anno.get('category', 'TRANSFER')
    t_arr = anno.get('t_arrival') if anno.get('gate_enabled') else None
    state_cells = anno.get('state_cells') or []

    wm = np.ones((T_LAT, H_LAT, W_LAT), np.float32)
    for t in range(T_LAT):
        if variant == 'motionauto':
            m = mot[t] | corr[t]
            fr = per[min(t, len(per) - 1)]
            if fr.get('target_cell'):
                gy, gx = fr['target_cell']
                box = np.zeros_like(m)
                stamp_box(box, gy, gx, 2, 2)
                m |= box
            wm[t][m] = 3.0
            continue
        # skilltiered: 低档先铺, 高档逐层覆盖 (max 语义)
        if arm_on:
            wm[t][mot[t] | corr[t]] = W_ARM
        grip_m = np.zeros((H_LAT, W_LAT), bool)
        for side in ('left', 'right'):
            c = tracks[side][t]
            if c is not None:
                stamp_box(grip_m, c[0], c[1], 3, 4)
        wm[t][grip_m] = np.maximum(wm[t][grip_m], W_GRIP)
        fr = per[min(t, len(per) - 1)]
        if fr.get('target_cell'):
            gy, gx = fr['target_cell']
            obj_m = np.zeros((H_LAT, W_LAT), bool)
            stamp_box(obj_m, gy, gx, 3, 3)
            wm[t][obj_m] = np.maximum(wm[t][obj_m], W_OBJ)
        if cat == 'TRANSFER' and t_arr is not None and t < t_arr:
            dest_m = np.zeros((H_LAT, W_LAT), bool)
            for gy, gx in fr.get('b_cells') or []:
                if 0 <= gy < H_LAT and 0 <= gx < W_LAT:
                    dest_m[gy, gx] = True
            wm[t][dest_m] = np.maximum(wm[t][dest_m], W_DEST)
        if cat == 'STATE' and state_cells:
            st_m = np.zeros((H_LAT, W_LAT), bool)
            for gy, gx in state_cells:
                if 0 <= gy < H_LAT and 0 <= gx < W_LAT:
                    st_m[gy, gx] = True
            wm[t][dilate(st_m, 1)] = np.maximum(wm[t][dilate(st_m, 1)], W_STATE)
    np.clip(wm, 1.0, CAP, out=wm)
    return wm


def build_capped(name, variant):
    """覆盖率超 0.33 时自适应收敛: 升运动分位 -> 去膨胀 -> 弃臂档。"""
    for pct, dil, arm_on in ((75, 1, True), (82, 1, True), (88, 0, True), (88, 0, False)):
        wm = build_one(name, variant, pct, dil, arm_on)
        cov = float((wm > 1.0).mean())
        if cov <= COVER_CAP:
            return wm, cov, (pct, dil, arm_on)
    return wm, cov, (pct, dil, arm_on)


def one(job):
    name, variant, out_dir = job
    try:
        wm, cov, cfg = build_capped(name, variant)
        np.save(f'{out_dir}/{name}.npy', wm)
        return name, variant, cov, cfg, 'ok'
    except Exception as e:  # noqa: BLE001
        return name, variant, 0.0, None, f'FAIL {type(e).__name__}: {str(e)[:80]}'


def viz_sample(names, n):
    """抽样目检: 覆盖 9 skill, 画 t0/t12/t22 三帧叠加 (专项看后段白臂覆盖)。"""
    os.makedirs(VIZ, exist_ok=True)
    by_skill = {}
    for name in names:
        sk = json.load(open(f'{ANNO}/{name}.json')).get('skill', '?')
        by_skill.setdefault(sk, []).append(name)
    picked = []
    while len(picked) < n and any(by_skill.values()):
        for sk in list(by_skill):
            if by_skill[sk]:
                picked.append(by_skill[sk].pop(0))
    for name in picked[:n]:
        wp = f'{OUT_SKILL}/{name}.npy'
        if not os.path.exists(wp):
            continue
        wm = np.load(wp)
        cap = cv2.VideoCapture(f'{CLEAN}/{name}.mp4')
        fr = {}
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in (0, 48, 88):
                fr[i] = f
            i += 1
        cap.release()
        panels = []
        for px, t in ((0, 0), (48, 12), (88, 22)):
            if px not in fr:
                continue
            heat = cv2.resize(((wm[t] - 1.0) / (CAP - 1.0) * 255).astype(np.uint8),
                              (W, H), interpolation=cv2.INTER_NEAREST)
            ov = fr[px].copy()
            red = np.zeros_like(ov)
            red[:, :, 2] = heat
            ov = cv2.addWeighted(ov, 1.0, red, 0.6, 0)
            cv2.putText(ov, f't={t}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            panels.append(ov)
        sk = json.load(open(f'{ANNO}/{name}.json')).get('skill', '?')
        img = np.concatenate(panels, axis=1)
        cv2.putText(img, f'{name} [{sk}]', (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(f'{VIZ}/{name}.jpg', img)
    print(f'[viz] {min(len(picked), n)} clips -> {VIZ}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['motionauto', 'skilltiered', 'both'], default='both')
    ap.add_argument('--viz-n', type=int, default=30)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()

    names = sorted(f[:-5] for f in os.listdir(ANNO)
                   if f.endswith('.json') and not f.startswith('_'))
    print(f'[wmap] {len(names)} annos')
    variants = ['motionauto', 'skilltiered'] if a.variant == 'both' else [a.variant]
    for variant in variants:
        out_dir = OUT_MOTION if variant == 'motionauto' else OUT_SKILL
        os.makedirs(out_dir, exist_ok=True)
        jobs = [(n, variant, out_dir) for n in names
                if not os.path.exists(f'{out_dir}/{n}.npy')]
        covs, fails, fallback = [], 0, 0
        with Pool(a.workers) as p:
            for name, _v, cov, cfg, st in p.imap_unordered(one, jobs):
                if st != 'ok':
                    fails += 1
                    print(f'  {name}: {st}', flush=True)
                else:
                    covs.append(cov)
                    if cfg != (75, 1, True):
                        fallback += 1
        done = len([f for f in os.listdir(out_dir) if f.endswith('.npy')])
        if covs:
            cs = sorted(covs)
            print(f'[{variant}] built={len(covs)} total={done}/777 fail={fails} '
                  f'cap_fallback={fallback} cov min/med/max='
                  f'{cs[0]:.3f}/{cs[len(cs) // 2]:.3f}/{cs[-1]:.3f}')
        else:
            print(f'[{variant}] total={done}/777 (all cached)')
    if a.viz_n:
        viz_sample(names, a.viz_n)


if __name__ == '__main__':
    main()
