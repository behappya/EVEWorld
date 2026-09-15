#!/usr/bin/env python3
"""权重图人眼校准: 左原图帧 | 右叠权重区 (颜色=区域, 亮度=权重)。
颜色: 物体轨迹=绿, 抓取/放置窗=红, 该空区=蓝, 背景=不染。圆点=每帧 target_cell。
每条视频出一张多帧网格 PNG。用法: python t4g_weightmap_viz.py --vids 2,21,12 --out-dir <dir>
"""
import argparse
import os

import cv2
import numpy as np

import t4g_probe as P
from t4g_weightmap import build_weightmap, load_anno, T_LAT, H_LAT, W_LAT, W_OBJ, W_TRANS, W_EMPTY, W_BG

NF, HPIX, WPIX, CELL = 93, 480, 768, 16
VIDEO_ROOT = '/data/datasets/gagi/gr1_finetune_data/raw_data'


def classify_color(anno, wmapinfo, t, gy, gx, w):
    """按权重值反推区域色 (与 build 的 max 语义一致, 取最高优先区)。"""
    # 颜色以 RGB 指定 (渲染最后 RGB2BGR)。红=6x抓放 绿=4x物体 青=3x高危该空/空爪 蓝=2x原位该空
    from t4g_weightmap import W_BDEST, W_GRIP
    if abs(w - W_TRANS) < 1e-3:
        return (235, 55, 55)      # 红: 6x 转换窗(抓取/放置)
    if abs(w - W_OBJ) < 1e-3:
        return (55, 220, 55)      # 绿: 4x 物体轨迹
    if abs(w - W_BDEST) < 1e-3 or abs(w - W_GRIP) < 1e-3:
        return (60, 210, 210)     # 青: 3x B目的地/空爪 (高危)
    if abs(w - W_EMPTY) < 1e-3:
        return (55, 150, 235)     # 蓝: 2x 原位该空
    return None


def render(vid, out_dir, allframes=False):
    anno = load_anno(vid)
    w, seg = build_weightmap(anno)
    traj = seg.get('traj') or [None] * T_LAT
    frames = P.sample_frames_like_training(f'{VIDEO_ROOT}/{vid}.mp4', NF, HPIX, WPIX)
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    if allframes:                            # 全部 24 latent 帧
        ts = list(range(T_LAT))
    else:
        ts = [t for t in [1, seg['t_grasp'], (seg['t_grasp'] or 4) + 2,
                          seg['t_arrival'], seg['t_release'], T_LAT - 1] if t is not None]
        ts = sorted(set(int(np.clip(t, 0, T_LAT - 1)) for t in ts))
        while len(ts) < 6:                   # 补足 6 帧均匀
            extra = [t for t in np.linspace(1, T_LAT - 1, 6).astype(int) if t not in ts]
            if not extra:
                break
            ts = sorted(set(ts + [extra[0]]))
        ts = ts[:6]
    status = 'RELIABLE' if seg['traj_reliable'] else 'DEGRADED(empty-only)'
    rows = []
    for t in ts:
        p = int(rep[t])
        left = np.ascontiguousarray(frames[p].copy())
        right = frames[p].copy().astype(np.float32)
        overlay = np.zeros_like(right)
        for gy in range(H_LAT):
            for gx in range(W_LAT):
                col = classify_color(anno, seg, t, gy, gx, w[t, gy, gx])
                if col:
                    overlay[gy * CELL:(gy + 1) * CELL, gx * CELL:(gx + 1) * CELL] = col
        m = (overlay.sum(2) > 0)[..., None]
        right = np.where(m, 0.48 * right + 0.52 * overlay, right).astype(np.uint8)
        # 细网格线, 方便数格子
        for gy in range(0, H_LAT + 1, 2):
            cv2.line(right, (0, gy * CELL), (WPIX, gy * CELL), (90, 90, 90), 1)
        for gx in range(0, W_LAT + 1, 3):
            cv2.line(right, (gx * CELL, 0), (gx * CELL, HPIX), (90, 90, 90), 1)
        tag = f't={t}'
        for nm, tt in [('grasp', seg['t_grasp']), ('arr', seg['t_arrival']), ('rel', seg['t_release'])]:
            if tt == t:
                tag += f' [{nm}]'
        cv2.putText(left, f'{vid} {tag} {status}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(right, 'R=6x(grasp/place) G=4x(obj) Cyan=3x(Bdest/gripper) B=2x(origin) bg=0.5x',
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        rows.append(np.concatenate([left, right], axis=1))
    suffix = '_allframes' if allframes else ''
    out = os.path.join(out_dir, f'wmap_{vid}{suffix}.png')
    cv2.imwrite(out, cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    print(f'  {vid}: reliable={seg["traj_reliable"]} t_grasp={seg["t_grasp"]} t_arr={seg["t_arrival"]} '
          f't_release={seg["t_release"]} jumps={seg["jumps"]} '
          f'obj%={float((w==W_OBJ).mean()):.3f} trans%={float((w==W_TRANS).mean()):.3f} -> {out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vids', default='2,21,12,6,15,89')
    ap.add_argument('--all', action='store_true', help='渲染 t4g_anno 全部视频')
    ap.add_argument('--allframes-vids', default='', help='指定视频渲染全部24帧(逗号分隔)')
    ap.add_argument('--out-dir', default='/data/datasets/gagi/eve_v2_outputs/track4gen_probe/weightmap_viz')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from t4g_weightmap import ANNO_DIR
    if a.allframes_vids:
        for vid in a.allframes_vids.split(','):
            render(vid.strip(), a.out_dir, allframes=True)
        return
    if a.all:
        vids = sorted((f[:-5] for f in os.listdir(ANNO_DIR) if f[0].isdigit()), key=int)
    else:
        vids = [v.strip() for v in a.vids.split(',')]
    for vid in vids:
        try:
            render(vid, a.out_dir)
        except Exception as e:
            print(f'  {vid} FAIL: {str(e)[:100]}', flush=True)


if __name__ == '__main__':
    main()
