#!/usr/bin/env python3
"""WMB 适配 W1:自动生成合同权重图 cache(替代 GR1 线手工 armfix)。

每条视频 -> (24,30,40) float32, 值 {1.0, 3.0}:
  3.0 = 运动格(DIS 光流格均值幅度 > 分位阈值; 覆盖机械臂+被操作物) ∪ GDINO 对象格(target/b/inventory, 膨胀1格)
  1.0 = 背景
时间对齐: 93 像素帧 -> 24 latent 帧(帧 4t 采样, 与 VAE 时间 4x 一致)。
输出: wmb_adapt/wmap_cache_v1/<name>.npy + 可视化 wmap_viz_v1/(前 20 条)
"""
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

TRAIN = "/data/datasets/gagi/wmb_adapt/trainset_v1"
ANNO = "/data/datasets/gagi/wmb_adapt/t4g_anno"
OUT = "/data/datasets/gagi/wmb_adapt/wmap_cache_v1"
VIZ = "/data/datasets/gagi/wmb_adapt/wmap_viz_v1"
T_LAT, H_LAT, W_LAT = 24, 30, 40
NF, H, W = 93, 480, 640
MOTION_PCT = 75          # 每帧网格运动量的分位阈值
MOTION_MIN = 0.3         # 像素/帧下限, 防静止帧全图入选
DILATE = 1


def grid_pool(field, gh, gw):
    return field.reshape(gh, field.shape[0] // gh, gw, field.shape[1] // gw).mean(axis=(1, 3))


def dilate(mask, r):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def one(name):
    try:
        cap = cv2.VideoCapture(f"{TRAIN}/{name}.mp4")
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
        if len(frames) < NF:
            return name, f"frames={len(frames)}"
        # latent 帧 t 对应像素帧 4t(t=0..23 -> 0..92)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        wm = np.ones((T_LAT, H_LAT, W_LAT), np.float32)
        anno = None
        ap = f"{ANNO}/{name}.json"
        if os.path.exists(ap):
            anno = json.load(open(ap))
        for t in range(T_LAT):
            p = min(4 * t, NF - 2)
            flow = dis.calc(frames[p], frames[p + 1], None)
            mag = np.linalg.norm(flow, axis=2)
            g = grid_pool(mag, H_LAT, W_LAT)
            thr = max(np.percentile(g, MOTION_PCT), MOTION_MIN)
            mask = g >= thr
            if anno:
                fr = (anno.get("per_lat_frame") or [{}] * T_LAT)[min(t, T_LAT - 1)]
                cells = []
                if fr.get("target_cell"):
                    cells.append(fr["target_cell"])
                cells += fr.get("b_cells") or []
                if t == 0:
                    cells += anno.get("inventory_cells") or []
                for gy, gx in cells:
                    if 0 <= gy < H_LAT and 0 <= gx < W_LAT:
                        mask[gy, gx] = True
            mask = dilate(mask, DILATE)
            wm[t][mask] = 3.0
        np.save(f"{OUT}/{name}.npy", wm)
        return name, "ok"
    except Exception as e:  # noqa: BLE001
        return name, f"FAIL {type(e).__name__}: {e}"


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(VIZ, exist_ok=True)
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{TRAIN}/*.mp4"))
    ok = 0
    with Pool(12) as p:
        for n, st in p.imap_unordered(one, names):
            if st == "ok":
                ok += 1
            else:
                print(f"  {n}: {st}", flush=True)
    print(f"wmap {ok}/{len(names)}")
    # 可视化前 20 条: 首帧 + t0/t12 权重叠加
    for n in names[:20]:
        wp = f"{OUT}/{n}.npy"
        if not os.path.exists(wp):
            continue
        wm = np.load(wp)
        cap = cv2.VideoCapture(f"{TRAIN}/{n}.mp4")
        _, f0 = cap.read()
        for _ in range(47):
            cap.read()
        _, f48 = cap.read()
        cap.release()
        panels = []
        for f, t in ((f0, 0), (f48, 12)):
            m = cv2.resize((wm[t] == 3.0).astype(np.uint8) * 255, (W, H),
                           interpolation=cv2.INTER_NEAREST)
            ov = f.copy()
            ov[m > 0] = (0.5 * ov[m > 0] + np.array([0, 0, 127])).astype(np.uint8)
            panels.append(ov)
        cv2.imwrite(f"{VIZ}/{n}.jpg", np.concatenate(panels, axis=1))
    # 覆盖率统计
    cov = []
    for n in names:
        wp = f"{OUT}/{n}.npy"
        if os.path.exists(wp):
            wm = np.load(wp)
            cov.append(float((wm == 3.0).mean()))
    print(f"覆盖率: min={min(cov):.3f} median={sorted(cov)[len(cov)//2]:.3f} max={max(cov):.3f}")


if __name__ == "__main__":
    main()
