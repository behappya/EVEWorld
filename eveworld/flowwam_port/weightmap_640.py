#!/usr/bin/env python3
"""FlowWAM_WorldArena 权重图预计算（Wan 几何 31x30x40）。

论文 eq:pipeline 的静态部分: W_BG 基线 + 目标轨迹处 W_OBJ（按逐帧框半尺寸
膨胀）。贴块区域的上权重（W_PASTE=W_TRANS 量级）由训练时 transform 在
采样 paste plan 后动态叠加。轨迹空洞用最近邻填补。输出 npy 到
weightmap_cache_640/<task>__<episode>.npy，几何与常数记录于 sidecar meta。
"""
from __future__ import annotations

import json
import os

import numpy as np

ANNO_DIR = "/data/datasets/gagi/flowwam/igr/anno_640"
OUT_DIR = "/data/datasets/gagi/flowwam/igr/weightmap_cache_640"
N_LAT, H_LAT, W_LAT = 31, 30, 40
CELL_PX = 16
W_BG, W_OBJ = 0.5, 4.0
OBJ_MARGIN = 1
DEFAULT_HALF = (2, 2)


def halfsize_from_box(box) -> tuple[int, int]:
    if not box:
        return DEFAULT_HALF
    x0, y0, x1, y1 = box
    hh = max(1, int(round((y1 - y0) / 2 / CELL_PX)))
    hw = max(1, int(round((x1 - x0) / 2 / CELL_PX)))
    return (hh, hw)


def fill_traj(per_lat) -> list:
    """target_cell 缺测用最近已测帧填补。"""
    cells = [e.get("target_cell") for e in per_lat]
    known = [t for t, c in enumerate(cells) if c is not None]
    if not known:
        return cells
    out = []
    for t, c in enumerate(cells):
        if c is not None:
            out.append(c)
        else:
            nearest = min(known, key=lambda k: abs(k - t))
            out.append(cells[nearest])
    return out


def stamp(w, t, cell, half, val, margin) -> None:
    gy, gx = cell
    hh, hw = half[0] + margin, half[1] + margin
    y0, y1 = max(0, gy - hh), min(H_LAT, gy + hh + 1)
    x0, x1 = max(0, gx - hw), min(W_LAT, gx + hw + 1)
    w[t, y0:y1, x0:x1] = np.maximum(w[t, y0:y1, x0:x1], val)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n, means = 0, []
    for f in sorted(os.listdir(ANNO_DIR)):
        if not f.endswith(".json"):
            continue
        anno = json.load(open(os.path.join(ANNO_DIR, f)))
        assert anno["n_lat"] == N_LAT and anno["H_lat"] == H_LAT and anno["W_lat"] == W_LAT, f
        per_lat = anno["per_lat_frame"]
        traj = fill_traj(per_lat)
        w = np.full((N_LAT, H_LAT, W_LAT), W_BG, np.float32)
        for t in range(N_LAT):
            if traj[t] is None:
                continue
            half = halfsize_from_box(per_lat[t].get("target_box_px"))
            stamp(w, t, traj[t], half, W_OBJ, OBJ_MARGIN)
        np.save(os.path.join(OUT_DIR, f[:-5] + ".npy"), w)
        means.append(float(w.mean()))
        n += 1
    meta = {
        "n": n, "shape": [N_LAT, H_LAT, W_LAT], "cell_px": CELL_PX,
        "W_BG": W_BG, "W_OBJ": W_OBJ, "obj_margin": OBJ_MARGIN,
        "paste_weight_note": "贴块区域 W_PASTE=6.0 由训练 transform 动态叠加(对齐 giga W_TRANS 量级)",
        "normalize_note": "归一化(均值=1)在 trainer 侧做, 与 giga 口径一致",
        "mean_raw_median": float(np.median(means)) if means else None,
    }
    json.dump(meta, open(os.path.join(OUT_DIR, "_meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"预计算 {n} 条 -> {OUT_DIR}")
    print(f"均值(归一化前): 中位 {np.median(means):.3f} 范围 [{min(means):.3f},{max(means):.3f}]")


if __name__ == "__main__":
    main()
