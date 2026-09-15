#!/usr/bin/env python3
"""IGR 窗口级贴块（论文 eq:paste）for FlowWAM 短窗训练。

给定像素帧窗口（F 帧, 对应 latent 帧 [t0, t0+T_lat)）与 episode 的
anno/静态权重图：
  1. 从窗口首帧按 target_box_px 裁目标 patch p；
  2. 安全区 Ω = 窗口内全程 (静态权重==W_BG) 且非机械臂格；
  3. 采样 paste plan（latent 子窗、位置、尺度 0.8-1.2、alpha 融合）；
  4. 像素域贴入 -> 污染窗口 x̃；权重窗口在贴块区叠加 W_PASTE。
返回 (corrupted_frames, weight_window, plan)；无可行 plan 时返回原样
（clean 样本, plan=None）——与 giga t4g_aug 的回退行为一致。
"""
from __future__ import annotations

import numpy as np

CELL_PX = 16
GH, GW = 30, 40
W_BG, W_PASTE = 0.5, 6.0
PASTE_MARGIN = 1


def lat_to_frame_local(t_local: int, t0: int) -> int:
    """窗口内 latent 帧 -> 窗口内像素帧下标（窗口从 latent t0 的像素帧起）。"""
    g0 = 0 if t0 == 0 else 4 * t0 - 1
    g = 0 if (t0 + t_local) == 0 else 4 * (t0 + t_local) - 1
    return g - g0


def crop_target_patch(frames: np.ndarray, anno: dict, t0: int) -> np.ndarray | None:
    """从窗口首个有框的 latent 帧裁目标 patch (RGB uint8)。"""
    per = anno["per_lat_frame"]
    T_lat_win = min(len(per) - t0, (frames.shape[0] + 3) // 4 + 1)
    for tl in range(T_lat_win):
        box = per[t0 + tl].get("target_box_px")
        if not box:
            continue
        f_idx = min(lat_to_frame_local(tl, t0), frames.shape[0] - 1)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frames.shape[2], x1), min(frames.shape[1], y1)
        if x1 - x0 >= 12 and y1 - y0 >= 12:
            return frames[f_idx, y0:y1, x0:x1].copy()
    return None


def safe_zone_2d(weight_win: np.ndarray, arm_cells: list) -> np.ndarray:
    """(T,GH,GW) 权重窗口 -> 2D 安全掩码：全窗口 W_BG 且非机械臂格。"""
    safe = (weight_win <= W_BG + 1e-6).all(axis=0)
    for gy, gx in arm_cells:
        if 0 <= gy < GH and 0 <= gx < GW:
            safe[gy, gx] = False
    return safe


def sample_plan(
    safe: np.ndarray,
    patch_hw: tuple[int, int],
    t_lat_win: int,
    rng: np.random.Generator,
    k_retry: int = 60,
    min_dur: int = 2,
) -> dict | None:
    hp0, wp0 = patch_hw
    # RoboTwin 域目标框可达 ~300px(9x19格), 而安全区多为 ~150 个散落格,
    # 大矩形放不下 -> 复制品统一压到 MAX_PASTE_PX 内(保持纵横比)再抖动
    MAX_PASTE_PX = 96
    base = min(1.0, MAX_PASTE_PX / max(hp0, wp0, 1))
    for _ in range(k_retry):
        if t_lat_win <= min_dur:
            return None
        t_lo = int(rng.integers(1, t_lat_win - min_dur + 1))
        t_hi = t_lat_win if rng.random() < 0.5 else int(rng.integers(t_lo + min_dur, t_lat_win + 1))
        scale = base * float(rng.uniform(0.8, 1.2))
        hp, wp = max(12, int(hp0 * scale)), max(12, int(wp0 * scale))
        if hp > GH * CELL_PX // 2 or wp > GW * CELL_PX // 2:
            continue
        gh_need = (hp + CELL_PX - 1) // CELL_PX
        gw_need = (wp + CELL_PX - 1) // CELL_PX
        ys, xs = np.where(safe)
        if len(ys) == 0:
            return None
        k = int(rng.integers(len(ys)))
        gy, gx = int(ys[k]), int(xs[k])
        if gy + gh_need > GH or gx + gw_need > GW:
            continue
        if not safe[gy:gy + gh_need, gx:gx + gw_need].all():
            continue
        return {
            "t_lo": t_lo, "t_hi": t_hi,
            "y0": gy * CELL_PX, "x0": gx * CELL_PX,
            "hp": hp, "wp": wp, "scale": scale,
            "alpha": float(rng.uniform(0.85, 1.0)),
        }
    return None


def apply_paste(
    frames: np.ndarray,
    weight_win: np.ndarray,
    anno: dict,
    t0: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """frames (F,H,W,3) uint8（窗口）; weight_win (T_lat_win,GH,GW) float32。"""
    import cv2

    t_lat_win = weight_win.shape[0]
    patch = crop_target_patch(frames, anno, t0)
    if patch is None:
        return frames, weight_win, None
    plan = sample_plan(safe_zone_2d(weight_win, anno.get("arm_cells", [])),
                       patch.shape[:2], t_lat_win, rng)
    if plan is None:
        return frames, weight_win, None

    hp, wp = plan["hp"], plan["wp"]
    patch_r = cv2.resize(patch, (wp, hp), interpolation=cv2.INTER_AREA)
    # 羽化边缘 alpha
    mask = np.ones((hp, wp), np.float32) * plan["alpha"]
    feather = max(2, min(hp, wp) // 8)
    mask[:feather] *= np.linspace(0.2, 1, feather)[:, None]
    mask[-feather:] *= np.linspace(1, 0.2, feather)[:, None]
    mask[:, :feather] *= np.linspace(0.2, 1, feather)[None, :]
    mask[:, -feather:] *= np.linspace(1, 0.2, feather)[None, :]
    mask = mask[..., None]

    out = frames.copy()
    y0, x0 = plan["y0"], plan["x0"]
    f_lo = lat_to_frame_local(plan["t_lo"], t0)
    f_hi = frames.shape[0] if plan["t_hi"] >= t_lat_win else lat_to_frame_local(plan["t_hi"], t0)
    for f in range(f_lo, f_hi):
        roi = out[f, y0:y0 + hp, x0:x0 + wp].astype(np.float32)
        out[f, y0:y0 + hp, x0:x0 + wp] = (
            roi * (1 - mask) + patch_r.astype(np.float32) * mask
        ).astype(np.uint8)

    w = weight_win.copy()
    gy0 = max(0, y0 // CELL_PX - PASTE_MARGIN)
    gx0 = max(0, x0 // CELL_PX - PASTE_MARGIN)
    gy1 = min(GH, (y0 + hp + CELL_PX - 1) // CELL_PX + PASTE_MARGIN)
    gx1 = min(GW, (x0 + wp + CELL_PX - 1) // CELL_PX + PASTE_MARGIN)
    w[plan["t_lo"]:plan["t_hi"], gy0:gy1, gx0:gx1] = np.maximum(
        w[plan["t_lo"]:plan["t_hi"], gy0:gy1, gx0:gx1], W_PASTE)
    return out, w, plan
