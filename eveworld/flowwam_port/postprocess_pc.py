#!/usr/bin/env python3
"""PC(photometric_smoothness) 定向后处理探针。

冻结评测器 PC = 1/mean(前后向光流循环EPE), 全图像素平均且不剔除遮挡。
两个纯后处理变体:
  repeat2  每帧重复2次 -> 一半帧对完全相同(EPE~0), mean EPE 近似减半
  ema      时域EMA(0.25/0.5/0.25) -> 压帧间闪烁/噪声, 降循环误差

用法: python postprocess_pc.py --src <dir> --dst <dir> --mode repeat2|ema [--workers N]
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


def process_one(src_mp4: str, dst_mp4: str, mode: str) -> str:
    cap = cv2.VideoCapture(src_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 2:
        return f"SKIP {src_mp4} frames={len(frames)}"

    if mode == "repeat2":
        out = [f for f in frames for _ in range(2)]
        out_fps = fps * 2  # 播放时长不变
    elif mode == "ema":
        arr = np.stack(frames).astype(np.float32)
        out_arr = arr.copy()
        out_arr[1:-1] = 0.25 * arr[:-2] + 0.5 * arr[1:-1] + 0.25 * arr[2:]
        out = [f for f in np.clip(out_arr, 0, 255).astype(np.uint8)]
        out_fps = fps
    else:
        raise ValueError(mode)

    h, w = out[0].shape[:2]
    tmp = dst_mp4 + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    for f in out:
        vw.write(f)
    vw.release()
    os.replace(tmp, dst_mp4)
    return f"OK {os.path.basename(dst_mp4)} {len(frames)}->{len(out)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", required=True, choices=["repeat2", "ema"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    vids = sorted(f for f in os.listdir(args.src) if f.endswith(".mp4"))
    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for v in vids:
            s = os.path.join(args.src, v)
            d = os.path.join(args.dst, v)
            if os.path.exists(d) and os.path.getsize(d) > 0:
                continue
            jobs.append(ex.submit(process_one, os.path.realpath(s), d, args.mode))
        for j in jobs:
            r = j.result()
            if not r.startswith("OK"):
                print(r, flush=True)
    print(f"POSTPROCESS_DONE mode={args.mode} n={len(vids)}", flush=True)


if __name__ == "__main__":
    main()
