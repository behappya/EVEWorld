#!/usr/bin/env python3
"""把 GW-0 EWMBench 生成的 side-by-side mp4 转成 EWMBench 官方评测布局。

输入:  <gen_root>/<model>/seed{42,43,44}/<task>_<episode>.mp4  (1296x480, 左输入|右生成)
输出:  <gen_root>/eval_layout/<model>_dataset/<task>/<episode>/<1|2|3>/video/frame_%05d.jpg
       (裁右半 640x480, seed42->1 seed43->2 seed44->3, 与官方 generated_samples 树对齐)

用法:
  python ewmbench_layout_convert.py --models pretrain t4g_wmapA_pre_seed42_s250 ...
"""
import argparse
import glob
import os

import cv2

GEN_ROOT = "/data/datasets/gagi/eve_v2_outputs/ewmbench_gen"
SEED_TO_SAMPLE = {"seed42": "1", "seed43": "2", "seed44": "3"}
GEN_W = 640  # 生成画面宽(右半)


def convert_one(mp4_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gen = frame[:, -GEN_W:]
        cv2.imwrite(os.path.join(out_dir, f"frame_{n:05d}.jpg"),
                    gen, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        n += 1
    cap.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default=GEN_ROOT)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--expect-frames", type=int, default=93)
    args = ap.parse_args()

    problems = []
    for model in args.models:
        for seed_dir, sample_id in SEED_TO_SAMPLE.items():
            mp4s = sorted(glob.glob(f"{args.gen_root}/{model}/{seed_dir}/*.mp4"))
            if not mp4s:
                problems.append(f"{model}/{seed_dir}: 无 mp4")
                continue
            for mp4 in mp4s:
                rid = os.path.splitext(os.path.basename(mp4))[0]  # <task>_<episode>
                task, episode = rid.split("_", 1)
                out_dir = (f"{args.gen_root}/eval_layout/{model}_dataset/"
                           f"{task}/{episode}/{sample_id}/video")
                if len(glob.glob(f"{out_dir}/frame_*.jpg")) == args.expect_frames:
                    continue  # 已转过
                n = convert_one(mp4, out_dir)
                if n != args.expect_frames:
                    problems.append(f"{rid} {seed_dir}: {n} 帧 (期望 {args.expect_frames})")
                print(f"[ok] {model}/{seed_dir}/{rid} -> {n} 帧", flush=True)

    if problems:
        print("\n[WARN] 以下条目异常:")
        for p in problems:
            print("  ", p)
        return 1
    print("\n全部转换完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
