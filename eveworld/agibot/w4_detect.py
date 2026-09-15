#!/usr/bin/env python3
"""WMB 适配 W1 前置:对 trainset_v1 500 条跑 GDINO 检测(复用 GR1 线 t4g_detect)。

与 GR1 线差异(monkey-patch):
- VIDEO_ROOT -> trainset_v1;分辨率 768x480 -> 640x480, latent 网格 W_LAT 48 -> 40
- prompt 用原生指令(<name>.txt.orig, 名词简单利于 parse_objects/GDINO)
产出: wmb_adapt/t4g_anno/<name>.json(与 GR1 anno 同 schema, 供 wmap/aug_prep 复用)
用法: python w4_detect.py --shard-index i --num-shards n
"""
import argparse
import glob
import json
import os
import sys

TRACK4GEN = "eveworld/pipeline"
sys.path.insert(0, TRACK4GEN)

TRAIN = "/data/datasets/gagi/wmb_adapt/trainset_v1"
OUT_DEFAULT = "/data/datasets/gagi/wmb_adapt/t4g_anno"

import t4g_detect as D  # noqa: E402

# WMB 口径 patch: 640x480, latent 30x40(VAE 8x + patch 16 -> 640/16=40, 480/16=30)
D.VIDEO_ROOT = TRAIN
D.WIMG = 640
D.HIMG = 480
D.W_LAT = 40
D.H_LAT = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from t4g_gdino import GDinoLocator

    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{TRAIN}/*.mp4"))
    names = names[a.shard_index::a.num_shards]
    loc = GDinoLocator(device=a.device)
    print(f"[wmb-detect] shard {a.shard_index}/{a.num_shards}: {len(names)} videos", flush=True)
    for name in names:
        out = f"{a.out_dir}/{name}.json"
        if os.path.exists(out):
            continue
        orig = f"{TRAIN}/{name}.txt.orig"
        prompt = open(orig).read().strip() if os.path.exists(orig) else open(f"{TRAIN}/{name}.txt").read().strip()
        try:
            anno = D.process(loc, name, prompt)
            json.dump(anno, open(out, "w"), ensure_ascii=False)
            print(f"  {name}: gate={anno['gate_enabled']}({anno.get('gate_reason')}) "
                  f"det={anno.get('n_detected_frames')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} FAIL: {str(e)[:120]}", flush=True)
    print("[wmb-detect] shard done", flush=True)


if __name__ == "__main__":
    main()
