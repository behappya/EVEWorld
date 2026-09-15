#!/usr/bin/env python3
"""WMB 适配 A1:生成 Copy-Paste 增广资产(wrapper 复用 GR1 线 t4g_aug_prep)。

先 patch t4g_ghost_probe 的网格/路径常量(640x480 -> W_LAT 40), 再 import t4g_aug_prep
(其 from-import 在此后求值, patch 生效)。vids 显式传 trainset 全名单。
用法: python w6_aug_prep.py [--shard-index i --num-shards n]
产出: wmb_adapt/aug_assets_v1/<name>.npz
"""
import glob
import os
import sys

TRACK4GEN = "eveworld/pipeline"
sys.path.insert(0, TRACK4GEN)

TRAIN = "/data/datasets/gagi/wmb_adapt/trainset_v1"
ANNO = "/data/datasets/gagi/wmb_adapt/t4g_anno"
OUT = "/data/datasets/gagi/wmb_adapt/aug_assets_v1"

os.environ.setdefault("T4G_W_LAT", "40")
os.environ.setdefault("T4G_WPIX", "640")

import t4g_ghost_probe as G  # noqa: E402

G.W_LAT = 40
G.WPIX = 640
G.VIDEO_ROOT = TRAIN
G.ANNO_DIR = ANNO

import t4g_aug_prep as AP  # noqa: E402  (from-import 此时读 patched 值)


def main():
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{TRAIN}/*.mp4"))
    names = [n for n in names if not os.path.exists(f"{OUT}/{n}.npz")]
    if not names:
        print("[w6] 全部资产已存在, 无需补")
        return
    argv = ["w6_aug_prep",
            "--out-dir", OUT,
            "--anno-dir", ANNO,
            "--video-root", TRAIN,
            "--vids", *names]
    # 透传分片参数
    for k in ("--shard-index", "--num-shards", "--device"):
        if k in sys.argv:
            i = sys.argv.index(k)
            argv += [k, sys.argv[i + 1]]
    sys.argv = argv
    AP.main()


if __name__ == "__main__":
    main()
