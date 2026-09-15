#!/usr/bin/env python3
"""EWMBench 适配训练集定稿:任务均衡下采样 -> agibot_ewm_train_final/。

规则: 每任务上限 CAP(默认150), 超出按固定 seed 随机下采样(优先保留不同 episode,
同 episode 内按段序);不足全收。输出 mp4+txt 硬链接(零拷贝)+ 定稿清单。
用法: python w10_finalize_agibot_trainset.py [--cap 150]
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

SRC = "/data/datasets/gagi/agibot_ewm_train"
DST = "/data/datasets/gagi/agibot_ewm_train_final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(DST, exist_ok=True)

    by_task = defaultdict(lambda: defaultdict(list))  # task -> ep -> [name]
    for p in sorted(glob.glob(f"{SRC}/*.mp4")):
        name = os.path.splitext(os.path.basename(p))[0]
        task, ep, seg = name.split("_")
        by_task[task][ep].append(name)

    manifest = {}
    for task, eps in sorted(by_task.items()):
        # 轮转采样: 先每 episode 取第 1 段, 再取第 2 段... 直到 CAP(episode 多样性优先)
        ep_list = list(eps.keys())
        random.shuffle(ep_list)
        picked = []
        rnd = 0
        while len(picked) < a.cap:
            added = False
            for ep in ep_list:
                segs = sorted(eps[ep])
                if rnd < len(segs):
                    picked.append(segs[rnd])
                    added = True
                    if len(picked) >= a.cap:
                        break
            if not added:
                break
            rnd += 1
        manifest[task] = sorted(picked)
        for name in picked:
            for ext in (".mp4", ".txt"):
                d = f"{DST}/{name}{ext}"
                if not os.path.exists(d):
                    os.link(f"{SRC}/{name}{ext}", d)

    total = sum(len(v) for v in manifest.values())
    json.dump({"cap": a.cap, "seed": a.seed, "total": total,
               "per_task": {k: len(v) for k, v in manifest.items()},
               "files": manifest},
              open(f"{DST}/_finalize_manifest.json", "w"), indent=1)
    print("定稿分布:", {k: len(v) for k, v in manifest.items()}, "| 总:", total)


if __name__ == "__main__":
    main()
