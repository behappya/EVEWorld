#!/usr/bin/env python3
"""WMB 适配 D3:从 OXE 索引抽样构造 trainset_v1。

- 防泄漏:对 WMB 50 题首帧, 剔除库内汉明距 <= HAM_THR 的所有 episode(审计落盘)
- 按 50 题源占比抽样: bridge 300 / taco 100 / berkeley 60 / jaco 40
- 每条: steps 帧序列 -> 时间均匀采样 93 帧 -> 640x480 mp4@16fps + 同名 .txt(原生指令)
输出: /data/datasets/gagi/wmb_adapt/trainset_v1/
"""
import glob
import io
import json
import os
import pickle
import random
import tarfile
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

OXE_ROOT = "/data/datasets/OpenX-Embodiment"
ADAPT = "/data/datasets/gagi/wmb_adapt"
OUT = f"{ADAPT}/trainset_v1"
WMB_IMG = "/home/jovyan/gagibench/WorldModelBench/images"
HAM_THR = 12
QUOTA = {"bridge": 300, "taco_play": 100, "berkeley_autolab_ur5": 60, "jaco_play": 40}
N_FRAMES, FPS, W, H = 93, 16, 640, 480
MIN_EP_FRAMES = 12
SEED = 42


def dhash64(img):
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    b = 0
    for r in range(8):
        for c in range(8):
            b = (b << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
    return b


def ham(a, b):
    return bin(a ^ b).count("1")


def build_exclusion():
    wmb = []
    for p in sorted(glob.glob(f"{WMB_IMG}/*")):
        n = os.path.basename(p)
        if n.split("_")[0] in ("bridge", "taco", "berkeley", "jaco", "robot"):
            wmb.append((n, dhash64(Image.open(p))))
    excluded = {}  # (ds, tar, sample) -> [wmb_name, dist]
    rows_all = {}
    for ds in QUOTA:
        rows = [json.loads(l) for l in open(f"{ADAPT}/oxe_index/{ds}.jsonl")]
        rows = [r for r in rows if r.get("first_dhash")]
        rows_all[ds] = rows
        for r in rows:
            h = int(r["first_dhash"], 16)
            for name, wh in wmb:
                d = ham(h, wh)
                if d <= HAM_THR:
                    excluded[(ds, r["tar"], r["sample"])] = [name, d]
                    break
    return rows_all, excluded


def resample_indices(n, k):
    if n <= 0:
        return []
    return [min(n - 1, round(i * (n - 1) / (k - 1))) for i in range(k)]


def convert_one(args):
    ds, tar_name, sample, instr, out_name = args
    try:
        with tarfile.open(f"{OXE_ROOT}/{ds}/{tar_name}") as tf:
            ep = pickle.load(tf.extractfile(sample))
        steps = ep["steps"]
        img_key = (ep.get("image_list") or ["image"])[0]
        frames = []
        for i in resample_indices(len(steps), N_FRAMES):
            raw = steps[i]["observation"][img_key]
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            frames.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC))
        vp = f"{OUT}/{out_name}.mp4"
        vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
        with open(f"{OUT}/{out_name}.txt", "w") as f:
            f.write(instr.strip())
        return out_name, "ok"
    except Exception as e:  # noqa: BLE001
        return out_name, f"FAIL {type(e).__name__}: {e}"


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(SEED)
    rows_all, excluded = build_exclusion()
    json.dump({f"{k[0]}/{k[1]}/{k[2]}": v for k, v in excluded.items()},
              open(f"{ADAPT}/trainset_v1_exclusion_audit.json", "w"), indent=1)
    print(f"排除 episode 数: {len(excluded)}")

    jobs = []
    manifest = []
    for ds, quota in QUOTA.items():
        cands = [r for r in rows_all[ds]
                 if (ds, r["tar"], r["sample"]) not in excluded
                 and r.get("n_frames", 0) >= MIN_EP_FRAMES
                 and r.get("instruction", "").strip()]
        random.shuffle(cands)
        picked = cands[:quota]
        print(f"{ds}: 候选 {len(cands)}, 抽 {len(picked)}")
        for r in picked:
            idx = r["sample"].replace("sample_", "").replace(".data.pickle", "")
            out_name = f"{ds}_{idx}"
            jobs.append((ds, r["tar"], r["sample"], r["instruction"], out_name))
            manifest.append({"name": out_name, "dataset": ds, "tar": r["tar"],
                             "sample": r["sample"], "instruction": r["instruction"],
                             "n_frames_src": r["n_frames"]})
    json.dump(manifest, open(f"{ADAPT}/trainset_v1_manifest.json", "w"), indent=1)

    ok = 0
    with Pool(12) as p:
        for name, st in p.imap_unordered(convert_one, jobs):
            if st == "ok":
                ok += 1
            else:
                print(f"  {name}: {st}", flush=True)
    print(f"完成 {ok}/{len(jobs)} -> {OUT}")


if __name__ == "__main__":
    main()
