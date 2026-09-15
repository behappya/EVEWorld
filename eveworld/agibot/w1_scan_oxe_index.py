#!/usr/bin/env python3
"""WMB 适配 D1:扫描本地 OXE 4 源(bridge/taco_play/berkeley_autolab_ur5/jaco_play),
为每个 episode 建索引:tar 分片、sample 名、语言指令、帧数、首帧 dhash(64bit)。

输出: /data/datasets/gagi/wmb_adapt/oxe_index/<dataset>.jsonl
用途: 与 WMB robotics 50 题首帧做感知哈希去重 + 按指令/时长抽样构造 trainset。
多进程按 tar 分片并行。
"""
import argparse
import glob
import io
import json
import os
import pickle
import tarfile
from multiprocessing import Pool

from PIL import Image

OXE_ROOT = "/data/datasets/OpenX-Embodiment"
OUT_ROOT = "/data/datasets/gagi/wmb_adapt/oxe_index"
DATASETS = ["bridge", "taco_play", "berkeley_autolab_ur5", "jaco_play"]


def dhash64(img: Image.Image) -> str:
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
    return f"{bits:016x}"


def first_image_and_instr(ep: dict):
    steps = ep.get("steps") or []
    if not steps:
        return None, "", 0
    obs = steps[0].get("observation", {})
    img_key = (ep.get("image_list") or ["image"])[0]
    raw = obs.get(img_key)
    instr = obs.get("natural_language_instruction", b"")
    if isinstance(instr, bytes):
        instr = instr.decode("utf-8", "ignore")
    img = Image.open(io.BytesIO(raw)) if isinstance(raw, (bytes, bytearray)) else None
    return img, instr.strip(), len(steps)


def scan_tar(args):
    tar_path, out_path = args
    rows = []
    try:
        with tarfile.open(tar_path) as tf:
            for m in tf:
                if not m.name.endswith(".data.pickle"):
                    continue
                try:
                    ep = pickle.load(tf.extractfile(m))
                    img, instr, n = first_image_and_instr(ep)
                    rows.append({
                        "tar": os.path.basename(tar_path),
                        "sample": m.name,
                        "instruction": instr,
                        "n_frames": n,
                        "first_dhash": dhash64(img) if img is not None else "",
                        "wh": list(img.size) if img is not None else None,
                    })
                except Exception as e:  # noqa: BLE001
                    rows.append({"tar": os.path.basename(tar_path), "sample": m.name,
                                 "error": f"{type(e).__name__}: {e}"})
    except Exception as e:  # noqa: BLE001
        rows.append({"tar": os.path.basename(tar_path), "error": f"TAR {type(e).__name__}: {e}"})
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tar_path, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    jobs = []
    for ds in a.datasets:
        shard_dir = os.path.join(OUT_ROOT, f"{ds}_shards")
        os.makedirs(shard_dir, exist_ok=True)
        for t in sorted(glob.glob(f"{OXE_ROOT}/{ds}/*.tar")):
            out = os.path.join(shard_dir, os.path.basename(t) + ".jsonl")
            if not os.path.exists(out):
                jobs.append((t, out))
    print(f"待扫描 tar: {len(jobs)}", flush=True)
    with Pool(a.workers) as p:
        for i, (t, n) in enumerate(p.imap_unordered(scan_tar, jobs), 1):
            print(f"[{i}/{len(jobs)}] {os.path.basename(t)}: {n} eps", flush=True)
    # 合并
    for ds in a.datasets:
        merged = os.path.join(OUT_ROOT, f"{ds}.jsonl")
        with open(merged, "w") as out:
            for sf in sorted(glob.glob(f"{OUT_ROOT}/{ds}_shards/*.jsonl")):
                out.write(open(sf).read())
        n = sum(1 for _ in open(merged))
        print(f"[merged] {ds}: {n} rows -> {merged}", flush=True)


if __name__ == "__main__":
    main()
