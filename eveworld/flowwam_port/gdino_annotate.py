#!/usr/bin/env python3
"""FlowWAM_WorldArena GDINO 逐 latent 帧目标定位标注（IGR/TIA 共享产物）。

对 manifest 每条 episode：取前 121 帧（Wan latent T=31，t=0 对应帧0，
t>=1 对应帧 4t-1），GDINO 定位 target_name 与机械臂，输出 giga t4g_anno
同构 JSON（几何换 Wan：n_lat=31, H_lat=30, W_lat=40, cell=16px）。
支持 --shard i/n 分片与断点续跑（已存在且合法的输出跳过）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from t4g_gdino import GDinoLocator  # noqa: E402

N_LAT, H_LAT, W_LAT = 31, 30, 40
NF, WPIX, HPIX = 121, 640, 480
ARM_QUERY = "robot arm"


def lat_to_frame(t: int) -> int:
    return 0 if t == 0 else min(4 * t - 1, NF - 1)


def box_to_cells(box, gw=W_LAT, gh=H_LAT, img_w=WPIX, img_h=HPIX):
    x0, y0, x1, y1 = box
    gx0 = int(np.clip(x0 / img_w * gw, 0, gw - 1))
    gx1 = int(np.clip(x1 / img_w * gw, 0, gw - 1))
    gy0 = int(np.clip(y0 / img_h * gh, 0, gh - 1))
    gy1 = int(np.clip(y1 / img_h * gh, 0, gh - 1))
    return [[gy, gx] for gy in range(gy0, gy1 + 1) for gx in range(gx0, gx1 + 1)]


def read_lat_frames(video: str):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = {lat_to_frame(t) for t in range(N_LAT)}
    frames, idx = {}, 0
    while idx <= max(want) and idx < total:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            frames[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return frames


def annotate_episode(loc: GDinoLocator, row: dict) -> dict:
    frames = read_lat_frames(row["video"])
    per_lat, arm_cells_union = [], set()
    n_det = 0
    for t in range(N_LAT):
        fr = frames.get(lat_to_frame(t))
        entry = {"target_cell": None, "target_box_px": None, "score": None}
        if fr is not None:
            hit = loc.locate(fr, row["target_name"])
            if hit is not None:
                cx, cy, box, score = hit
                gx = int(np.clip(cx / WPIX * W_LAT, 0, W_LAT - 1))
                gy = int(np.clip(cy / HPIX * H_LAT, 0, H_LAT - 1))
                entry = {"target_cell": [gy, gx],
                         "target_box_px": [round(v, 1) for v in box],
                         "score": round(score, 4)}
                n_det += 1
            arm = loc.locate(fr, ARM_QUERY, box_thr=0.25)
            if arm is not None:
                arm_cells_union.update(map(tuple, box_to_cells(arm[2])))
        per_lat.append(entry)
    first = next((e for e in per_lat if e["target_cell"]), None)
    return {
        "vid": f"{row['task']}/{row['episode']}",
        "prompt": row["instruction"],
        "target_name": row["target_name"],
        "b_name": None,
        "gate_enabled": False,
        "gate_reason": "flowwam_port_no_gate",
        "t_arrival": None,
        "n_lat": N_LAT, "H_lat": H_LAT, "W_lat": W_LAT,
        "cell_px": 16, "n_frames": NF, "width": WPIX, "height": HPIX,
        "target_cell_0": first["target_cell"] if first else None,
        "inventory_cells": [first["target_cell"]] if first else [],
        "distractor_cells": [],
        "a_cells": [],
        "arm_cells": sorted(map(list, arm_cells_union)),
        "n_inventory": 1 if first else 0,
        "n_detected_frames": n_det,
        "per_lat_frame": per_lat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data/datasets/gagi/flowwam/igr/manifest_640.json")
    ap.add_argument("--out-dir", default="/data/datasets/gagi/flowwam/igr/anno_640")
    ap.add_argument("--shard", default="0/1", help="i/n")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    rows = json.load(open(args.manifest))
    rows = [r for k, r in enumerate(rows) if k % n == i]
    os.makedirs(args.out_dir, exist_ok=True)

    loc = GDinoLocator(device=args.device)
    done = skip = fail = 0
    for k, row in enumerate(rows):
        out = os.path.join(args.out_dir, f"{row['task']}__{row['episode']}.json")
        if os.path.exists(out):
            try:
                d = json.load(open(out))
                if d.get("n_lat") == N_LAT and len(d.get("per_lat_frame", [])) == N_LAT:
                    skip += 1
                    continue
            except Exception:
                pass
        try:
            anno = annotate_episode(loc, row)
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(anno, f)
            os.replace(tmp, out)
            done += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"FAIL {row['task']}/{row['episode']}: {type(exc).__name__}: {exc}", flush=True)
        if (k + 1) % 20 == 0:
            print(f"shard {args.shard}: {k + 1}/{len(rows)} done={done} skip={skip} fail={fail}", flush=True)
    print(f"SHARD_DONE {args.shard}: done={done} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
