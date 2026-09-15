#!/usr/bin/env python3
"""T4G-EXAM v3: inv0 从条件首帧图(ground truth)建立, 解决 v2 单帧抖动导致的
高 undetectable。检测/判定逻辑完全复用 v2, 仅改库存基线来源。

inv0 策略(优先级):
  1. 条件首帧真实图的 count_valid_instances(最可靠, 图清晰)
  2. 若条件图=0, 退回生成视频前 3 帧 count 的最大值(抵消单帧抖动)
  3. 仍为 0 -> undetectable
DUP/VANISH 判定与 v2 完全一致。
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

import t4g_probe as P
from t4g_gdino import GDinoLocator, parse_objects
from t4g_detect import detect_all
from t4g_exam_v2 import count_valid_instances, NF, HPIX, WPIX, T_LAT


def inv0_from_cond(loc, cond_img_path, name):
    if not cond_img_path or not os.path.exists(cond_img_path):
        return 0
    img = np.array(Image.open(cond_img_path).convert("RGB"))
    obj = detect_all(loc, img, name, topk=6, box_thr=0.35)
    grip = detect_all(loc, img, "robot gripper", topk=3, box_thr=0.15)
    return count_valid_instances(obj, grip)


def exam_video_v3(loc, path, prompt, cond_img):
    frames = P.sample_frames_like_training(path, NF, HPIX, WPIX)
    name = parse_objects(prompt)['mover']
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    counts = []
    for t in rep:
        fr = frames[int(t)]
        obj = detect_all(loc, fr, name, topk=6, box_thr=0.35)
        grip = detect_all(loc, fr, "robot gripper", topk=3, box_thr=0.15)
        counts.append(count_valid_instances(obj, grip))

    inv0 = inv0_from_cond(loc, cond_img, name)
    inv0_src = "cond"
    if inv0 == 0:
        inv0 = max(counts[:3]) if counts else 0
        inv0_src = "gen3"
    if inv0 == 0:
        return dict(undetectable=True, counts=counts, inv0=0, inv0_src="none",
                    dup=False, vanish=False, dup_frames=0, zero_frames=0, max_count=max(counts))

    run = mx = 0
    for c in counts:
        run = run + 1 if c > inv0 else 0
        mx = max(mx, run)
    dup = mx >= 2
    zrun = zmx = 0
    for t in range(2, min(22, len(counts))):
        zrun = zrun + 1 if counts[t] == 0 else 0
        zmx = max(zmx, zrun)
    vanish = zmx >= 3
    return dict(undetectable=False, counts=counts, inv0=inv0, inv0_src=inv0_src,
                dup=dup, vanish=vanish, dup_frames=mx, zero_frames=zmx, max_count=max(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list-file', required=True)  # arm \t video \t prompt \t cond_img
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    rows = [l.rstrip('\n').split('\t') for l in open(a.list_file) if l.strip()]
    rows = rows[a.shard_index::a.num_shards]
    loc = GDinoLocator(device=a.device)
    out = []
    for i, r in enumerate(rows):
        arm, vp, prompt, cond = (r + [''] * 4)[:4]
        try:
            rec = exam_video_v3(loc, vp, prompt, cond)
        except Exception as e:  # noqa: BLE001
            rec = dict(undetectable=True, error=str(e)[:100], counts=[], inv0=0, max_count=0,
                       dup=False, vanish=False)
        rec['arm'] = arm
        rec['video'] = os.path.basename(vp)
        out.append(rec)
        if i % 10 == 0:
            print(f"[v3 shard{a.shard_index}] {i}/{len(rows)}", flush=True)
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
