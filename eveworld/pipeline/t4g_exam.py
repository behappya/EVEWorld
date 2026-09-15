#!/usr/bin/env python3
"""T4G-EXAM: GDINO 恒存性体检 (44 号评测仪器, 确定性数病例; LLM 判官对复制病失明的替代)。

对每条生成视频逐 latent 帧检测目标物实例数, 三项判定:
  inv0    = 帧0 实例数 (帧0=条件帧=真实图, 即库存清单; =0 则该条判 undetectable 剔除)
  DUP     = 存在 >=2 连续帧 count > inv0        (复制/终态提前/夹爪出生均表现为多实例)
  VANISH  = 帧 [2,22) 内存在 >=3 连续帧 count==0 (原物消失; 抓取遮挡有假阳但各臂同偏, 相对可比)
去重: 同帧检测中心距 <=40px 视为同一实例 (E6/E7 同口径)。box_thr=0.20 (t4g_detect 同款)。
用法: 经 t4g_exam_dispatch.py 分片; --list-file 每行 "arm\tvideo_path\tprompt"。
"""
import argparse
import json
import os

import numpy as np

import t4g_probe as P
from t4g_gdino import GDinoLocator, parse_objects
from t4g_detect import detect_all

NF, HPIX, WPIX, T_LAT = 93, 480, 768, 24


def count_instances(dets, min_dist=40):
    centers = []
    for (cx, cy, box, sc) in dets:
        if all(abs(cx - a) + abs(cy - b) > min_dist for a, b in centers):
            centers.append((cx, cy))
    return len(centers)


def exam_video(loc, path, prompt):
    frames = P.sample_frames_like_training(path, NF, HPIX, WPIX)
    name = parse_objects(prompt)['mover']
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    counts = []
    for t in rep:
        dets = detect_all(loc, frames[int(t)], name, topk=6)
        counts.append(count_instances(dets))
    inv0 = counts[0]
    if inv0 == 0:
        return dict(undetectable=True, counts=counts, inv0=0, dup=False, vanish=False,
                    dup_frames=0, zero_frames=0, max_count=max(counts))
    # DUP: >=2 连续帧超库存
    dup = False; run = 0; dup_frames = 0
    for c in counts:
        if c > inv0:
            run += 1; dup_frames += 1
            if run >= 2:
                dup = True
        else:
            run = 0
    # VANISH: 帧[2,22) 内 >=3 连续 0
    vanish = False; run = 0; zero_frames = 0
    for t in range(2, min(22, T_LAT)):
        if counts[t] == 0:
            run += 1; zero_frames += 1
            if run >= 3:
                vanish = True
        else:
            run = 0
    return dict(undetectable=False, counts=counts, inv0=inv0, dup=dup, vanish=vanish,
                dup_frames=dup_frames, zero_frames=zero_frames, max_count=max(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list-file', required=True)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    items = []
    for line in open(a.list_file):
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 3:
            items.append(parts)
    items = items[a.shard_index::a.num_shards]
    loc = GDinoLocator(device=a.device)
    print(f'[exam] shard {a.shard_index}/{a.num_shards}: {len(items)} videos', flush=True)
    recs = []
    for i, (arm, path, prompt) in enumerate(items):
        try:
            r = exam_video(loc, path, prompt)
            r.update(arm=arm, video=os.path.basename(path))
            recs.append(r)
            if (i + 1) % 20 == 0:
                print(f'  {i+1}/{len(items)}', flush=True)
        except Exception as e:
            print(f'  FAIL {path}: {str(e)[:100]}', flush=True)
    json.dump(recs, open(a.out, 'w'))
    print(f'[exam] shard {a.shard_index} -> {a.out} ({len(recs)} recs)', flush=True)


if __name__ == '__main__':
    main()
