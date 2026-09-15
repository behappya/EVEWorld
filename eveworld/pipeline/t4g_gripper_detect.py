#!/usr/bin/env python3
"""T4G-GRIPPER-DETECT: 92 条逐帧检测机械爪 -> gripper_anno/<vid>.json (空爪监督用)。

每 latent 帧检测 "robot gripper" (最多 2 个爪), 存中心格 + 框格。
判"空爪": 该爪框与当前物体格距离 > thr (未持物) -> 该爪区标 W_GRIP。
先跑少量 + 人眼验框 (--viz), 准了再全量。GDINO 需 train venv (transformers 5.11)。
"""
import argparse
import json
import os

import numpy as np

import t4g_probe as P
from t4g_gdino import GDinoLocator
from t4g_detect import detect_all, px_to_cell, box_to_cells

VIDEO_ROOT = '/data/datasets/gagi/gr1_finetune_data/raw_data'
NUM_FRAMES, HIMG, WIMG = 93, 480, 768
T_LAT = 24
GRIPPER_QUERIES = 'robot gripper. robotic hand.'


def detect_grippers(loc, frame_rgb, topk=2, box_thr=0.25, text_thr=0.2):
    from PIL import Image
    import torch
    img = Image.fromarray(frame_rgb)
    inputs = loc.proc(images=img, text=GRIPPER_QUERIES, return_tensors='pt').to(loc.device)
    with torch.no_grad():
        out = loc.model(**inputs)
    res = loc.proc.post_process_grounded_object_detection(
        out, inputs['input_ids'], threshold=box_thr, text_threshold=text_thr,
        target_sizes=[img.size[::-1]])[0]
    dets = []
    for i in range(len(res['boxes'])):
        x0, y0, x1, y1 = res['boxes'][i].tolist()
        area = (x1 - x0) * (y1 - y0)
        if area > HIMG * WIMG * 0.25:            # 过大误检(整臂)跳过
            continue
        dets.append((0.5 * (x0 + x1), 0.5 * (y0 + y1), (x0, y0, x1, y1), float(res['scores'][i])))
    dets.sort(key=lambda d: -d[3])
    return dets[:topk]


def process(loc, vid):
    fr = P.sample_frames_like_training(f'{VIDEO_ROOT}/{vid}.mp4', NUM_FRAMES, HIMG, WIMG)
    rep = np.linspace(0, NUM_FRAMES - 1, T_LAT).astype(int)
    per = []
    for t in rep:
        gs = detect_grippers(loc, fr[t])
        per.append([{'center': px_to_cell(g[0], g[1]), 'cells': box_to_cells(g[2]),
                     'score': round(g[3], 3)} for g in gs])
    return {'vid': vid, 'n_lat': T_LAT, 'per_lat_gripper': per,
            'n_det_frames': sum(1 for p in per if p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', default='/data/datasets/gagi/eve_v2_outputs/track4gen_probe/gripper_anno')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    key = {i['source_file_name'].split('.')[0]: i['prompt']
           for i in json.load(open('/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json'))}
    vids = sorted(key, key=lambda x: int(x) if x.isdigit() else 1e9)
    vids = vids[a.shard_index::a.num_shards]
    if a.limit:
        vids = vids[:a.limit]
    loc = GDinoLocator(device=a.device)
    print(f'[gripper] shard {a.shard_index}/{a.num_shards}: {len(vids)} videos', flush=True)
    for vid in vids:
        try:
            anno = process(loc, vid)
            json.dump(anno, open(f'{a.out_dir}/{vid}.json', 'w'))
            avg = np.mean([len(p) for p in anno['per_lat_gripper']])
            print(f'  {vid}: det={anno["n_det_frames"]}/{T_LAT} avg_grippers/frame={avg:.1f}', flush=True)
        except Exception as e:
            print(f'  {vid} FAIL: {str(e)[:120]}', flush=True)
    print('[gripper] shard done', flush=True)


if __name__ == '__main__':
    main()
