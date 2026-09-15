#!/usr/bin/env python3
"""T4G-DETECT: 92 条 GT 视频逐帧 GDINO 检测 -> 训练标注缓存 (42 号 §1)。

每条视频产出 t4g_anno/<vid>.json:
  gate_enabled, t_arrival(latent帧或null), n_lat, H_lat, W_lat,
  per_lat_frame:[{target_cell:[gy,gx]或null, b_cells:[[gy,gx]...], detected:bool}],
  inventory_cells(帧0所有X实例中心格), distractor_cells(非目标X实例), a_cells, target_name, b_name, gate_reason
目标=帧0各X实例中位移最大者; 干扰物=其余X实例(静止)。
用法: python t4g_detect.py --shard-index i --num-shards n --out-dir <dir>
"""
import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

import t4g_probe as P
from t4g_gdino import GDinoLocator, parse_objects

VIDEO_ROOT = '/data/datasets/gagi/gr1_finetune_data/raw_data'
NUM_FRAMES, HIMG, WIMG = 93, 480, 768
H_LAT, W_LAT, T_LAT = 30, 48, 24          # block17 特征网格 (VAE 8x空间/16patch, 时间 93->24)
VAGUE_WORDS = ('table', 'desk', 'surface', 'floor')


def detect_all(loc, frame_rgb, name, topk=5, box_thr=0.20, text_thr=0.15):
    """检测某类物体的所有实例, 返回 [(cx,cy,box,score),...] 按分数降序。"""
    from PIL import Image
    import torch
    img = Image.fromarray(frame_rgb)
    q = name.lower().strip().rstrip('.') + '.'
    inputs = loc.proc(images=img, text=q, return_tensors='pt').to(loc.device)
    with torch.no_grad():
        out = loc.model(**inputs)
    res = loc.proc.post_process_grounded_object_detection(
        out, inputs['input_ids'], threshold=box_thr, text_threshold=text_thr,
        target_sizes=[img.size[::-1]])[0]
    dets = []
    for i in range(len(res['boxes'])):
        x0, y0, x1, y1 = res['boxes'][i].tolist()
        dets.append((0.5 * (x0 + x1), 0.5 * (y0 + y1), (x0, y0, x1, y1), float(res['scores'][i])))
    dets.sort(key=lambda d: -d[3])
    return dets[:topk]


def px_to_cell(cx, cy):
    return [int(np.clip(cy / HIMG * H_LAT, 0, H_LAT - 1)),
            int(np.clip(cx / WIMG * W_LAT, 0, W_LAT - 1))]


def box_to_cells(box):
    x0, y0, x1, y1 = box
    gy0, gx0 = px_to_cell(x0, y0); gy1, gx1 = px_to_cell(x1, y1)
    return [[gy, gx] for gy in range(min(gy0, gy1), max(gy0, gy1) + 1)
            for gx in range(min(gx0, gx1), max(gx0, gx1) + 1)]


def center_in_box(cx, cy, box):
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def process(loc, vid, prompt):
    fr = P.sample_frames_like_training(f'{VIDEO_ROOT}/{vid}.mp4', NUM_FRAMES, HIMG, WIMG)
    objs = parse_objects(prompt)
    mover_name, b_name, a_name = objs['mover'], objs.get('tgt'), objs.get('src')

    # 像素帧 -> latent 帧代表 (均匀)
    rep = np.linspace(0, NUM_FRAMES - 1, T_LAT).astype(int)

    # 帧0: 全量清单 + A + B
    inv = detect_all(loc, fr[0], mover_name)
    a_det = detect_all(loc, fr[0], a_name)[:1] if a_name else []
    b_det0 = detect_all(loc, fr[0], b_name)[:1] if b_name else []

    # 每 latent 帧: 目标(取分数最高 X)+B
    per_frame_X = []   # (cx,cy,box) 或 None
    per_frame_B = []
    for t in rep:
        xs = detect_all(loc, fr[t], mover_name, topk=3)
        per_frame_X.append((xs[0][0], xs[0][1], xs[0][2]) if xs else None)
        bs = detect_all(loc, fr[t], b_name, topk=1) if b_name else []
        per_frame_B.append(bs[0][2] if bs else (b_det0[0][2] if b_det0 else None))

    # 目标选定: 清单里位移最大的实例。用帧0清单每个实例, 在末帧找最近 X 检测, 算位移
    target_idx = 0
    if len(inv) > 1:
        disp = []
        last = per_frame_X[-1]
        for (cx, cy, box, sc) in inv:
            if last:
                disp.append((last[0] - cx) ** 2 + (last[1] - cy) ** 2)
            else:
                disp.append(0)
        target_idx = int(np.argmax(disp))
    target0 = inv[target_idx] if inv else None
    distractors = [inv[i] for i in range(len(inv)) if i != target_idx]

    # 闸门可用性
    gate_enabled, gate_reason = True, 'ok'
    if not b_name or not b_det0:
        gate_enabled, gate_reason = False, 'no_B_detected'
    elif any(w in (b_name or '').lower() for w in VAGUE_WORDS):
        gate_enabled, gate_reason = False, 'vague_container_word'
    else:
        b_area = (b_det0[0][2][2] - b_det0[0][2][0]) * (b_det0[0][2][3] - b_det0[0][2][1])
        if b_area > HIMG * WIMG * 0.35:
            gate_enabled, gate_reason = False, 'B_box_too_large'

    # 到达帧 t* (单调): 目标中心首次入 B 框
    t_arrival = None
    if gate_enabled:
        for ti in range(T_LAT):
            X, B = per_frame_X[ti], per_frame_B[ti]
            if X and B and center_in_box(X[0], X[1], B):
                t_arrival = ti
                break

    per_lat = []
    for ti in range(T_LAT):
        X, B = per_frame_X[ti], per_frame_B[ti]
        per_lat.append({
            'target_cell': px_to_cell(X[0], X[1]) if X else None,
            'b_cells': box_to_cells(B) if B else [],
            'detected': X is not None,
        })

    return {
        'vid': vid, 'prompt': prompt, 'target_name': mover_name, 'b_name': b_name,
        'gate_enabled': gate_enabled, 'gate_reason': gate_reason, 't_arrival': t_arrival,
        'n_lat': T_LAT, 'H_lat': H_LAT, 'W_lat': W_LAT,
        'target_cell_0': px_to_cell(target0[0], target0[1]) if target0 else None,
        'inventory_cells': [px_to_cell(d[0], d[1]) for d in inv],
        'distractor_cells': [px_to_cell(d[0], d[1]) for d in distractors],
        'a_cells': box_to_cells(a_det[0][2]) if a_det else [],
        'n_inventory': len(inv), 'n_detected_frames': sum(1 for x in per_frame_X if x),
        'per_lat_frame': per_lat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', default='/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    key = {i['source_file_name'].split('.')[0]: i['prompt']
           for i in json.load(open('/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json'))}
    vids = sorted(key, key=lambda x: int(x) if x.isdigit() else 1e9)
    vids = vids[a.shard_index::a.num_shards]
    loc = GDinoLocator(device=a.device)
    print(f'[detect] shard {a.shard_index}/{a.num_shards}: {len(vids)} videos', flush=True)
    for vid in vids:
        try:
            anno = process(loc, vid, key[vid])
            json.dump(anno, open(f'{a.out_dir}/{vid}.json', 'w'), ensure_ascii=False)
            print(f'  {vid}: gate={anno["gate_enabled"]}({anno["gate_reason"]}) '
                  f't*={anno["t_arrival"]} inv={anno["n_inventory"]} det={anno["n_detected_frames"]}/{T_LAT}', flush=True)
        except Exception as e:
            print(f'  {vid} FAIL: {str(e)[:120]}', flush=True)
    print('[detect] shard done', flush=True)


if __name__ == '__main__':
    main()
