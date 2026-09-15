#!/usr/bin/env python3
"""T4G-VIZ: 人工看图诊断——把特征相似度热力图/argmax 落点/首帧污染对照导成 PNG。

回答: EPE=5.5 格的 NO-GO 是"特征真没结构"还是"读出方式/指标"的假阴性?
对单条真实视频、选定层, 对一个动组 query cell:
  行 = 若干目标帧; 每行三联:
    [左] 真实帧 + 真值位置(绿) + argmax 落点(红)
    [中] 相似度热力图叠真实帧 (ref t0 -> t, 首帧 query)
    [右] 相似度热力图叠真实帧 (t-1 -> t, 前一帧 query, 排除首帧污染)
看点: 中图/右图有没有"清晰峰落在物体上"(即便 argmax 不精确)。

GPU 作业 (复用 t4g_probe 的特征提取)。用法见 t4g_viz_kjob 或直接:
  python t4g_viz.py --video-id 13 --layers block13,block17,block21 --sigma 0.4 --out-dir <dir>
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

import t4g_probe as P


def cosine_simmap(feat_thwd, qy, qx, t_query, t_target):
    """query = feat[t_query, qy, qx]; 返回 feat[t_target] 全网格的余弦相似度 (H, W)。"""
    import torch
    feat = feat_thwd.float()
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    q = fn[t_query, qy, qx]                       # (D,)
    sim = (fn[t_target].reshape(H * W, D) @ q).reshape(H, W)
    return sim.cpu().numpy()


def overlay_heat(frame_bgr, simmap):
    """simmap (H,W) in [-1,1] -> resize 到帧尺寸, jet 叠加。"""
    h, w = frame_bgr.shape[:2]
    s = (simmap - simmap.min()) / (simmap.max() - simmap.min() + 1e-8)
    s = cv2.resize((s * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(s, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.5, heat, 0.5, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-id', required=True)
    ap.add_argument('--video-root', default='/data/datasets/gagi/gr1_finetune_data/raw_data')
    ap.add_argument('--model-dir', default='/data/datasets/gagi/eve_v2_outputs/anchor_models/probe_anmix_s200')
    ap.add_argument('--layers', default='block13,block17,block21')
    ap.add_argument('--sigma', type=float, default=0.4)
    ap.add_argument('--num-frames', type=int, default=93)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    import torch
    from giga_models.nn import EDMLoss
    device = 'cuda'
    dtype = torch.bfloat16
    os.makedirs(a.out_dir, exist_ok=True)
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)

    # --- 载模型 + 前向 + hook (复用 t4g_probe) ---
    models = P.load_models(f'{a.model_dir}/transformer', f'{a.model_dir}/vae',
                           f'{a.model_dir}/text_encoder', device, dtype)
    frames_rgb = P.sample_frames_like_training(
        f'{a.video_root}/{a.video_id}.mp4', a.num_frames, a.height, a.width)
    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
    prompt = P.read_prompt(a.video_root, a.video_id)
    fr = torch.from_numpy(frames_rgb).float().permute(0, 3, 1, 2) / 255.0
    frames_norm = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = models['text_encoder'].encode_prompts(prompt, max_length=512).to(device)
    store = P.forward_with_hooks(models, frames_norm, emb, a.sigma, a.fps, device, dtype, edm)

    # --- 光流(仅供真值轨迹) ---
    flows = P.dense_flow_sequence(frames_rgb)
    any_feat = store[a.layers.split(',')[0]]
    T, H, W, _ = any_feat.shape

    # --- GDINO 定位真物体当 query (替代光流最大格) ---
    from t4g_gdino import GDinoLocator, parse_objects, pixel_to_grid
    objs = parse_objects(prompt)
    loc = GDinoLocator(device=device)
    hit = loc.locate(frames_rgb[0], objs['mover'])
    if hit is not None:
        cx, cy, box, score = hit
        qy, qx = pixel_to_grid(cx, cy, a.width, a.height, W, H)
        print(f'[viz] GDINO "{objs["mover"]}" @ px({cx:.0f},{cy:.0f}) score={score:.2f} -> cell({qy},{qx})', flush=True)
    else:
        tot = P.motion_grid(flows, H, W)
        dyn, _ = P.classify_dyn_static(tot)
        qy, qx = max(dyn, key=lambda c: tot[c[0], c[1]]) if dyn else (H // 2, W // 2)
        print(f'[viz] GDINO 未框到 "{objs["mover"]}", 回退光流最大格 ({qy},{qx})', flush=True)
    pmap = P.pixel_to_latent_map(len(frames_bgr), T)
    cell_h = flows[0].shape[0] / H
    cell_w = flows[0].shape[1] / W
    gt = P.gt_trajectories(flows, [(qy, qx)], pmap, cell_h, cell_w)  # (T,1,2)

    # 目标帧: 均匀取 4 个 latent 帧
    tgt_ts = [int(x) for x in np.linspace(1, T - 1, 4)]
    sy, sx = a.height / H, a.width / W

    def latent_to_pix_frame(t):
        return frames_bgr[min(len(frames_bgr) - 1, int(pmap[t]))]

    for layer in a.layers.split(','):
        feat = store[layer].to(device)
        rows = []
        for t in tgt_ts:
            base = latent_to_pix_frame(t).copy()
            gy, gx = gt[t, 0]
            # ref t0 -> t
            sm0 = cosine_simmap(feat, qy, qx, 0, t)
            a0 = np.unravel_index(sm0.argmax(), sm0.shape)
            # t-1 -> t
            sm1 = cosine_simmap(feat, int(gt[t - 1, 0, 0]) if False else qy, qx, t - 1, t)
            a1 = np.unravel_index(sm1.argmax(), sm1.shape)

            left = base.copy()
            cv2.circle(left, (int(gx * sx + sx / 2), int(gy * sy + sy / 2)), 10, (0, 255, 0), 2)   # GT 绿
            cv2.circle(left, (int(a0[1] * sx + sx / 2), int(a0[0] * sy + sy / 2)), 8, (0, 0, 255), 2)  # argmax 红
            cv2.putText(left, f't={t} GT(grn) amax(red)', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            mid = overlay_heat(base, sm0)
            cv2.putText(mid, f'ref0->t peak={sm0.max():.2f}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            right = overlay_heat(base, sm1)
            cv2.putText(right, f't-1->t peak={sm1.max():.2f}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rows.append(np.hstack([left, mid, right]))
        grid = np.vstack(rows)
        # 顶注: query cell 在首帧的位置
        f0 = frames_bgr[0].copy()
        cv2.circle(f0, (int(qx * sx + sx / 2), int(qy * sy + sy / 2)), 12, (0, 255, 255), 3)
        cv2.putText(f0, f'{a.video_id} {layer} sig{a.sigma} query(cyan) "{prompt[:40]}"',
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        f0r = cv2.resize(f0, (grid.shape[1], int(f0.shape[0] * grid.shape[1] / f0.shape[1])))
        out = np.vstack([f0r, grid])
        dst = f'{a.out_dir}/viz_{a.video_id}_{layer}_sig{a.sigma}.jpg'
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'[viz] {dst}  query=({qy},{qx}) T={T}', flush=True)


if __name__ == '__main__':
    main()
