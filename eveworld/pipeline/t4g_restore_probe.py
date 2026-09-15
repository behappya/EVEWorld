#!/usr/bin/env python3
"""T4G-RESTORE (E12): 消失补回探针 —— E10 保管员定理的镜像。

命题
----
把段1(抓取前)的物体从输入抹掉(confident empty), 一次性修复, 看模型补不补回来。
E10 证明"多余物体照单保留"(retention 0.92-0.99); 本探针测镜像: "缺失物体照单保留缺失"吗?
若成立(restore≈0) -> 模型不会查首帧库存补货 -> 消失贴这门课必须开。

协议 (与 E10 同款: 一次前向修复, 不训练不跑30轮链; sigma 扫描=模拟链条不同阶段)
--------------------------------------------------------------------------------
对每条视频、每个 kind∈{target, distractor, background}、每个 mode∈{full, half}、每个 sigma:
  1. erase: 在段1窗口 [2, t_grasp) 用 ring-median 平填 (+小噪) 抹掉 box (mode=half 则 0.5 混合);
  2. 双 VAE 编码: clean(物体在) / erased(物体无);
  3. forward_x0hat(erased 加噪) -> x0_hat;
  4. restore = mean_region |x0_hat - erased| / |clean - erased|   (1=完全补回, 0=保留缺失);
     direction = cos(x0_hat - erased, clean - erased)             (确认补的就是该物体)
三 kind 的意义 (对抗性控制, 关键):
  target      首帧库存有 + 文本点名     -> 补=可能查库存 或 只听文本
  distractor  首帧库存有 + 文本不提     -> 补=真查图像库存 (排除"只听文本")
  background  非物体、任何库存都没有    -> 补=纯抗损坏反射 (证伪"restore=查库存"): 若≈target 则 restore 无意义
判决: restore(target) 且 restore(distractor) 显著 > restore(background) -> 模型会查库存补货;
      全≈background 或全≈0 -> 保管员镜像成立, 消失贴必开。
两模型对比: round0(底座习惯) vs s150(我们训练踩深多少) = 同协议各跑一次。
用法: kjob 8 卡, transformer-dir 可覆盖 (round0 / s150)。
"""
import argparse
import json
import os

import cv2
import numpy as np

import t4g_probe as P
from t4g_ghost_probe import (forward_x0hat, decode_latent, T_LAT, H_LAT, W_LAT,
                             HPIX, WPIX, NF, CELL_PX, ANNO_DIR, VIDEO_ROOT)

ASSETS_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets'


# ---------------------------------------------------------------- 段1/t_grasp
def estimate_t_grasp(anno):
    """t_grasp = target_cell 首次离开初始位 >2 格的 latent 帧 (段1 = [0, t_grasp))。"""
    per = anno['per_lat_frame']
    c0 = anno.get('target_cell_0') or (per[0]['target_cell'] if per[0]['target_cell'] else None)
    if c0 is None:
        return None
    for t in range(1, T_LAT):
        c = per[t]['target_cell']
        if c and abs(c[0] - c0[0]) + abs(c[1] - c0[1]) > 2:
            return t
    return None


def box_from_cell(cell, hw):
    """以 cell 为中心、给定像素高宽建 box (钳在画面内)。"""
    h, w = hw
    cy, cx = cell[0] * CELL_PX + CELL_PX // 2, cell[1] * CELL_PX + CELL_PX // 2
    y0 = int(np.clip(cy - h // 2, 0, HPIX - h)); x0 = int(np.clip(cx - w // 2, 0, WPIX - w))
    return [y0, x0, y0 + h, x0 + w]


def ring_fill(frame, box, rng, noise=6.0):
    """ring-median 平填: box 外一圈 ring 的中位色填满 box (+小高斯噪), 返回填充块 uint8。"""
    y0, x0, y1, x1 = box
    m = 6
    ry0, rx0, ry1, rx1 = max(0, y0 - m), max(0, x0 - m), min(HPIX, y1 + m), min(WPIX, x1 + m)
    ring = frame[ry0:ry1, rx0:rx1].reshape(-1, 3).astype(np.float32)
    med = np.median(ring, axis=0)
    fill = np.clip(med + rng.normal(0, noise, (y1 - y0, x1 - x0, 3)), 0, 255).astype(np.uint8)
    return fill


def apply_erase(frames, box, p_lo, p_hi, kind_rng, alpha=1.0):
    """在 [p_lo,p_hi) 帧的 box 处 erase (alpha=1 全抹, 0.5 渐隐)。返回新 frames。"""
    out = frames.copy()
    y0, x0, y1, x1 = box
    for p in range(p_lo, p_hi):
        fill = ring_fill(frames[p], box, kind_rng)
        out[p, y0:y1, x0:x1] = (alpha * fill + (1 - alpha) * frames[p, y0:y1, x0:x1]).astype(np.uint8)
    return out


def box_to_lat(box):
    y0, x0, y1, x1 = box
    return (y0 // 8, min(60, (y1 + 7) // 8), x0 // 8, min(96, (x1 + 7) // 8))


# ---------------------------------------------------------------- 主流程
def select_vids(min_grasp=5, min_det=16):
    out = []
    for f in sorted(os.listdir(ANNO_DIR)):
        if not f[0].isdigit():
            continue
        d = json.load(open(os.path.join(ANNO_DIR, f)))
        tg = estimate_t_grasp(d)
        if tg is not None and tg >= min_grasp and d['n_detected_frames'] >= min_det \
                and d.get('target_cell_0'):
            out.append((d['vid'], tg))
    return sorted(out, key=lambda x: int(x[0]))


def region_metrics(x0_hat, clean_lat, erased_lat, lat_box, t_lo, t_hi):
    ly0, ly1, lx0, lx1 = lat_box
    reg = (slice(None), slice(t_lo, t_hi), slice(ly0, ly1), slice(lx0, lx1))
    num = (x0_hat - erased_lat[0].cpu())[reg]
    den = (clean_lat[0].cpu() - erased_lat[0].cpu())[reg]
    d_norm = float(den.abs().mean())
    if d_norm < 1e-4:
        return None, None
    restore = float(num.abs().mean()) / d_norm
    import torch
    direction = float(torch.nn.functional.cosine_similarity(num.flatten(), den.flatten(), dim=0))
    return min(2.0, restore), direction


def run(args):
    import torch
    from giga_models.nn import EDMLoss
    dtype, device = torch.bfloat16, 'cuda'
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)
    tf_dir = args.transformer_dir or f'{args.model_dir}/transformer'
    models = P.load_models(tf_dir, f'{args.model_dir}/vae', f'{args.model_dir}/text_encoder',
                           device, dtype)
    sigmas = [float(s) for s in args.sigmas.split(',')]
    vids = select_vids()[args.shard_index::args.num_shards]
    eye_vids = set(v for v, _ in vids[:1])
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    os.makedirs(os.path.join(args.out_dir, 'eye_png'), exist_ok=True)
    print(f'[restore] shard {args.shard_index}/{args.num_shards}: {len(vids)} vids '
          f'model={os.path.basename(os.path.dirname(tf_dir))} sigmas={sigmas}', flush=True)

    def vae(fr_u8):
        t = torch.from_numpy(fr_u8).float().permute(0, 3, 1, 2) / 255.0
        t = ((t - 0.5) / 0.5).unsqueeze(0).to(device)
        return P._forward_vae(models['vae'], models['lat_mean'], models['lat_std'], t).float(), t

    out = {'records': [], 'model': os.path.basename(os.path.dirname(tf_dir))}
    for vid, t_grasp in vids:
        anno = json.load(open(os.path.join(ANNO_DIR, f'{vid}.json')))
        vp = os.path.join(VIDEO_ROOT, f'{vid}.mp4')
        if not os.path.exists(vp):
            continue
        frames = P.sample_frames_like_training(vp, NF, HPIX, WPIX)
        p_lo, p_hi = int(rep[2]), int(rep[t_grasp])
        if p_hi - p_lo < 6:
            continue
        # box: target (aug_assets 精确框, 回退 target_cell_0)
        tbox = None
        ap = os.path.join(ASSETS_DIR, f'{vid}.npz')
        if os.path.exists(ap):
            d = np.load(ap)
            if 'box' in d:
                tbox = [int(v) for v in d['box']]
        if tbox is None:
            tbox = box_from_cell(anno['target_cell_0'], (56, 56))
        thw = (tbox[2] - tbox[0], tbox[3] - tbox[1])
        kinds = {'target': tbox}
        # distractor (库存有、文本不提): 用同尺寸框
        if anno.get('distractor_cells'):
            kinds['distractor'] = box_from_cell(anno['distractor_cells'][0], thw)
        # background 对照 (非物体、任何库存都没有): 远离一切、静止背景
        rng0 = np.random.RandomState(int(vid))
        excl = {tuple(c) for c in anno.get('inventory_cells', [])} | \
               {tuple(c) for c in anno.get('distractor_cells', [])}
        for _ in range(60):
            gy, gx = rng0.randint(4, H_LAT - 4), rng0.randint(4, W_LAT - 4)
            if all(abs(gy - c[0]) + abs(gx - c[1]) > 5 for c in excl):
                kinds['background'] = box_from_cell([gy, gx], thw)
                break

        clean_lat, _ = vae(frames)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(anno['prompt'], max_length=512).to(device)

        # 首帧条件 latent (台账干净, 每 kind/sigma 复用)
        ref_c = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        ref_c = ((ref_c - 0.5) / 0.5).unsqueeze(0).to(device); ref_c[:, 1:] = 0
        ref_lat_c = P._forward_vae(models['vae'], models['lat_mean'], models['lat_std'], ref_c).float()

        for kind, box in kinds.items():
            for mode, alpha in [('full', 1.0), ('half', 0.5)]:
                krng = np.random.RandomState(hash((vid, kind, mode)) % (2**31))
                fr_er = apply_erase(frames, box, p_lo, p_hi, krng, alpha)
                fr_full = fr_er.copy(); fr_full[0] = frames[0]        # 帧0 永不抹 (清单锚)
                lat_in, _ = vae(fr_full)                              # 缺失视频 latent (loss 参照=erased)
                lb = box_to_lat(box)
                for sigma in sigmas:
                    gen = torch.Generator(device=device).manual_seed(int(vid) * 97 + int(sigma * 10))
                    x0h = forward_x0hat(models, lat_in, ref_lat_c, emb, sigma, args.fps,
                                        device, dtype, edm, gen)
                    restore, direction = region_metrics(x0h, clean_lat, lat_in, lb, 2, T_LAT)
                    if restore is None:
                        continue
                    out['records'].append(dict(vid=vid, kind=kind, mode=mode, sigma=sigma,
                                               restore=restore, direction=direction, t_grasp=t_grasp))
                    if vid in eye_vids and kind == 'target' and mode == 'full' and sigma in (0.8, 3.0):
                        rec = decode_latent(models, x0h.unsqueeze(0).to(device))
                        rows = []
                        for t in (4, min(t_grasp - 1, T_LAT - 1)):
                            p = int(rep[t])
                            a = frames[p].copy(); b = fr_full[p].copy(); c = rec[p].copy()
                            for im, tag in [(a, 'clean'), (b, 'erased-in'), (c, f'x0hat s={sigma}')]:
                                cv2.putText(im, f'{tag} t={t}', (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (0, 255, 255), 2)
                            rows.append(np.concatenate([a, b, c], axis=1))
                        cv2.imwrite(os.path.join(args.out_dir, 'eye_png', f'{vid}_s{sigma}.png'),
                                    cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
        print(f'  {vid}: t_grasp={t_grasp} kinds={list(kinds)} done', flush=True)

    p = os.path.join(args.out_dir, f'partial_shard{args.shard_index}.json')
    json.dump(out, open(p, 'w'), indent=1)
    print(f'[restore] shard {args.shard_index} -> {p}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--transformer-dir', default=None)
    ap.add_argument('--sigmas', default='0.3,0.8,1.5,3.0')
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
