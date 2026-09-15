#!/usr/bin/env python3
"""T4G-GHOST: 影子存在性探针 (44 号阶段0) + 保留探针 (合成增广可行性)。

命题 (G2 对冲-锐化机制的可证伪预测)
------------------------------------
高噪一次性修复 x0_hat 在"确定该空"的时空区 (到达前的 B 区) 的偏离, 应系统性高于
纹理匹配的静止控制区 —— 即模型在不确定时把概率质量"对冲"到了目标位置 (影子)。

测什么 (每条视频 x 每 sigma)
----------------------------
dev(c,t) = mean_ch |x0_hat - x0| 在 latent 2x2 池化到 30x48 特征格。分区:
  B_pre      到达前 B 区 (t <= 0.75*t_arr, GT静止过滤)   <- 影子主证区
  B_pre_bins B 区 dev 随 t/t_arr 的 4 段时间纹           <- ramp=影子, 平=纹理混淆
  A_post     拿走后的原位 A 区                            <- respawn 影子第二证区
  distr      干扰物格 (静止纹理控制, 有则测)
  static_far 远离一切目标/路径的静背景 (地板+tol标定)
  path_dt    路径格 dev 随 |t - t_visit| 衰减             <- 模型时间涂抹宽度, 定走廊Δ
  corridor   目标当前格 +-1 (位置不确定性幅度, 对照)
判据 (方向性; 结论须人眼 PNG + CI):
  paired(B_pre - static_far) > 0 且随 sigma 增强 且 B_pre_bins 递增  -> 影子实锤
保留探针 (--paste): 把帧0目标物 patch 贴到 B 中心 (窗口 [t_arr/2, t_arr-1)),
  retention = |x0_hat - x0_clean| / |x0_paste - x0_clean| (贴入窗口内, 越接近1=修复保留复制品)
  direction = cos(x0_hat - x0_clean, x0_paste - x0_clean)  (确认偏离方向就是贴入物)
人眼产物: eye_png/{vid}_s{sigma}.png 三联图 (GT | decode(x0) roundtrip | decode(x0_hat), B框红/目标绿)
          paste_png/{vid}_a{alpha}_s{sigma}.png (贴入输入 | 修复输出)
用法: kjob 8 卡, 经 t4g_ghost_dispatch.py 分片合并。
"""
import argparse
import json
import os

import cv2
import numpy as np

import t4g_probe as P

ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
VIDEO_ROOT = '/data/datasets/gagi/gr1_finetune_data/raw_data'
T_LAT, H_LAT, W_LAT = 24, 30, 48
HPIX, WPIX, NF = 480, 768, 93
CELL_PX = 16                     # 480/30 = 768/48 = 16 像素/格
THETA_STATIC = 3.0               # GT 静止阈 (灰度差, P-M 地图探针复核)


# ---------------------------------------------------------------- 纯 CPU 逻辑 (smoke 可测)
def cell_motion(frames, rep):
    """相邻代表帧灰度绝对差 -> (T_LAT, 30, 48)。motion[0]=motion[1]。"""
    grays = [cv2.cvtColor(frames[p], cv2.COLOR_RGB2GRAY).astype(np.float32) for p in rep]
    mo = np.zeros((T_LAT, H_LAT, W_LAT), np.float32)
    for t in range(1, T_LAT):
        mo[t] = P._avgpool_grid(np.abs(grays[t] - grays[t - 1]), H_LAT, W_LAT)
    mo[0] = mo[1]
    return mo


def _dilate(cells, r, hi=H_LAT, wi=W_LAT):
    out = set()
    for (gy, gx) in cells:
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy, xx = gy + dy, gx + dx
                if 0 <= yy < hi and 0 <= xx < wi:
                    out.add((yy, xx))
    return out


def build_zones(anno, motion, theta=THETA_STATIC):
    """返回 dict[zone_name] = list[(t, gy, gx)] + 附加 profile 索引。"""
    t_arr = anno['t_arrival']
    per = anno['per_lat_frame']
    static = motion < theta                                        # (T,30,48)
    b_sets = [set(map(tuple, fr['b_cells'])) for fr in per]
    visited = {}                                                   # cell -> [t...]
    for t, fr in enumerate(per):
        if fr['target_cell']:
            visited.setdefault(tuple(fr['target_cell']), []).append(t)
    vis_cells = set(visited)
    distr = _dilate([tuple(c) for c in anno.get('distractor_cells', [])], 1)
    a_cells = set(map(tuple, anno.get('a_cells', [])))

    z = {k: [] for k in ['B_pre', 'B_post', 'A_post', 'distr', 'static_far', 'corridor']}
    bins = {k: [] for k in ['bin0', 'bin1', 'bin2', 'bin3']}       # B区 t/t_arr 时间纹
    path = {f'dt{d}': [] for d in range(1, 9)}                     # 路径格时间涂抹

    # A_post 起点: 目标离开 A 框后 2 帧
    t_depart = None
    if a_cells:
        was_in = False
        for t, fr in enumerate(per):
            c = tuple(fr['target_cell']) if fr['target_cell'] else None
            if c and c in a_cells:
                was_in = True
            elif c and was_in:
                t_depart = t
                break

    for t in range(1, T_LAT):
        cur = tuple(per[t]['target_cell']) if per[t]['target_cell'] else None
        cur_nb = _dilate([cur], 2) if cur else set()
        # B 区
        for c in b_sets[t]:
            if t >= t_arr:
                z['B_post'].append((t, c[0], c[1]))
            elif static[t, c[0], c[1]] and c not in cur_nb:
                rb = t / max(t_arr, 1)
                bins[f'bin{min(3, int(rb * 4))}'].append((t, c[0], c[1]))
                if rb <= 0.75:
                    z['B_pre'].append((t, c[0], c[1]))
        # A 原位 (拿走后)
        if t_depart is not None and t >= t_depart + 2:
            for c in a_cells:
                if static[t, c[0], c[1]] and c not in cur_nb and c not in b_sets[t]:
                    z['A_post'].append((t, c[0], c[1]))
        # 干扰物 (纹理控制)
        for c in distr:
            if static[t, c[0], c[1]] and c not in b_sets[t]:
                z['distr'].append((t, c[0], c[1]))
        # 走廊对照 (目标当前格 +-1)
        if cur:
            for c in _dilate([cur], 1):
                z['corridor'].append((t, c[0], c[1]))
        # 路径格时间涂抹: 该格此刻静止、非B、离访问时刻 dt 帧
        for c, tv in visited.items():
            if static[t, c[0], c[1]] and c not in b_sets[t] and c not in cur_nb:
                dt = min(abs(t - x) for x in tv)
                if 1 <= dt <= 8:
                    path[f'dt{dt}'].append((t, c[0], c[1]))
    # 远端静背景: 离一切已知内容 >5 格、全程静止
    excl = _dilate(vis_cells | set().union(*b_sets) | distr | a_cells, 5) if b_sets else set()
    far = [(gy, gx) for gy in range(H_LAT) for gx in range(W_LAT)
           if (gy, gx) not in excl and static[1:, gy, gx].all()]
    rng = np.random.RandomState(0)
    far = [far[i] for i in rng.permutation(len(far))[:60]]
    for t in range(1, T_LAT):
        for c in far:
            z['static_far'].append((t, c[0], c[1]))
    z.update({f'Bbin_{k}': v for k, v in bins.items()})
    z.update({f'path_{k}': v for k, v in path.items()})
    return z


def zone_means(dev, zones):
    """dev (T,30,48) -> 每 zone 的均值 (空 zone -> nan)。"""
    out = {}
    for k, idx in zones.items():
        out[k] = float(np.mean([dev[t, y, x] for (t, y, x) in idx])) if idx else float('nan')
    return out


def cell_dev(x0_hat, x0):
    """(z,T,60,96) torch float -> (T,30,48) numpy: 通道 L1 均值 + 2x2 空间池化。"""
    d = (x0_hat - x0).abs().mean(0)
    T = d.shape[0]
    return d.reshape(T, H_LAT, 2, W_LAT, 2).mean(axis=(2, 4)).numpy()


# ---------------------------------------------------------------- GPU 前向
def forward_x0hat(models, latents, ref_latents, emb, sigma, fps, device, dtype, edm, gen):
    """一次性修复。latents/ref_latents: (1,z,T,h,w) 归一化 latent (float32, device)。
    返回 x0_hat (z,T,h,w) float32 cpu。首帧照 trainer 用 ref 替换。"""
    import torch
    B, Tlat = 1, latents.shape[2]
    sigma_t = torch.tensor([sigma], device=device, dtype=torch.float32).view(B, 1)
    noise = torch.randn(latents.reshape(B, -1).shape, generator=gen, device=device)
    input_latents, c_noise = edm.add_noise(latents.float(), noise=noise, sigma=sigma_t)
    input_latents = input_latents.to(dtype)
    ref_mask = torch.zeros((B, 1, Tlat, 1, 1), dtype=dtype, device=device)
    ref_mask[:, :, 0] = 1.0
    augment_sigma = torch.tensor([1e-4], device=device, dtype=input_latents.dtype).view(1, 1, 1, 1, 1)
    input_latents = ref_mask * ref_latents.to(dtype) + (1 - ref_mask) * input_latents
    input_masks = ref_mask.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])
    x_in = torch.cat([input_latents, input_masks], dim=1)
    timesteps = c_noise.to(dtype).view(1, 1, 1, 1, 1).expand(B, -1, Tlat, -1, -1)
    t_cond = augment_sigma / (augment_sigma + 1)
    timesteps = (ref_mask * t_cond + (1 - ref_mask) * timesteps).to(dtype)
    padding_mask = torch.zeros((B, 1, HPIX, WPIX), dtype=dtype, device=device)
    with torch.no_grad():
        pred = models['transformer'](x=x_in, timesteps=timesteps, crossattn_emb=emb.to(dtype),
                                     padding_mask=padding_mask, fps=fps)
    x0_hat = edm.denoise(pred.float()).reshape(latents.shape)
    x0_hat[:, :, 0] = ref_latents.float()[:, :, 0]
    return x0_hat[0].float().cpu()


def decode_latent(models, lat):
    """归一化 latent (1,z,T,h,w) -> uint8 (Tpix,H,W,3)。"""
    import torch
    vae = models['vae']
    raw = (lat.to(models['lat_mean'].device) / models['lat_std'].float() + models['lat_mean'].float()).to(vae.dtype)
    with torch.no_grad():
        px = vae.decode(raw).sample
    return ((px.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)[0].permute(1, 2, 3, 0).cpu().numpy()


# ---------------------------------------------------------------- 人眼产物
def _draw_boxes(img, anno, t):
    im = np.ascontiguousarray(img.copy())
    bc = anno['per_lat_frame'][t]['b_cells']
    if bc:
        ys = [c[0] for c in bc]; xs = [c[1] for c in bc]
        cv2.rectangle(im, (min(xs) * CELL_PX, min(ys) * CELL_PX),
                      ((max(xs) + 1) * CELL_PX, (max(ys) + 1) * CELL_PX), (255, 40, 40), 2)
    tc = anno['per_lat_frame'][t]['target_cell']
    if tc:
        cv2.circle(im, (tc[1] * CELL_PX + 8, tc[0] * CELL_PX + 8), 10, (40, 255, 40), 2)
    return im


def save_eye_png(out, frames, rec_rt, rec_hat, anno, rep, sigma, path):
    """三联图: GT | roundtrip | x0_hat, 4 个时刻 (0.3/0.6/0.9*t_arr, t_arr+2)。"""
    t_arr = anno['t_arrival']
    ts = sorted({max(1, int(t_arr * f)) for f in (0.3, 0.6, 0.9)} | {min(T_LAT - 1, t_arr + 2)})
    rows = []
    for t in ts:
        p = rep[t]
        row = [_draw_boxes(frames[p], anno, t), _draw_boxes(rec_rt[p], anno, t), _draw_boxes(rec_hat[p], anno, t)]
        for i, (im, tag) in enumerate(zip(row, ['GT', 'VAE-roundtrip', f'x0hat s={sigma}'])):
            cv2.putText(im, f'{tag} t={t}/{t_arr}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            row[i] = im
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------- 保留探针
def run_paste(models, loc, anno, frames, lat_clean, ref_lat, emb, fps, device, dtype, edm,
              vid, png_dir, sigmas=(0.3, 0.8, 1.5), alphas=(0.5, 1.0)):
    import torch
    from t4g_detect import detect_all
    t_arr = anno['t_arrival']
    dets = detect_all(loc, frames[0], anno['target_name'], topk=3)
    if not dets:
        return None
    x0b, y0b, x1b, y1b = [max(0, int(v)) for v in dets[0][2]]
    patch = frames[0][y0b:y1b, x0b:x1b].copy()
    ph, pw = patch.shape[:2]
    if ph < 8 or pw < 8:
        return None
    bc = anno['per_lat_frame'][max(0, t_arr - 1)]['b_cells'] or anno['per_lat_frame'][0]['b_cells']
    if not bc:
        return None
    ys = [c[0] for c in bc]; xs = [c[1] for c in bc]
    cy = int((min(ys) + max(ys) + 1) / 2 * CELL_PX)
    cx = int((min(xs) + max(xs) + 1) / 2 * CELL_PX)
    tp0, tp1 = max(2, t_arr // 2), t_arr - 1
    if tp1 - tp0 < 2:
        return None
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    p0, p1 = int(rep[tp0]), int(rep[tp1])
    y0 = max(0, cy - ph // 2); x0 = max(0, cx - pw // 2)
    y1 = min(HPIX, y0 + ph); x1 = min(WPIX, x0 + pw)
    sub = patch[:y1 - y0, :x1 - x0].astype(np.float32)
    # 贴入窗口的 latent 索引 (帧内 & 空间)
    lt_lo, lt_hi = tp0 + 1, tp1
    ly0, lx0, ly1, lx1 = y0 // 8, x0 // 8, min(60, (y1 + 7) // 8), min(96, (x1 + 7) // 8)
    res = []
    for alpha in alphas:
        fp = frames.copy()
        for pi in range(p0, p1):
            reg = fp[pi, y0:y1, x0:x1].astype(np.float32)
            fp[pi, y0:y1, x0:x1] = (alpha * sub + (1 - alpha) * reg).astype(np.uint8)
        fr = torch.from_numpy(fp).float().permute(0, 3, 1, 2) / 255.0
        fr = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        lat_paste = P._forward_vae(models['vae'], models['lat_mean'], models['lat_std'], fr).float()
        delta = (lat_paste - lat_clean)[0, :, lt_lo:lt_hi, ly0:ly1, lx0:lx1].cpu()
        denom = float(delta.abs().mean())
        if denom < 1e-4:
            continue
        for sigma in sigmas:
            gen = torch.Generator(device=device).manual_seed(int(vid) * 100 + int(sigma * 10))
            x0h = forward_x0hat(models, lat_paste, ref_lat, emb, sigma, fps, device, dtype, edm, gen)
            num = (x0h - lat_clean[0].cpu())[:, lt_lo:lt_hi, ly0:ly1, lx0:lx1]
            retention = float(num.abs().mean()) / denom
            dcos = float(torch.nn.functional.cosine_similarity(
                num.flatten(), delta.flatten(), dim=0))
            res.append(dict(alpha=alpha, sigma=sigma, retention=retention, direction=dcos,
                            denom=denom, win=[tp0, tp1]))
            if abs(alpha - 1.0) < 1e-6 and abs(sigma - 0.8) < 1e-6 and png_dir:
                rec = decode_latent(models, x0h.unsqueeze(0).to(device))
                ts = [tp0 + 1, (tp0 + tp1) // 2]
                rows = []
                for t in ts:
                    a = _draw_boxes(fp[rep[t]], anno, t)
                    b = _draw_boxes(rec[rep[t]], anno, t)
                    cv2.putText(a, f'pasted input t={t}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(b, f'x0hat s={sigma}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    rows.append(np.concatenate([a, b], axis=1))
                cv2.imwrite(os.path.join(png_dir, f'{vid}_a{alpha}_s{sigma}.png'),
                            cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
        del lat_paste
    return res


# ---------------------------------------------------------------- 主流程
def select_vids(min_tarr=5, min_det=18):
    vids = []
    for f in sorted(os.listdir(ANNO_DIR)):
        if not f[0].isdigit():
            continue
        d = json.load(open(os.path.join(ANNO_DIR, f)))
        if d['gate_enabled'] and d['t_arrival'] is not None and d['t_arrival'] >= min_tarr \
                and d['n_detected_frames'] >= min_det:
            vids.append(d['vid'])
    return sorted(vids, key=int)


def run(args):
    import torch
    from giga_models.nn import EDMLoss
    dtype, device = torch.bfloat16, 'cuda'
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)
    tf_dir = args.transformer_dir or f'{args.model_dir}/transformer'
    models = P.load_models(tf_dir, f'{args.model_dir}/vae',
                           f'{args.model_dir}/text_encoder', device, dtype)
    loc = None
    sigmas = [float(s) for s in args.sigmas.split(',')]
    vids = select_vids()[args.shard_index::args.num_shards]
    eye_vids = set(vids[:1])                      # 每 shard 1 条人眼
    paste_vids = set(vids[:args.paste_per_shard])
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    os.makedirs(os.path.join(args.out_dir, 'eye_png'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'paste_png'), exist_ok=True)
    print(f'[ghost] shard {args.shard_index}/{args.num_shards}: {len(vids)} vids '
          f'sigmas={sigmas} eye={sorted(eye_vids)} paste={sorted(paste_vids)}', flush=True)

    out = {'per_video': {}, 'paste': []}
    for vid in vids:
        anno = json.load(open(os.path.join(ANNO_DIR, f'{vid}.json')))
        vp = os.path.join(VIDEO_ROOT, f'{vid}.mp4')
        if not os.path.exists(vp):
            continue
        frames = P.sample_frames_like_training(vp, NF, HPIX, WPIX)
        if args.paste_only:
            if vid not in paste_vids:
                continue
            zones = None
        else:
            motion = cell_motion(frames, rep)
            zones = build_zones(anno, motion)
            if not zones['B_pre']:
                print(f'  {vid}: B_pre 空 (静止过滤后), 跳过', flush=True)
                continue
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        fr = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        lat = P._forward_vae(models['vae'], models['lat_mean'], models['lat_std'], fr).float()
        ref = fr.clone(); ref[:, 1:] = 0.0
        ref_lat = P._forward_vae(models['vae'], models['lat_mean'], models['lat_std'], ref).float()
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(anno['prompt'], max_length=512).to(device)
        vrec = {'t_arrival': anno['t_arrival'],
                'zones_n': {k: len(v) for k, v in zones.items()} if zones else {}}
        rec_rt = None
        for sigma in ([] if args.paste_only else sigmas):
            gen = torch.Generator(device=device).manual_seed(int(vid))
            x0h = forward_x0hat(models, lat, ref_lat, emb, sigma, args.fps, device, dtype, edm, gen)
            dev = cell_dev(x0h, lat[0].cpu())
            zm = zone_means(dev, zones)
            # tol 标定: static_far 格值分位数
            sf = np.array([dev[t, y, x] for (t, y, x) in zones['static_far']])
            zm['tol_p50'], zm['tol_p90'], zm['tol_p95'] = [float(np.percentile(sf, q)) for q in (50, 90, 95)] \
                if len(sf) else (float('nan'),) * 3
            vrec[f's{sigma}'] = zm
            if vid in eye_vids and sigma in (1.5, 3.0):
                if rec_rt is None:
                    rec_rt = decode_latent(models, lat)
                rec_hat = decode_latent(models, x0h.unsqueeze(0).to(device))
                save_eye_png(args.out_dir, frames, rec_rt, rec_hat, anno, rep, sigma,
                             os.path.join(args.out_dir, 'eye_png', f'{vid}_s{sigma}.png'))
                del rec_hat
        if not args.paste_only:
            out['per_video'][vid] = vrec
        if vid in paste_vids:
            if loc is None:
                from t4g_gdino import GDinoLocator
                loc = GDinoLocator(device=device)
            pr = run_paste(models, loc, anno, frames, lat, ref_lat, emb, args.fps, device, dtype,
                           edm, vid, os.path.join(args.out_dir, 'paste_png'))
            if pr:
                out['paste'].append(dict(vid=vid, results=pr))
        del lat, ref_lat
        torch.cuda.empty_cache()
        nb = len(zones['B_pre']) if zones else -1
        nf = len(zones['static_far']) if zones else -1
        print(f'  {vid}: t*={anno["t_arrival"]} B_pre={nb} far={nf} done', flush=True)

    p = os.path.join(args.out_dir, f'partial_shard{args.shard_index}.json')
    json.dump(out, open(p, 'w'), indent=1)
    print(f'[ghost] shard {args.shard_index} -> {p}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--sigmas', default='0.3,0.7,1.5,3.0')
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--paste-per-shard', type=int, default=2)
    ap.add_argument('--transformer-dir', default=None,
                    help='transformer 权重覆盖 (如训后 checkpoint/transformer); vae/T5 仍取 model-dir')
    ap.add_argument('--paste-only', action='store_true', help='只跑保留探针 (E10 复测)')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
