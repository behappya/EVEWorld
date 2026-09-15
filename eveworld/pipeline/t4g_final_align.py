#!/usr/bin/env python3
"""终局臂在线抠图对账 (72 号 smoke 附加项): 在线 latent 回填目标 vs 离线像素金标准。

命题: A2 的在线擦除目标构造 (prefill_target: 指认格回填出现前帧 latent) 是否
逼近离线 case builder 的像素域修补金标准 (selfcase_build: 病程前帧回填+羽化)?

协议 (40 例 case_bank_all, 病例真掩码, 零模型前向):
  rollout = base(修补版) + 贴回 patch (alpha, 与 T4GSelfCaseTransform 输入端同构)
  Z_roll = VAE(rollout), Z_base = VAE(base)  # 离线金标准的 latent 像
  hit    = 病例区真掩码 (npz lat -> 特征格 ÷2, ceil)
  T_onl  = prefill_target(Z_roll, hit)       # 在线回填目标 (训练同款函数)
  指标 (病例区内): mse(T_onl, Z_base) vs mse(Z_roll, Z_base) (不修补基线)
                   + 逐格通道向量 cos(T_onl, Z_base)
  判据: 中位 mse_ratio = mse_onl/mse_roll < 0.5 且中位 cos > 0.9 -> 对账通过。
GPU 单卡 (只载 VAE); 输出 <out>/align_report.json + 判决行。
"""
import argparse
import json
import os

import numpy as np

CASE_BANK = '/data/datasets/gagi/eve_v2_outputs/selfcase/case_bank_all'
VAE_PATH = '/data/datasets/gagi/giga_world_0_video_pretrain/vae'
OUT_DEFAULT = '/data/datasets/gagi/eve_v2_outputs/t4g_final/align'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default=CASE_BANK)
    ap.add_argument('--vae', default=VAE_PATH)
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--limit', type=int, default=100)
    args = ap.parse_args()

    import torch
    from diffusers.models import AutoencoderKLWan
    from einops import rearrange
    from eveworld.pipeline.t4g_final_trainer import (pixel_prefill, prefill_target)

    dtype, device = torch.bfloat16, 'cuda'
    vae = AutoencoderKLWan.from_pretrained(args.vae).to(device, dtype).eval()
    vae.requires_grad_(False)
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    ls = (1.0 / torch.tensor(vae.config.latents_std)).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    def encode(u8):
        x = torch.from_numpy(u8.astype(np.float32)).div_(127.5).sub_(1.0)
        x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype)      # (1,T,C,H,W)
        with torch.no_grad():
            z = vae.encode(rearrange(x, 'b t c h w -> b c t h w')).latent_dist.sample()
        return ((z - lm) * ls).float()

    files = sorted(f for f in os.listdir(args.bank) if f.endswith('.npz'))[:args.limit]
    rows = []
    for i, f in enumerate(files):
        d = np.load(os.path.join(args.bank, f))
        base, patch, boxes = d['base'], d['patch'], d['boxes']
        d0, dend = [int(v) for v in d['frames']]
        a = float(d['alpha'])
        roll = base.copy()
        for p in range(d0, dend):
            y0, x0, y1, x1 = [int(v) for v in boxes[p]]
            roll[p, y0:y1, x0:x1] = (a * patch + (1 - a) * roll[p, y0:y1, x0:x1]).astype(np.uint8)
        z_base, z_roll = encode(base), encode(roll)
        lt0, lt1, ly0, ly1, lx0, lx1 = [int(v) for v in d['lat']]
        hit = torch.zeros(1, 24, 30, 48, dtype=torch.bool, device=z_roll.device)
        hit[0, lt0:lt1, ly0 // 2:min(30, (ly1 + 1) // 2), lx0 // 2:min(48, (lx1 + 1) // 2)] = True
        tgt, hit_up = prefill_target(z_roll, hit)
        tgt0 = z_roll[:, :, :1].expand_as(z_roll)         # 对照: 全回填帧0 (无时间混叠)
        # ---- 像素精确版 (修复指令): decode(z_roll) -> 像素回填+羽化 -> 重编码 ----
        with torch.no_grad():
            raw = (z_roll.to(dtype) / ls) + lm
            pix_hat = vae.decode(raw).sample.float().clamp(-1, 1)     # (1,3,NF,H,W)
            pix_fill, _ = pixel_prefill(pix_hat, hit)
            z2 = vae.encode(pix_fill.clamp(-1, 1).to(dtype)).latent_dist.sample()
            tgt_px = ((z2 - lm) * ls).float()
        zdim = z_roll.shape[1]
        denom = hit_up.sum() * zdim + 1e-8
        reg = hit_up[0, 0] > 0                            # (T,60,96)

        def _m(t):
            return float((((t - z_base) ** 2) * hit_up).sum() / denom)

        def _c(t):
            return float(torch.nn.functional.cosine_similarity(
                t[0, :, reg].T, z_base[0, :, reg].T, dim=1).mean())

        mse_onl, mse_f0, mse_rll, mse_px = _m(tgt), _m(tgt0), _m(z_roll), _m(tgt_px)
        cos, cos_f0, cos_px = _c(tgt), _c(tgt0), _c(tgt_px)
        rows.append(dict(case=f, mse_online=mse_onl, mse_frame0=mse_f0,
                         mse_pixel=mse_px, mse_rollout=mse_rll,
                         mse_ratio=mse_onl / max(mse_rll, 1e-8),
                         mse_ratio_f0=mse_f0 / max(mse_rll, 1e-8),
                         mse_ratio_px=mse_px / max(mse_rll, 1e-8),
                         cos_online=cos, cos_frame0=cos_f0, cos_pixel=cos_px,
                         n_hit_lat=int(hit_up.sum())))
        print(f'  [{i + 1}/{len(files)}] {f}: ratio={rows[-1]["mse_ratio"]:.3f} '
              f'ratio_px={rows[-1]["mse_ratio_px"]:.3f} cos={cos:.4f} '
              f'cos_px={cos_px:.4f}', flush=True)

    med = {k: float(np.median([r[k] for r in rows]))
           for k in ('mse_ratio', 'mse_ratio_f0', 'mse_ratio_px',
                     'cos_online', 'cos_frame0', 'cos_pixel')}
    ok = med['mse_ratio'] < 0.5 and med['cos_online'] > 0.9
    ok_px = med['mse_ratio_px'] < 0.3 and med['cos_pixel'] > 0.9   # 修复指令判据
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(dict(n_cases=len(rows), medians=med, passed_prefill=bool(ok),
                   passed_pixel=bool(ok_px), rows=rows),
              open(os.path.join(args.out_dir, 'align_report.json'), 'w'), indent=1)
    print(f'\n[ALIGN] n={len(rows)} | latent-prefill: ratio={med["mse_ratio"]:.3f} '
          f'cos={med["cos_online"]:.4f} -> {"PASS" if ok else "FAIL"}(判据<0.5,>0.9) | '
          f'**pixel-precise: ratio={med["mse_ratio_px"]:.3f} '
          f'cos={med["cos_pixel"]:.4f} -> {"PASS" if ok_px else "FAIL"}** '
          f'(判据<0.3,>0.9) | frame0 对照: ratio={med["mse_ratio_f0"]:.3f} '
          f'cos={med["cos_frame0"]:.4f}', flush=True)
    print('ALIGN_DONE', flush=True)


if __name__ == '__main__':
    main()
