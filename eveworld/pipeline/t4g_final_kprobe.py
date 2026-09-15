#!/usr/bin/env python3
"""终局臂 K 扫描探针 (72 号 S4 升级): 实测定 A2 滚雪球轮数 K, 不拍脑袋。

命题: pretrain 底座在训练式前向 (teacher-forced 起点 + 大步幅 σ 自我喂养) 里滚
K 轮后, ICH-D 固化检测器能否在末轮 x̂0 上检出自发涌现 (M 命中)? K 多大才稳定?

协议:
  条件 = 病例高发 12 条件 (case_bank_all vid 按病例数降序: 3 香蕉族 x13, 66, 2,
    32, 41, 6, 62, 39, 46, 67, 74, 79), GT latent 起点 + 帧0 ref 条件 (I2V 契约)。
  滚动 = K∈{2,4,6,8} 轮, **大步幅 σ 调度**: geomspace(3.0 -> 0.3) K 轮均匀覆盖
    全轨迹高→低 (4 大步≈30 小步的轨迹覆盖, K 能小的关键设计); 每轮
    x_t = x̂0_prev + σ_k·ε -> 前向 -> x̂0 (全 no_grad, 自我喂养)。
  判据 = 末轮前向特征过 ICH-D 固化头 -> M (24,30,48);
    命中 = 帧1+ 中 M>0.5 的格数 >= min_cells(4);
    GDINO 复核 = 末轮 x̂0 解码, 中后段 3 帧数 mover 实例 (>1 = 可见复制品),
    证据帧存 png。
输出: <out>/kprobe_report.json + K-命中率曲线打印 + 判决行
  (选"命中率首次稳定"的最小 K; K=8 仍无信号 -> 报停, 方案重设计)。
GPU 单卡; `--selftest` 纯 CPU 干跑 (σ 调度/命中判据/条件选择)。
"""
import argparse
import json
import os

import numpy as np

OUT_DEFAULT = '/data/datasets/gagi/eve_v2_outputs/t4g_final/kprobe'
PRETRAIN = '/data/datasets/gagi/giga_world_0_video_pretrain'
IT2V = '/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json'
CASE_BANK_ALL = '/data/datasets/gagi/eve_v2_outputs/selfcase/case_bank_all'
K_LIST = (2, 4, 6, 8)


# ====================================================================================
#  纯函数
# ====================================================================================
def sigma_schedule(k, s_hi=3.0, s_lo=0.3):
    """K 轮大步幅 σ: 几何均匀覆盖高->低。"""
    return [float(s) for s in np.geomspace(s_hi, s_lo, k)]


def top_case_vids(bank_dir=CASE_BANK_ALL, n=12):
    """病例库 vid 按病例数降序取前 n (确定性: 数量降序, vid 升序破平)。"""
    cnt = {}
    for f in sorted(os.listdir(bank_dir)):
        if f.endswith('.npz'):
            cnt[f.split('__')[0]] = cnt.get(f.split('__')[0], 0) + 1
    order = sorted(cnt, key=lambda v: (-cnt[v], int(v)))
    return order[:n], cnt


def hit_verdict(m, thr=0.5, min_cells=4):
    """M (T,H,W) -> (命中?, 命中格数, m_mean, m_max); 帧0 恒 ref 排除。"""
    mm = m[1:]
    n_hit = int((mm > thr).sum())
    return n_hit >= min_cells, n_hit, float(mm.mean()), float(mm.max())


def pick_k(per_k_hit_rate, k_list=K_LIST, stable_tol=0.999):
    """选"命中率首次稳定"的最小 K: 首个 K 使 hit_rate[K] >= max(后续所有 K)·tol
    且 hit_rate[K] > 0。全为 0 -> None (报停)。"""
    ks = list(k_list)
    for i, k in enumerate(ks):
        r = per_k_hit_rate[k]
        if r <= 0:
            continue
        if all(r >= per_k_hit_rate[k2] * stable_tol for k2 in ks[i + 1:]):
            return k
    for k in ks:                     # 兜底: 有信号但未"稳定"档 -> 取命中率最高的最小 K
        if per_k_hit_rate[k] == max(per_k_hit_rate.values()) and per_k_hit_rate[k] > 0:
            return k
    return None


def selftest():
    print('[kprobe selftest] start')
    s4 = sigma_schedule(4)
    assert len(s4) == 4 and abs(s4[0] - 3.0) < 1e-6 and abs(s4[-1] - 0.3) < 1e-6
    assert all(s4[i] > s4[i + 1] for i in range(3)), '必须高->低'
    s2, s8 = sigma_schedule(2), sigma_schedule(8)
    assert (s2[0], s2[-1]) == (s4[0], s4[-1]) == (s8[0], s8[-1]), '各 K 覆盖同轨迹'
    m = np.zeros((24, 30, 48), np.float32)
    assert hit_verdict(m)[0] is False
    m[0, :, :] = 1.0                                     # 帧0 应被排除
    assert hit_verdict(m)[0] is False
    m[5, 10:12, 20:22] = 0.9                             # 4 格 -> 命中
    ok, n, _, mx = hit_verdict(m)
    assert ok and n == 4 and mx > 0.8
    vids, cnt = top_case_vids()
    print(f'[kprobe selftest] top12 vids={vids} (counts={[cnt[v] for v in vids]})')
    assert vids[0] == '3' and len(vids) == 12
    assert pick_k({2: 0.0, 4: 0.5, 6: 0.5, 8: 0.5}) == 4
    assert pick_k({2: 0.0, 4: 0.25, 6: 0.5, 8: 0.5}) == 6
    assert pick_k({2: 0.0, 4: 0.0, 6: 0.0, 8: 0.0}) is None
    print('[kprobe selftest] SELFTEST_OK')


# ====================================================================================
#  GPU 主流程
# ====================================================================================
def run(args):
    import torch
    from giga_models.nn import EDMLoss
    import t4g_probe as P
    from t4g_gdino import GDinoLocator, parse_objects
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from eveworld.pipeline.t4g_ich_d_trainer import ICHDModule, NOVELTY_BLOCKS, CIC_BLOCK

    dtype, device = torch.bfloat16, 'cuda'
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)
    models = P.load_models(f'{args.model_dir}/transformer', f'{args.model_dir}/vae',
                           f'{args.model_dir}/text_encoder', device, dtype)
    tfm, vae = models['transformer'], models['vae']
    lat_mean, lat_std = models['lat_mean'], models['lat_std']
    ich = ICHDModule(channels=tfm.config.model_channels, weights_path=args.weights)
    ich.to(device, dtype)
    for nm in ('lr_w', 'lr_b', 'lr_mu', 'lr_sd'):
        getattr(ich, nm).data = getattr(ich, nm).data.float()
    ich.eval()
    for name in NOVELTY_BLOCKS + (CIC_BLOCK,):           # 探针只 observe, 不注入擦除
        tfm.blocks[name].register_forward_hook(
            lambda _m, _i, out, _n=name: ich.observe(_n, out))
    gdino = GDinoLocator(device=device)

    it2v = {int(str(r['request_id']).split('_')[0]): r for r in json.load(open(args.it2v))}
    vids, cnt = top_case_vids(n=args.n_conds)
    print(f'[kprobe] conds={vids} (case counts={[cnt[v] for v in vids]}) '
          f'K_list={list(K_LIST)} sigma={sigma_schedule(4)}', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, 'evidence_png')
    os.makedirs(png_dir, exist_ok=True)

    def rollout_step(x0, ref_latents, ref_mask, emb, sigma, grad=False):
        """一轮自我喂养: x0 -> renoise(σ) -> 前向 -> x̂0 (帧0 ref 恒定)。"""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            input_latents, c_noise = edm.add_noise(x0.float(), sigma=sigma)
            # dtype 纪律: fp32 数学 -> 网络输入前统一 bf16 (fp32/bf16 混拼会崩 cat/linear)
            rm = ref_mask.float()
            input_latents = rm * ref_latents.float() + (1 - rm) * input_latents
            in_masks = rm.repeat(1, 1, 1, input_latents.shape[-2],
                                 input_latents.shape[-1])
            x_in = torch.cat([input_latents, in_masks], dim=1).to(dtype)
            ts = c_noise.float().view(1, 1, 1, 1, 1).expand(1, -1, x0.shape[2], -1, -1)
            t_cond = torch.tensor([1e-4], device=device).view(1, 1, 1, 1, 1)
            ts = rm * (t_cond / (t_cond + 1)) + (1 - rm) * ts
            padding_mask = torch.zeros((1, 1, 480, 768), dtype=dtype, device=device)
            pred = tfm(x=x_in, timesteps=ts.to(dtype), crossattn_emb=emb.to(dtype),
                       padding_mask=padding_mask, fps=args.fps)
            x0_new = edm.denoise(pred.float())
            x0_new = ref_mask.float() * ref_latents.float() + \
                (1 - ref_mask.float()) * x0_new
        return x0_new

    def decode_frames(x0, idxs):
        """x̂0 归一化 latent -> 像素帧 uint8 dict{pix_idx: HxWx3}。"""
        with torch.no_grad():
            lat = (x0.to(dtype) / lat_std) + lat_mean
            vid_t = vae.decode(lat.to(dtype)).sample[0]  # (C,T,H,W) [-1,1]
        vid_t = ((vid_t.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        rep = np.linspace(0, vid_t.shape[1] - 1, 93).astype(int)
        inv = {}
        for p in idxs:                                    # 像素帧 -> 最近 latent 帧
            lt = int(np.abs(rep - p).argmin())
            inv[p] = vid_t[:, lt].permute(1, 2, 0).cpu().numpy()
        return inv

    def count_mover(frame_rgb, mover):
        from PIL import Image
        q = mover.lower().strip().rstrip('.') + '.'
        img = Image.fromarray(frame_rgb)
        inputs = gdino.proc(images=img, text=q, return_tensors='pt').to(device)
        with torch.no_grad():
            out = gdino.model(**inputs)
        res = gdino.proc.post_process_grounded_object_detection(
            out, inputs['input_ids'], threshold=0.25, text_threshold=0.2,
            target_sizes=[img.size[::-1]])[0]
        return len(res['boxes'])

    report = dict(conds=vids, k_list=list(K_LIST), sigma_hi=3.0, sigma_lo=0.3,
                  per_cond={}, per_k={})
    for ci, v in enumerate(vids):
        meta = it2v[int(v)]
        frames = P.sample_frames_like_training(meta['source_video'], 93, 480, 768)
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        fn = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(meta['prompt'],
                                                        max_length=512).to(device)
            lat_gt = P._forward_vae(vae, lat_mean, lat_std, fn).float()
            ref_fr = fn.clone(); ref_fr[:, 1:] = 0.0
            ref_lat = P._forward_vae(vae, lat_mean, lat_std, ref_fr).float()
        ref_mask = torch.zeros((1, 1, lat_gt.shape[2], 1, 1), device=device)
        ref_mask[:, :, 0] = 1.0
        mover = parse_objects(meta['prompt'])['mover']
        # ---- GDINO 基线: GT 像素帧实例数 (场景可能天然多实例, 复制判据必须做差) ----
        n_inst_gt = 0
        for p in (40, 65, 90):
            n_inst_gt = max(n_inst_gt, count_mover(frames[p], mover))
        crec = {}
        for K in (0,) + tuple(K_LIST):                    # K=0 = teacher-forced 对照
            sigmas = sigma_schedule(K) if K > 0 else [args.m_base_sigma]
            torch.manual_seed(args.seed + ci)             # 各 K 同起始噪声流
            x0 = lat_gt.clone()
            for k, s in enumerate(sigmas):
                ich.reset()
                sig = torch.tensor([[s]], device=device, dtype=torch.float32)
                x0 = rollout_step(x0, ref_lat, ref_mask, emb, sig)
                if K == 0:
                    x0 = lat_gt.clone()                   # 对照: 不滚动, 只测 M 底噪
            with torch.no_grad():
                m = ich.detect()[0].float().cpu().numpy() # (24,30,48) 末轮特征
            mm = m[1:]
            qs = {q: float(np.quantile(mm, q)) for q in (0.5, 0.9, 0.99)}
            cells09 = int((mm > 0.9).sum())
            hit, n_hit, m_mean, m_max = hit_verdict(m, args.m_thr, args.min_cells)
            if K == 0:
                n_inst = n_inst_gt                        # 对照档: GT 帧基线
            else:
                pix = decode_frames(x0, (40, 65, 90))
                n_inst = 0
                for p, im in pix.items():
                    n_inst = max(n_inst, count_mover(im, mover))
                    if K in (2, max(K_LIST)):
                        import imageio
                        imageio.imwrite(os.path.join(png_dir, f'v{v}_K{K}_f{p}.png'), im)
            crec[K] = dict(hit=bool(hit), n_hit_cells=n_hit, cells_gt09=cells09,
                           m_mean=m_mean, m_p50=qs[0.5], m_p90=qs[0.9],
                           m_p99=qs[0.99], m_max=m_max, gdino_max_inst=int(n_inst),
                           gdino_inst_gt=int(n_inst_gt))
            print(f'  [{ci + 1}/{len(vids)}] v{v} K={K}: cells@.5={n_hit} '
                  f'cells@.9={cells09} m_p90={qs[0.9]:.3f} p99={qs[0.99]:.3f} '
                  f'gdino_inst={n_inst} (gt={n_inst_gt}, mover={mover})', flush=True)
        report['per_cond'][v] = crec

    # ---- 汇总: 增量判据 (相对 K=0 teacher-forced 底噪对照) ----
    print('\n[KPROBE] K-涌现曲线 (Δcells@.9 = 相对 K=0 对照的 M>0.9 格增量; '
          'gdino_dup = 实例数超 GT 基线)', flush=True)
    print(f'{"K":<4}{"emergent_rate":<15}{"gdino_dup_rate":<16}{"mean_dcells09":<15}'
          f'{"mean_cells09":<14}{"mean_p99":<10}', flush=True)
    per_k_rate = {}
    for K in (0,) + tuple(K_LIST):
        rows = [report['per_cond'][v][K] for v in vids]
        base = [report['per_cond'][v][0]['cells_gt09'] for v in vids]
        dcells = [r['cells_gt09'] - b for r, b in zip(rows, base)]
        gdino_hits = [r['gdino_max_inst'] > r['gdino_inst_gt'] for r in rows]
        emergent = [(d >= args.min_cells) or g for d, g in zip(dcells, gdino_hits)]
        er = float(np.mean(emergent))
        gr = float(np.mean(gdino_hits))
        if K > 0:
            per_k_rate[K] = er
        report['per_k'][K] = dict(
            emergent_rate=er, gdino_dup_rate=gr,
            mean_dcells09=float(np.mean(dcells)),
            mean_cells09=float(np.mean([r['cells_gt09'] for r in rows])),
            mean_p99=float(np.mean([r['m_p99'] for r in rows])))
        rk = report['per_k'][K]
        print(f'{K:<4}{er:<15.3f}{gr:<16.3f}{rk["mean_dcells09"]:<15.1f}'
              f'{rk["mean_cells09"]:<14.1f}{rk["mean_p99"]:<10.3f}', flush=True)
    k_star = pick_k(per_k_rate)
    report['k_star'] = k_star
    json.dump(report, open(os.path.join(args.out_dir, 'kprobe_report.json'), 'w'),
              indent=1)
    if k_star is None:
        print('[KPROBE] VERDICT: K=8 仍无涌现增量信号 (相对 teacher-forced 对照) '
              '-> 停, 滚雪球在训练前向造不出现场, 方案需重设计 (勿硬跑)', flush=True)
    else:
        print(f'[KPROBE] VERDICT: K*={k_star} (涌现率首次稳定的最小 K; '
              f'曲线 {[round(per_k_rate[k], 3) for k in K_LIST]}; '
              f'K=0 底噪 cells@.9={report["per_k"][0]["mean_cells09"]:.0f})', flush=True)
    print('KPROBE_DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=PRETRAIN)
    ap.add_argument('--weights', default='/data/datasets/gagi/eve_v2_outputs/selfcase/cic_match/ich_d_frozen_lr.npz')
    ap.add_argument('--it2v', default=IT2V)
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--n-conds', type=int, default=12)
    ap.add_argument('--m-thr', type=float, default=0.5)
    ap.add_argument('--m-base-sigma', type=float, default=0.3,
                    help='K=0 对照档的前向 sigma (与滚雪球末轮同口径)')
    ap.add_argument('--min-cells', type=int, default=4)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run(args)


if __name__ == '__main__':
    main()
