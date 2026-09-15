#!/usr/bin/env python3
"""SELFCASE-G1: B1-G1 签名可分性统计 (71 号 G1 gate)。

命题: 真实自发复制病例中, 复制品出现处的前后帧特征自相似 cos(f[t,c], f[t-1,c])
在病例帧段内显著低于背景静止格 (novelty 信号), 即 dup-vs-background 可分。

输入: mine_round0 case bank 中 status=='ok' 的 12 条自发病例
  (case.video=rollout mp4, case.box=[y0,x0,y1,x1] 像素复制框,
   case.d0/case.dend=像素帧段 [d0, dend))。
特征: 与 t4g_change_test 同款 —— P.load_models + P.forward_with_hooks
  (round0_ema_st, sigmas 0.2/0.4, blocks 22..26), 特征网格 24x30x48 (÷16 像素/格)。
统计: 每 (block, sigma):
  dup 格集合在病例帧段内的逐帧自相似值 vs 背景 200 随机格 (不在 dup 框/画面边缘)
  同帧段同款值; AUC 以 1-selfsim 为分数 (低自相似=阳性)。
输出: /data/.../selfcase/g1/g1_signature_report.json + 打印 AUC 汇总表与结论行。
GPU 作业, 单卡单进程。
"""
import argparse
import json
import os

import numpy as np

NF, HPIX, WPIX, T_LAT = 93, 480, 768, 24
H_LAT, W_LAT = 30, 48                       # 480/16, 768/16 (VAE 8x + patch 2x)
REP = np.linspace(0, NF - 1, T_LAT).astype(int)   # latent 帧 -> 代表像素帧


# ====================================================================================
#  纯函数 (无 torch/cv2 依赖, 本机可单测)
# ====================================================================================
def load_ok_cases(report_path, it2v_path):
    """case bank 报告 -> [{case_id, video, prompt, box, d0, dend}] (status=='ok')。"""
    it2v = {int(str(r['request_id']).split('_')[0]): r
            for r in json.load(open(it2v_path))}
    out = []
    for r in json.load(open(report_path))['records']:
        if r.get('status') != 'ok':
            continue
        c = r['case']
        idx = int(os.path.basename(c['video']).split('_')[0])
        out.append(dict(case_id=c['case_id'], video=c['video'],
                        prompt=it2v[idx]['prompt'], box=[int(v) for v in c['box']],
                        d0=int(c['d0']), dend=int(c['dend'])))
    return out


def box_px_to_cells(box, h_lat=H_LAT, w_lat=W_LAT):
    """像素框 (y0,x0,y1,x1) -> latent 格 [(gy,gx)...] (÷16, t4g_detect.px_to_cell 同款)。"""
    y0, x0, y1, x1 = box
    gy0 = int(np.clip(y0 / HPIX * h_lat, 0, h_lat - 1))
    gy1 = int(np.clip(y1 / HPIX * h_lat, 0, h_lat - 1))
    gx0 = int(np.clip(x0 / WPIX * w_lat, 0, w_lat - 1))
    gx1 = int(np.clip(x1 / WPIX * w_lat, 0, w_lat - 1))
    return [(gy, gx) for gy in range(gy0, gy1 + 1) for gx in range(gx0, gx1 + 1)]


def lat_span(d0, dend):
    """像素帧段 [d0,dend) -> latent 帧段 [lt_lo,lt_hi) (selfcase_build 同款公式)。"""
    lt_lo = int(np.searchsorted(REP, d0))
    lt_hi = max(lt_lo + 1, int(np.searchsorted(REP, dend - 1, side='right')))
    return lt_lo, min(lt_hi, T_LAT)


def sample_bg_cells(dup_cells, n_bg=200, edge=2, seed=0, h_lat=H_LAT, w_lat=W_LAT):
    """随机背景格: 不在 dup 框 (含 1 格 margin)、不在画面边缘 (edge 圈)。"""
    dup = set(dup_cells)
    forbid = {(gy + dy, gx + dx) for gy, gx in dup
              for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    cand = [(gy, gx) for gy in range(edge, h_lat - edge)
            for gx in range(edge, w_lat - edge) if (gy, gx) not in forbid]
    rng = np.random.RandomState(seed)
    if len(cand) <= n_bg:
        return cand
    pick = rng.choice(len(cand), size=n_bg, replace=False)
    return [cand[i] for i in pick]


def rank_auc(pos_scores, neg_scores):
    """Mann-Whitney AUC (平均秩处理并列)。pos=阳性(dup)分数, neg=阴性(背景)分数。"""
    pos = np.asarray(pos_scores, np.float64)
    neg = np.asarray(neg_scores, np.float64)
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float('nan')
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


# ====================================================================================
#  GPU 主流程
# ====================================================================================
def run(args):
    import torch
    from giga_models.nn import EDMLoss
    import t4g_probe as P

    dtype, device = torch.bfloat16, 'cuda'
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)
    models = P.load_models(f'{args.model_dir}/transformer', f'{args.model_dir}/vae',
                           f'{args.model_dir}/text_encoder', device, dtype)
    cases = load_ok_cases(args.report, args.it2v)[:args.limit]
    sigmas = [float(s) for s in args.sigmas.split(',')]
    blocks = [f'block{i}' for i in range(args.block_lo, args.block_hi + 1)]
    print(f'[g1] {len(cases)} cases, sigmas={sigmas}, blocks={blocks}', flush=True)

    per_case = {}
    pool = {}          # (block, sigma) -> {'dup': [...], 'bg': [...], 'case_auc': [...]}
    for ci, c in enumerate(cases):
        cid = c['case_id']
        dup_cells = box_px_to_cells(c['box'])
        bg_cells = sample_bg_cells(dup_cells, n_bg=args.n_bg, edge=args.edge,
                                   seed=args.seed + ci)
        lt_lo, lt_hi = lat_span(c['d0'], c['dend'])
        ts = np.arange(max(lt_lo, 1), lt_hi)          # 帧段内可算 t-1 的帧
        if len(ts) == 0:
            print(f'  {cid}: 帧段过短, 跳过', flush=True)
            continue
        frames = P.sample_frames_like_training(c['video'], NF, HPIX, WPIX)
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        fn_norm = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(c['prompt'], max_length=512).to(device)
        dy, dx = np.array([p[0] for p in dup_cells]), np.array([p[1] for p in dup_cells])
        by, bx = np.array([p[0] for p in bg_cells]), np.array([p[1] for p in bg_cells])
        crec = per_case.setdefault(cid, dict(d0=c['d0'], dend=c['dend'],
                                             lt_span=[int(lt_lo), int(lt_hi)],
                                             n_dup_cells=len(dup_cells),
                                             n_bg_cells=len(bg_cells), stats={}))
        for sigma in sigmas:
            store = P.forward_with_hooks(models, fn_norm, emb, sigma, args.fps,
                                         device, dtype, edm)
            for b in blocks:
                if b not in store:
                    continue
                feat = store[b].float()               # (T, H, W, D)
                T, H, W, _ = feat.shape
                if (H, W) != (H_LAT, W_LAT):
                    print(f'  {cid} {b}: 特征网格 {H}x{W} != {H_LAT}x{W_LAT}, 跳过', flush=True)
                    continue
                fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                sim = (fn[1:] * fn[:-1]).sum(-1).numpy()   # (T-1,H,W); sim[t-1]=cos(f[t],f[t-1])
                tt = ts[ts < T] - 1
                dup_vals = sim[np.ix_(tt, dy, dx)].ravel()
                bg_vals = sim[np.ix_(tt, by, bx)].ravel()
                auc = rank_auc(1.0 - dup_vals, 1.0 - bg_vals)   # 低自相似=阳性
                k = (b, sigma)
                pr = pool.setdefault(k, dict(dup=[], bg=[], case_auc=[]))
                pr['dup'].append(dup_vals)
                pr['bg'].append(bg_vals)
                pr['case_auc'].append(auc)
                crec['stats'][f'{b}|{sigma}'] = dict(
                    dup_selfsim_mean=float(dup_vals.mean()),
                    bg_selfsim_mean=float(bg_vals.mean()),
                    n_dup=int(dup_vals.size), n_bg=int(bg_vals.size), auc=auc)
            del store
        print(f'  [{ci + 1}/{len(cases)}] {cid}: dup_cells={len(dup_cells)} '
              f'lt=[{lt_lo},{lt_hi}) done', flush=True)

    # ---- 汇总 ----
    summary = {}
    for (b, s), pr in sorted(pool.items()):
        dup = np.concatenate(pr['dup']); bg = np.concatenate(pr['bg'])
        ca = np.array([a for a in pr['case_auc'] if a == a])
        summary[f'{b}|{s}'] = dict(
            pooled_auc=rank_auc(1.0 - dup, 1.0 - bg),
            mean_case_auc=float(ca.mean()) if ca.size else float('nan'),
            dup_selfsim_mean=float(dup.mean()), bg_selfsim_mean=float(bg.mean()),
            n_cases=len(pr['case_auc']), n_dup_vals=int(dup.size), n_bg_vals=int(bg.size))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'g1_signature_report.json')
    json.dump(dict(args=vars(args), n_cases=len(per_case),
                   per_case=per_case, summary=summary),
              open(out_path, 'w'), indent=1)
    print(f'[g1] -> {out_path}', flush=True)

    print('\n[G1] BLOCK/SIGMA AUC 汇总表 (dup-vs-background, 分数=1-selfsim)', flush=True)
    print(f'{"block":<9}{"sigma":<7}{"pooled_auc":<12}{"mean_case_auc":<15}'
          f'{"dup_selfsim":<13}{"bg_selfsim":<12}{"n_cases":<8}', flush=True)
    best_k, best_v = None, -1.0
    for k, v in sorted(summary.items()):
        b, s = k.split('|')
        print(f'{b:<9}{s:<7}{v["pooled_auc"]:<12.4f}{v["mean_case_auc"]:<15.4f}'
              f'{v["dup_selfsim_mean"]:<13.4f}{v["bg_selfsim_mean"]:<12.4f}'
              f'{v["n_cases"]:<8}', flush=True)
        if v['pooled_auc'] == v['pooled_auc'] and v['pooled_auc'] > best_v:
            best_k, best_v = k, v['pooled_auc']
    if best_k:
        mca = summary[best_k]['mean_case_auc']
        print(f'\n[G1] CONCLUSION: best={best_k} pooled_auc={best_v:.4f} '
              f'mean_case_auc={mca:.4f} '
              f'({"可分 (>=0.8)" if best_v >= 0.8 else "弱可分 (0.7-0.8)" if best_v >= 0.7 else "不可分 (<0.7)"})',
              flush=True)
    print('G1_DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='/data/datasets/gagi/eve_v2_outputs/selfcase/mine_round0/case_bank_report.json')
    ap.add_argument('--it2v', default='/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json')
    ap.add_argument('--model-dir', default='/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--sigmas', default='0.2,0.4')
    ap.add_argument('--block-lo', type=int, default=22)
    ap.add_argument('--block-hi', type=int, default=26)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--n-bg', type=int, default=200)
    ap.add_argument('--edge', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--out-dir', default='/data/datasets/gagi/eve_v2_outputs/selfcase/g1')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
