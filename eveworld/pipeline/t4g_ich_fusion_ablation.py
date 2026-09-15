#!/usr/bin/env python3
"""ICH 融合消融 (71 号 B1, ICH 训练第 0 步): 同一 LOO 协议对比层融合宽度。

对比 (与 selfcase_g1p 完全同协议: 34 例, 逐格 4 维特征, LR, leave-one-case-out):
  1L_block16        : block16 单层            (4  维)
  3L_block10_12_16  : block10+12+16 拼接      (12 维)
  10L_block8_26     : blocks 8..26 隔层全拼   (40 维)

注意: g1p 作业只落了聚合报告 (g1p_report.json), 逐格特征没有缓存 —— 本脚本先在
GPU 上按 g1p 同款采样 (同 seed -> 同格/同窗, 数字可与 g1p 报告逐单层对账) 重抽一遍
并落盘 g1p_cell_features.npz; 缓存存在时纯 CPU 融合 (几分钟)。

输出: /data/.../selfcase/g1p/fusion_ablation.json + 打印三组 LOO-AUC 榜单。
`--selftest` 为纯 numpy 单测; `--fuse-only` 只跑 CPU 融合 (要求缓存已存在)。
"""
import argparse
import json
import os

import numpy as np

import selfcase_g1 as G1
import selfcase_g1p as G1P

G1P_DIR = '/data/datasets/gagi/eve_v2_outputs/selfcase/g1p'
CACHE_DEFAULT = os.path.join(G1P_DIR, 'g1p_cell_features.npz')
OUT_DEFAULT = os.path.join(G1P_DIR, 'fusion_ablation.json')

FUSIONS = {
    '1L_block16': ['block16'],
    '3L_block10_12_16': ['block10', 'block12', 'block16'],
    '10L_block8_26': [f'block{i}' for i in range(8, 27, 2)],
}


# ====================================================================================
#  GPU 抽特征 -> 逐格特征缓存 (g1p.run 同款采样, 只是把 X/y/uid 落盘)
# ====================================================================================
def extract(args):
    import torch
    from giga_models.nn import EDMLoss
    import t4g_probe as P

    dtype, device = torch.bfloat16, 'cuda'
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)
    models = P.load_models(f'{args.model_dir}/transformer', f'{args.model_dir}/vae',
                           f'{args.model_dir}/text_encoder', device, dtype)
    cases = G1P.load_all_cases(args.pools.split(','), args.it2v)[:args.limit]
    sigmas = [float(s) for s in args.sigmas.split(',')]
    blocks = FUSIONS['10L_block8_26']
    print(f'[fusion] extract: {len(cases)} cases, sigmas={sigmas}, blocks={blocks}',
          flush=True)

    rows = {(b, s): [] for b in blocks for s in sigmas}
    y_rows, uid_rows = [], []
    for ci, c in enumerate(cases):
        uid = c['uid']
        # ---- 与 selfcase_g1p.run 完全同款的格/窗采样 (rng 调用顺序也一致) ----
        dup_all = G1.box_px_to_cells(c['box'])
        rngc = np.random.RandomState(args.seed + 1000 + ci)
        dup_cells = dup_all
        if len(dup_cells) > args.n_dup_max:
            pick = rngc.choice(len(dup_cells), size=args.n_dup_max, replace=False)
            dup_cells = [dup_cells[i] for i in pick]
        bg_cells = G1.sample_bg_cells(dup_all, n_bg=args.n_bg, edge=args.edge,
                                      seed=args.seed + ci)
        lt_lo, lt_hi = G1.lat_span(c['d0'], c['dend'])
        wlo, whi = G1P.dup_window(lt_lo, args.win)
        dwins = [(wlo, whi)] * len(dup_cells)
        bwins = G1P.bg_windows(len(bg_cells), args.win, rng=rngc)

        frames = P.sample_frames_like_training(c['video'], G1.NF, G1.HPIX, G1.WPIX)
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        fn_norm = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(c['prompt'], max_length=512).to(device)
        for sigma in sigmas:
            store = P.forward_with_hooks(models, fn_norm, emb, sigma, args.fps,
                                         device, dtype, edm)
            for b in blocks:
                assert b in store, f'{uid}: 缺 {b}'
                feat = store[b].float().numpy()
                _, H, W, _ = feat.shape
                assert (H, W) == (G1.H_LAT, G1.W_LAT), (uid, b, H, W)
                sim_prev, nb_max, sim0 = G1P.selfsim_maps(feat)
                Xd = G1P.cell_features(sim_prev, nb_max, sim0, dup_cells, dwins)
                Xb = G1P.cell_features(sim_prev, nb_max, sim0, bg_cells, bwins)
                rows[(b, sigma)].append(np.vstack([Xd, Xb]))
            del store
        y_rows.append(np.r_[np.ones(len(dup_cells), int), np.zeros(len(bg_cells), int)])
        uid_rows.extend([uid] * (len(dup_cells) + len(bg_cells)))
        print(f'  [{ci + 1}/{len(cases)}] {uid}: dup={len(dup_cells)} bg={len(bg_cells)} '
              f'win=[{wlo},{whi}) done', flush=True)

    y = np.concatenate(y_rows)
    payload = dict(y=y, uid=np.array(uid_rows),
                   meta=np.array(json.dumps(dict(
                       pools=args.pools, sigmas=sigmas, blocks=blocks, seed=args.seed,
                       n_cases=len(cases), n_rows=int(len(y)))), dtype=object))
    for (b, s), xs in rows.items():
        arr = np.vstack(xs)
        assert len(arr) == len(y), (b, s, arr.shape, y.shape)
        payload[f'X_{b}_{s}'] = arr
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez_compressed(args.cache, **payload)
    print(f'[fusion] cell feature cache -> {args.cache} ({len(y)} rows)', flush=True)


# ====================================================================================
#  CPU 融合 LOO (纯 numpy/sklearn)
# ====================================================================================
def fuse_loo(cache_path, sigmas, out_path, force_numpy_lr=False):
    d = np.load(cache_path, allow_pickle=True)
    y, uid = d['y'], d['uid']
    backend, fit_predict = G1P.make_lr_backend(force_numpy=force_numpy_lr)
    print(f'[fusion] {len(y)} rows ({int(y.sum())} dup / {int((1 - y).sum())} bg), '
          f'{len(set(uid.tolist()))} cases, probe backend = {backend}', flush=True)

    results = {}
    for sigma in sigmas:
        for name, blocks in FUSIONS.items():
            X = np.hstack([d[f'X_{b}_{sigma}'] for b in blocks])
            loo_auc, mca, nf = G1P.loo_probe(X, y, uid, fit_predict)
            results[f'{name}|{sigma}'] = dict(
                loo_auc=loo_auc, mean_case_auc=mca, n_folds=nf,
                n_dims=int(X.shape[1]), blocks=blocks, backend=backend)
            print(f'  [loo] {name}|{sigma}: dims={X.shape[1]} loo_auc={loo_auc:.4f} '
                  f'mean_case={mca:.4f} folds={nf}', flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(dict(cache=cache_path, sigmas=sigmas, results=results),
              open(out_path, 'w'), indent=1)
    print(f'[fusion] -> {out_path}', flush=True)

    # ---- 榜单 (按 LOO-AUC 降序) ----
    print('\n[FUSION] 融合消融榜单 (同 g1p LOO 协议: 34 例逐例留出, LR)', flush=True)
    print(f'{"rank":<6}{"config":<20}{"sigma":<7}{"dims":<6}{"loo_auc":<10}'
          f'{"mean_case_auc":<15}{"n_folds":<8}', flush=True)
    rank = sorted(results.items(), key=lambda kv: -kv[1]['loo_auc'])
    for i, (k, v) in enumerate(rank):
        name, s = k.rsplit('|', 1)
        print(f'{i + 1:<6}{name:<20}{s:<7}{v["n_dims"]:<6}{v["loo_auc"]:<10.4f}'
              f'{v["mean_case_auc"]:<15.4f}{v["n_folds"]:<8}', flush=True)

    # ---- 结论行: 每 sigma 一行 1L vs 3L vs 10L ----
    for sigma in sigmas:
        r1 = results[f'1L_block16|{sigma}']['loo_auc']
        r3 = results[f'3L_block10_12_16|{sigma}']['loo_auc']
        r10 = results[f'10L_block8_26|{sigma}']['loo_auc']
        best = max([('1L', r1), ('3L', r3), ('10L', r10)], key=lambda t: t[1])
        print(f'[FUSION] CONCLUSION sigma={sigma}: 1L={r1:.4f} 3L={r3:.4f} '
              f'10L={r10:.4f} | best={best[0]} ({G1P.judge(best[1])})', flush=True)
    print('FUSION_DONE', flush=True)
    return results


# ====================================================================================
#  selftest: 合成多层特征 (每层含独立噪声通道), 纯 numpy 干跑全链路
# ====================================================================================
def selftest():
    print('[fusion selftest] start')
    rng = np.random.RandomState(0)
    n_case, n_pos, n_neg = 6, 40, 80
    y_rows, uid_rows = [], []
    per_block = {b: [] for bl in FUSIONS.values() for b in bl}
    for ci in range(n_case):
        y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
        signal = y * 1.0 + rng.randn(len(y)) * 0.4        # 共享判别信号
        for b in per_block:
            noise = rng.randn(len(y), 4) * 1.0            # 每层独立噪声
            x = noise.copy()
            x[:, 0] -= signal                              # f1 = min selfsim 低 = 阳性
            per_block[b].append(x)
        y_rows.append(y)
        uid_rows.extend([f'case{ci}'] * len(y))
    y = np.concatenate(y_rows)
    uid = np.array(uid_rows)
    _, fit_predict = G1P.make_lr_backend(force_numpy=True)
    aucs = {}
    for name, blocks in FUSIONS.items():
        X = np.hstack([np.vstack(per_block[b]) for b in blocks])
        assert X.shape == (len(y), 4 * len(blocks)), X.shape
        auc, _, nf = G1P.loo_probe(X, y, uid, fit_predict)
        aucs[name] = auc
        assert nf == n_case
        print(f'[fusion selftest] {name}: dims={X.shape[1]} loo_auc={auc:.4f}')
    assert all(a > 0.7 for a in aucs.values()), aucs
    # 独立噪声下融合应不差于单层 (留噪声余量)
    assert aucs['3L_block10_12_16'] >= aucs['1L_block16'] - 0.03, aucs
    print('[fusion selftest] SELFTEST_OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pools', default=G1P.DEFAULT_POOLS)
    ap.add_argument('--it2v', default='/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json')
    ap.add_argument('--model-dir', default='/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--sigmas', default='0.2,0.4')
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--win', type=int, default=3)
    ap.add_argument('--n-bg', type=int, default=200)
    ap.add_argument('--n-dup-max', type=int, default=200)
    ap.add_argument('--edge', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--cache', default=CACHE_DEFAULT)
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--force-numpy-lr', action='store_true')
    ap.add_argument('--fuse-only', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not os.path.exists(args.cache):
        assert not args.fuse_only, f'--fuse-only 但缓存不存在: {args.cache}'
        extract(args)
    else:
        print(f'[fusion] cache exists, skip extract: {args.cache}', flush=True)
    fuse_loo(args.cache, [float(s) for s in args.sigmas.split(',')], args.out,
             force_numpy_lr=args.force_numpy_lr)


if __name__ == '__main__':
    main()
