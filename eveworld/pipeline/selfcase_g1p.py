#!/usr/bin/env python3
"""SELFCASE-G1P: B1-G1' 瞬态签名 + 可训探针 (71 号 G1 路线升级版)。

背景: selfcase_g1 (12 例, blocks 22..26) 用病例帧段内全帧平均口径, AUC 仅 0.56-0.62;
诊断: 复制品出现后长期静止 (自相似回高), 判别信息集中在"出现时刻"几帧
(E7 先例: 出现窗最低自相似 0.36 vs 静止地板 0.80), 被全段平均稀释。

升级点 (G1'):
1. 病例库扩到 34 例, 四池: mine_round0 / mine_gr1_2b / mine_s150 / mine_wmapA_pre_s250;
2. 特征层扩到 blocks 8..26 隔层取样 (8,10,...,26), sigmas 0.2/0.4;
3. 瞬态统计: 复制格取 latent 帧窗 [lt_lo-1, lt_lo+2) 内逐格【最低】前后帧自相似,
   背景格取全时段随机 3 帧窗的同款最低值 (配对口径, 各≤200 格);
   AUC 分数 = 1 - min_selfsim (低自相似=阳性);
4. 可训探针: 每 (block, sigma) 逐格 4 维特征
     [窗内最低同位自相似, 窗内平均同位自相似,
      窗内最低 3x3 邻域最大匹配值, 窗内平均与帧 0 同位相似度]
   (复制格用瞬态窗, 背景格用随机窗), LogisticRegression 做 leave-one-case-out
   (34 例逐例留出) 交叉验证, 报告 pooled LOO-AUC;
   sklearn 缺失时退回手写 numpy 梯度下降逻辑回归 (特征仅 4 维)。

输出: /data/.../selfcase/g1p/g1p_report.json
  + 打印两张榜单 (瞬态 AUC 榜 / 可训探针 LOO-AUC 榜, 按 AUC 降序)
  + 结论行: top-2 层与判决 (>=0.8 可建 / 0.7-0.8 弱可分 / <0.7 止损)。
GPU 作业, 单卡单进程; `--selftest` 为纯 numpy 单测 (无 torch/GPU, 提交前干跑)。
"""
import argparse
import json
import os

import numpy as np

import selfcase_g1 as G1
from selfcase_g1 import NF, HPIX, WPIX, T_LAT, H_LAT, W_LAT

POOL_ROOT = '/data/datasets/gagi/eve_v2_outputs/selfcase'
DEFAULT_POOLS = 'mine_round0,mine_gr1_2b,mine_s150,mine_wmapA_pre_s250'


# ====================================================================================
#  纯函数 (无 torch 依赖, --selftest 可本机单测); 格映射/AUC 复用 selfcase_g1
# ====================================================================================
def load_all_cases(pool_names, it2v_path, pool_root=POOL_ROOT):
    """多池 case bank -> [{uid, pool, case_id, video, prompt, box, d0, dend}]。"""
    out = []
    for pool in pool_names:
        report = os.path.join(pool_root, pool, 'case_bank_report.json')
        for c in G1.load_ok_cases(report, it2v_path):
            c = dict(c, pool=pool, uid=f'{pool}:{c["case_id"]}')
            out.append(c)
    return out


def dup_window(lt_lo, win=3, t_lat=T_LAT):
    """复制出现瞬态窗 [lt_lo-1, lt_lo-1+win), 裁到可算 t-1 自相似的 [1, t_lat)。"""
    lo = max(1, min(lt_lo - 1, t_lat - 1))
    hi = max(lo + 1, min(lo + win, t_lat))
    return lo, hi


def bg_windows(n, win=3, t_lat=T_LAT, rng=None):
    """背景格全时段随机 win 帧窗: 起点 t0 ~ U[1, t_lat-win], 每格独立, 窗 [t0, t0+win)。"""
    rng = rng if rng is not None else np.random.RandomState(0)
    t0 = rng.randint(1, t_lat - win + 1, size=n)
    return [(int(t), int(t + win)) for t in t0]


def selfsim_maps(feat):
    """feat: np float32 (T,H,W,D) -> (sim_prev, nb_max, sim0)。
    sim_prev[t-1,y,x] = cos(f[t,y,x], f[t-1,y,x])           (T-1,H,W)
    nb_max[t-1,y,x]   = max_{|dy|,|dx|<=1} cos(f[t,y,x], f[t-1,y+dy,x+dx])  (T-1,H,W)
    sim0[t,y,x]       = cos(f[t,y,x], f[0,y,x])             (T,H,W)
    """
    f = np.asarray(feat, np.float32)
    fn = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)
    sim_prev = np.einsum('thwd,thwd->thw', fn[1:], fn[:-1])
    sim0 = np.einsum('thwd,hwd->thw', fn, fn[0])
    _, H, W, _ = fn.shape
    cur, prev = fn[1:], fn[:-1]
    nb = np.full(sim_prev.shape, -2.0, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cy = slice(max(0, -dy), H - max(0, dy))
            cx = slice(max(0, -dx), W - max(0, dx))
            py = slice(max(0, dy), H - max(0, -dy))
            px = slice(max(0, dx), W - max(0, -dx))
            s = np.einsum('thwd,thwd->thw', cur[:, cy, cx], prev[:, py, px])
            np.maximum(nb[:, cy, cx], s, out=nb[:, cy, cx])
    return sim_prev, nb, sim0


def cell_features(sim_prev, nb_max, sim0, cells, windows):
    """逐格 4 维特征 (窗 [lo,hi) 为 latent 帧区间):
    [min 同位自相似, mean 同位自相似, min 3x3 邻域最大匹配, mean 与帧 0 同位相似]。"""
    rows = np.empty((len(cells), 4), np.float64)
    for i, ((gy, gx), (lo, hi)) in enumerate(zip(cells, windows)):
        s = sim_prev[lo - 1:hi - 1, gy, gx]
        rows[i] = (s.min(), s.mean(), nb_max[lo - 1:hi - 1, gy, gx].min(),
                   sim0[lo:hi, gy, gx].mean())
    return rows


def _lr_fit_predict_numpy(Xtr, ytr, Xte, iters=800, lr=0.5, l2=1e-3):
    """手写逻辑回归 (标准化 + 全批梯度下降 + L2), sklearn 缺失时的退路。"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    w = np.zeros(A.shape[1])
    y = ytr.astype(np.float64)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        w -= lr * (A.T @ (p - y) / len(y) + l2 * w)
    return 1.0 / (1.0 + np.exp(-np.clip(B @ w, -30, 30)))


def make_lr_backend(force_numpy=False):
    """返回 (backend_name, fit_predict(Xtr, ytr, Xte) -> p(y=1|Xte))。"""
    if not force_numpy:
        try:
            from sklearn.linear_model import LogisticRegression

            def fit_predict(Xtr, ytr, Xte):
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
                clf = LogisticRegression(max_iter=1000)
                clf.fit((Xtr - mu) / sd, ytr)
                return clf.predict_proba((Xte - mu) / sd)[:, 1]

            return 'sklearn', fit_predict
        except ImportError:
            pass
    return 'numpy_gd', _lr_fit_predict_numpy


def loo_probe(X, y, uids, fit_predict):
    """leave-one-case-out 交叉验证: 每例留出、其余训练, 汇总留出分数。
    返回 (pooled LOO-AUC, mean_case_auc, n_folds)。"""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)
    uids = np.asarray(uids)
    scores = np.full(len(y), np.nan)
    case_auc = []
    for uid in sorted(set(uids.tolist())):
        te = uids == uid
        tr = ~te
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = fit_predict(X[tr], y[tr], X[te])
        scores[te] = sc
        a = G1.rank_auc(sc[y[te] == 1], sc[y[te] == 0])
        if a == a:
            case_auc.append(a)
    ok = ~np.isnan(scores)
    pooled = G1.rank_auc(scores[ok & (y == 1)], scores[ok & (y == 0)])
    mca = float(np.mean(case_auc)) if case_auc else float('nan')
    return pooled, mca, len(case_auc)


def judge(auc):
    if auc != auc:
        return '无有效 AUC'
    if auc >= 0.8:
        return '可建 (>=0.8)'
    if auc >= 0.7:
        return '弱可分 (0.7-0.8)'
    return '止损 (<0.7)'


# ====================================================================================
#  selftest: 合成数据干跑 (无 torch/GPU), 覆盖窗口/特征/AUC/LOO 全链路
# ====================================================================================
def selftest():
    print('[g1p selftest] start')
    assert dup_window(0) == (1, 4) and dup_window(5) == (4, 7)
    assert dup_window(23) == (22, 24) and dup_window(1) == (1, 4)
    rng = np.random.RandomState(0)
    for lo, hi in bg_windows(500, rng=rng):
        assert 1 <= lo and hi - lo == 3 and hi <= T_LAT

    # 合成 6 例: 背景格恒定向量+微噪 (自相似高), 复制格在 a_t 帧切换到新向量 (瞬态 dip)
    T, H, W, D = T_LAT, H_LAT, W_LAT, 16
    Xs, ys, uids, t_pos, t_neg = [], [], [], [], []
    for ci in range(6):
        r = np.random.RandomState(100 + ci)
        base = r.randn(1, H, W, D).astype(np.float32)
        feat = np.repeat(base, T, axis=0) + 0.05 * r.randn(T, H, W, D).astype(np.float32)
        box = [40, 200, 140, 360]
        dup_cells = G1.box_px_to_cells(box)
        a_t = 5 + ci * 3
        newv = r.randn(1, len(dup_cells), D).astype(np.float32)
        dy = np.array([p[0] for p in dup_cells])
        dx = np.array([p[1] for p in dup_cells])
        feat[a_t:, dy, dx] = newv + 0.05 * r.randn(T - a_t, len(dup_cells), D)
        sim_prev, nb_max, sim0 = selfsim_maps(feat)
        assert (nb_max >= sim_prev - 1e-4).all()
        bg_cells = G1.sample_bg_cells(dup_cells, n_bg=60, seed=ci)
        wlo, whi = dup_window(a_t)
        Xd = cell_features(sim_prev, nb_max, sim0, dup_cells,
                           [(wlo, whi)] * len(dup_cells))
        Xb = cell_features(sim_prev, nb_max, sim0, bg_cells,
                           bg_windows(len(bg_cells), rng=np.random.RandomState(ci)))
        t_pos.append(1.0 - Xd[:, 0])
        t_neg.append(1.0 - Xb[:, 0])
        Xs.append(np.vstack([Xd, Xb]))
        ys.append(np.r_[np.ones(len(Xd), int), np.zeros(len(Xb), int)])
        uids.extend([f'case{ci}'] * (len(Xd) + len(Xb)))
    t_auc = G1.rank_auc(np.concatenate(t_pos), np.concatenate(t_neg))
    print(f'[g1p selftest] planted transient AUC = {t_auc:.4f}')
    assert t_auc > 0.9, t_auc
    X, y = np.vstack(Xs), np.concatenate(ys)
    for force in (True, False):
        name, fp = make_lr_backend(force_numpy=force)
        auc, mca, nf = loo_probe(X, y, uids, fp)
        print(f'[g1p selftest] LOO-AUC[{name}] = {auc:.4f} mean_case={mca:.4f} folds={nf}')
        assert auc > 0.9, (name, auc)
    print('[g1p selftest] SELFTEST_OK')


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
    cases = load_all_cases(args.pools.split(','), args.it2v)[:args.limit]
    sigmas = [float(s) for s in args.sigmas.split(',')]
    blocks = [f'block{i}'
              for i in range(args.block_lo, args.block_hi + 1, args.block_step)]
    print(f'[g1p] {len(cases)} cases, sigmas={sigmas}, blocks={blocks}', flush=True)

    per_case = {}
    pool = {}   # (block, sigma) -> dict(dup_min/bg_min/case_auc/X/y/uid)
    for ci, c in enumerate(cases):
        uid = c['uid']
        dup_all = G1.box_px_to_cells(c['box'])
        rngc = np.random.RandomState(args.seed + 1000 + ci)
        dup_cells = dup_all
        if len(dup_cells) > args.n_dup_max:
            pick = rngc.choice(len(dup_cells), size=args.n_dup_max, replace=False)
            dup_cells = [dup_cells[i] for i in pick]
        bg_cells = G1.sample_bg_cells(dup_all, n_bg=args.n_bg, edge=args.edge,
                                      seed=args.seed + ci)
        lt_lo, lt_hi = G1.lat_span(c['d0'], c['dend'])
        wlo, whi = dup_window(lt_lo, args.win)
        dwins = [(wlo, whi)] * len(dup_cells)
        bwins = bg_windows(len(bg_cells), args.win, rng=rngc)
        frames = P.sample_frames_like_training(c['video'], NF, HPIX, WPIX)
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        fn_norm = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(c['prompt'], max_length=512).to(device)
        crec = per_case.setdefault(uid, dict(
            pool=c['pool'], d0=c['d0'], dend=c['dend'],
            lt_span=[int(lt_lo), int(lt_hi)], dup_window=[int(wlo), int(whi)],
            n_dup_cells=len(dup_cells), n_bg_cells=len(bg_cells), stats={}))
        for sigma in sigmas:
            store = P.forward_with_hooks(models, fn_norm, emb, sigma, args.fps,
                                         device, dtype, edm)
            for kk in list(store):
                if kk not in blocks:
                    del store[kk]
            for b in blocks:
                if b not in store:
                    continue
                feat = store[b].float().numpy()          # (T, H, W, D)
                T, H, W, _ = feat.shape
                if (H, W) != (H_LAT, W_LAT):
                    print(f'  {uid} {b}: 特征网格 {H}x{W} != {H_LAT}x{W_LAT}, 跳过',
                          flush=True)
                    continue
                sim_prev, nb_max, sim0 = selfsim_maps(feat)
                Xd = cell_features(sim_prev, nb_max, sim0, dup_cells, dwins)
                Xb = cell_features(sim_prev, nb_max, sim0, bg_cells, bwins)
                t_auc = G1.rank_auc(1.0 - Xd[:, 0], 1.0 - Xb[:, 0])
                pr = pool.setdefault((b, sigma), dict(
                    dup_min=[], bg_min=[], case_auc=[], X=[], y=[], uid=[]))
                pr['dup_min'].append(Xd[:, 0])
                pr['bg_min'].append(Xb[:, 0])
                pr['case_auc'].append(t_auc)
                pr['X'].extend([Xd, Xb])
                pr['y'].append(np.r_[np.ones(len(Xd), int), np.zeros(len(Xb), int)])
                pr['uid'].extend([uid] * (len(Xd) + len(Xb)))
                crec['stats'][f'{b}|{sigma}'] = dict(
                    transient_auc=t_auc,
                    dup_min_mean=float(Xd[:, 0].mean()),
                    bg_min_mean=float(Xb[:, 0].mean()))
            del store
        print(f'  [{ci + 1}/{len(cases)}] {uid}: dup_cells={len(dup_cells)} '
              f'lt=[{lt_lo},{lt_hi}) win=[{wlo},{whi}) done', flush=True)

    # ---- 汇总: 瞬态 AUC + 可训探针 LOO-AUC ----
    backend, fit_predict = make_lr_backend(force_numpy=args.force_numpy_lr)
    print(f'[g1p] probe backend = {backend}', flush=True)
    transient, probe = {}, {}
    for (b, s), pr in sorted(pool.items()):
        dmin = np.concatenate(pr['dup_min'])
        bmin = np.concatenate(pr['bg_min'])
        ca = np.array([a for a in pr['case_auc'] if a == a])
        transient[f'{b}|{s}'] = dict(
            pooled_auc=G1.rank_auc(1.0 - dmin, 1.0 - bmin),
            mean_case_auc=float(ca.mean()) if ca.size else float('nan'),
            dup_min_mean=float(dmin.mean()), bg_min_mean=float(bmin.mean()),
            n_cases=len(pr['case_auc']), n_dup=int(dmin.size), n_bg=int(bmin.size))
        loo_auc, loo_mca, n_folds = loo_probe(np.vstack(pr['X']),
                                              np.concatenate(pr['y']),
                                              pr['uid'], fit_predict)
        probe[f'{b}|{s}'] = dict(loo_auc=loo_auc, mean_case_auc=loo_mca,
                                 n_folds=n_folds, backend=backend)
        print(f'  [loo] {b}|{s}: loo_auc={loo_auc:.4f}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'g1p_report.json')
    json.dump(dict(args=vars(args), n_cases=len(per_case), per_case=per_case,
                   transient=transient, probe=probe),
              open(out_path, 'w'), indent=1)
    print(f'[g1p] -> {out_path}', flush=True)

    # ---- 榜单 1: 瞬态 AUC (降序) ----
    print('\n[G1P] 榜单 1: 瞬态 AUC (出现窗最低自相似 vs 背景随机窗, 分数=1-min_selfsim)',
          flush=True)
    print(f'{"rank":<6}{"block":<9}{"sigma":<7}{"pooled_auc":<12}{"mean_case_auc":<15}'
          f'{"dup_min":<9}{"bg_min":<9}{"n_cases":<8}', flush=True)
    t_rank = sorted(transient.items(), key=lambda kv: -kv[1]['pooled_auc'])
    for i, (k, v) in enumerate(t_rank):
        b, s = k.split('|')
        print(f'{i + 1:<6}{b:<9}{s:<7}{v["pooled_auc"]:<12.4f}{v["mean_case_auc"]:<15.4f}'
              f'{v["dup_min_mean"]:<9.4f}{v["bg_min_mean"]:<9.4f}{v["n_cases"]:<8}',
              flush=True)

    # ---- 榜单 2: 可训探针 LOO-AUC (降序) ----
    print('\n[G1P] 榜单 2: 可训探针 LOO-AUC (4 维特征逻辑回归, 34 例逐例留出)', flush=True)
    print(f'{"rank":<6}{"block":<9}{"sigma":<7}{"loo_auc":<10}{"mean_case_auc":<15}'
          f'{"n_folds":<9}{"backend":<10}', flush=True)
    p_rank = sorted(probe.items(), key=lambda kv: -kv[1]['loo_auc'])
    for i, (k, v) in enumerate(p_rank):
        b, s = k.split('|')
        print(f'{i + 1:<6}{b:<9}{s:<7}{v["loo_auc"]:<10.4f}{v["mean_case_auc"]:<15.4f}'
              f'{v["n_folds"]:<9}{v["backend"]:<10}', flush=True)

    # ---- 结论: top-2 层 (按探针 LOO-AUC, 每层取最优 sigma) + 判决 ----
    top_blocks, seen = [], set()
    for k, v in p_rank:
        b = k.split('|')[0]
        if b not in seen:
            seen.add(b)
            top_blocks.append((k, v['loo_auc'], transient[k]['pooled_auc']))
        if len(top_blocks) == 2:
            break
    if top_blocks:
        best_auc = top_blocks[0][1]
        desc = ', '.join(f'{k} loo_auc={a:.4f} transient={t:.4f}'
                         for k, a, t in top_blocks)
        best_t = t_rank[0]
        print(f'\n[G1P] CONCLUSION: top-2 = [{desc}] | '
              f'best_transient={best_t[0]} {best_t[1]["pooled_auc"]:.4f} | '
              f'判决(按最优 LOO-AUC {best_auc:.4f}): {judge(best_auc)}', flush=True)
    print('G1P_DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pools', default=DEFAULT_POOLS)
    ap.add_argument('--it2v', default='/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json')
    ap.add_argument('--model-dir', default='/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--sigmas', default='0.2,0.4')
    ap.add_argument('--block-lo', type=int, default=8)
    ap.add_argument('--block-hi', type=int, default=26)
    ap.add_argument('--block-step', type=int, default=2)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--win', type=int, default=3)
    ap.add_argument('--n-bg', type=int, default=200)
    ap.add_argument('--n-dup-max', type=int, default=200)
    ap.add_argument('--edge', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--force-numpy-lr', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--out-dir', default='/data/datasets/gagi/eve_v2_outputs/selfcase/g1p')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run(args)


if __name__ == '__main__':
    main()
