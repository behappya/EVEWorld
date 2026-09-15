#!/usr/bin/env python3
"""SELFCASE-CIC: CIC 式 t->t-1 "认亲"匹配特征验矿 (只验特征可分性, 不动任何训练)。

背景: g1p/fusion 现有基线 = 3L_block10_12_16 (12 维) LOO-AUC 0.8047@sigma0.4 (34 例)。
v1 (block22 单层 4 维): 40 例上 A=0.8113 B=0.8261 C=0.6921 @0.4, B-A=+0.0148 < 0.02 止损。
v2 (本版, 方向调整): 认亲匹配不用 block22, 改在 block10/12/16 (涌现信号最强三层)
逐层各算 F1-F4 (4x3=12 新维); 判决口径放宽 B-A>=0.01 即报"可并入" (除 AUC 增益外
还有论文叙事价值: ICH 反向认亲与 TIA 正向接力共享匹配原语的统一框架), 数字如实报。

对照 (同 g1p LOO 协议, 40 例 = 四池 34 + mine_pretrain 6):
  A = 现有 12 维 (block10/12/16 g1p 4 维)         —— 复现基线
  B = 12 + 新 12 维 (加三层认亲特征)               —— 增益判决
  C = 新 12 维单独                                  —— 单独战力
  C_blockXX = 各层认亲 4 维单独 (10 vs 12 vs 16 逐层对照)

新特征 (方向必须 t -> t-1 "认亲": t 帧每格在 t-1 帧**全图**找最佳匹配;
绝不反向 t-1 -> t —— 历史 E6 已否决正向):
  F1 来路距离: argmax 匹配位置到本格的欧氏格距
     (正常静止 ~0 / 合法移动 <=2 / 复制品 ~ 远处原物距离);
  F2 匹配锐度 margin: 最佳与次佳匹配值之差 (伴随特征: margin 低则 F1 不可信,
     防桌面均匀纹理假匹配, 让分类器自己学会打折);
  F3 来源重复认领数: 本格 argmax 源格总共被多少个 t 帧格子认领
     (只统计 margin > 当帧全图中位数的认领 —— rank 阈值, 非绝对相似度阈值;
      否则背景平坦区海量互相认领产生垃圾计数);
  F4 与帧 0 同位差异 = 1 - cos(f[t,c], f[0,c]) (内容门控: 背景格匹配特征无意义,
     给分类器一个"这格有没有新内容"的开关)。
逐格聚合 (窗 [lo,hi) 与 g1p 完全同款: dup 格用出现瞬态窗, 背景格用随机窗):
  [F1 = 窗内最大来路距离, F2 = 该帧的伴随 margin, F3 = 窗内最大重复认领数,
   F4 = 窗内平均 (1 - sim0)]。

对抗红线 (selftest 逐条验证, 提交前干跑):
  (1) 合法双静物 (distractor 对, 共享外观 + 位置上下文分量): 各自认领自己位置,
      距离 0, 不触发 (F1>=4 且 F3>=2) 签名;
  (2) 被搬运物体: 单认领 + 距离 1-2, 不触发签名;
  (3) 复制品: 原物位置被认领 >=2 且其中一个认领距离大 —— 要抓的签名;
  (4) 全部 rank/argmax/margin 特征, 禁用绝对相似度阈值 (E6 教训);
  (5) 匹配在 sigma∈{0.2,0.4} 加噪特征上做 (与 g1p 同协议)。

协议对齐 (可对账): 病例顺序 = 四旧池在前 + mine_pretrain 在后, 前 34 例的
rng 流 (格/窗采样) 与 fusion_ablation 完全一致; 旧缓存存在时先用旧缓存复算
A_repro (应精确复现 0.8047), 新抽取的 34 例子集 A34 作数值漂移对照。

输出: <out-dir>/cic_match_report.json
  + 逐格特征缓存 cic_cell_features.npz (供复用, --fuse-only 纯 CPU 重跑)
  + 榜单 (A/B/C/逐层 x sigma) + 结论行 (B-A 增益 / C 单独 / 逐层 / 判决)。
GPU 作业, 单卡单进程; `--selftest` 纯 numpy 单测 (无 torch/GPU)。
"""
import argparse
import json
import os

import numpy as np

import selfcase_g1 as G1
import selfcase_g1p as G1P

POOLS_DEFAULT = 'mine_round0,mine_gr1_2b,mine_s150,mine_wmapA_pre_s250,mine_pretrain'
OLD_POOLS = ('mine_round0', 'mine_gr1_2b', 'mine_s150', 'mine_wmapA_pre_s250')
A_BLOCKS = ['block10', 'block12', 'block16']     # 现有 12 维基线 (fusion 3L, 0.8047@0.4)
CIC_BLOCKS_DEFAULT = 'block10,block12,block16'   # v2: 认亲特征改在涌现信号最强三层

OUT_DIR_DEFAULT = '/data/datasets/gagi/eve_v2_outputs/selfcase/cic_match_v2'
CACHE_NAME = 'cic_cell_features.npz'
REPORT_NAME = 'cic_match_report.json'
OLD_CACHE = '/data/datasets/gagi/eve_v2_outputs/selfcase/g1p/g1p_cell_features.npz'

BASELINE_REF = 0.8047                            # 3L_block10_12_16|0.4 @ 34 例
GAIN_GATE_DEFAULT = 0.01                         # v2 放宽: B-A>=0.01 即"可并入"
                                                 # (叙事价值: ICH 反向认亲 x TIA 正向接力
                                                 #  共享匹配原语的统一框架)


# ====================================================================================
#  纯函数 (无 torch 依赖, --selftest 本机单测)
# ====================================================================================
def match_maps(feat):
    """feat: np float32 (T,H,W,D) -> t->t-1 全图认亲匹配的逐帧全图特征。
    返回 (dist, margin, claims, sim0):
      dist[t-1,y,x]   : t 帧格 (y,x) 在 t-1 全图 argmax 匹配位置到本格的欧氏格距
      margin[t-1,y,x] : 最佳与次佳匹配余弦之差
      claims[t-1,y,x] : 本格 argmax 源格被多少个 t 帧格子认领
                        (仅统计 margin > 当帧全图中位数的认领)
      sim0[t,y,x]     : cos(f[t,y,x], f[0,y,x])
    方向固定 t -> t-1 (认亲), 不做反向 (E6 已否决正向)。
    """
    f = np.asarray(feat, np.float32)
    T, H, W, D = f.shape
    fn = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)
    flat = fn.reshape(T, H * W, D)
    hw = H * W
    ys, xs = np.divmod(np.arange(hw), W)
    ii = np.arange(hw)
    dist = np.empty((T - 1, H, W), np.float32)
    margin = np.empty((T - 1, H, W), np.float32)
    claims = np.empty((T - 1, H, W), np.float32)
    for t in range(1, T):
        S = flat[t] @ flat[t - 1].T                    # (HW, HW): S[i,j]=cos(f_t[i], f_{t-1}[j])
        src = S.argmax(1)
        v1 = S[ii, src]
        S[ii, src] = -np.inf
        m = v1 - S.max(1)                              # top1 - top2
        d = np.hypot(ys - ys[src], xs - xs[src])
        med = np.median(m)
        cnt = np.bincount(src[m > med], minlength=hw)  # 过筛认领计数
        dist[t - 1] = d.reshape(H, W)
        margin[t - 1] = m.reshape(H, W)
        claims[t - 1] = cnt[src].reshape(H, W).astype(np.float32)
    sim0 = np.einsum('thwd,hwd->thw', fn, fn[0])
    return dist, margin, claims, sim0


def cic_cell_features(dist, margin, claims, sim0, cells, windows):
    """逐格 4 维 CIC 特征 (窗 [lo,hi) 为 latent 帧区间, lo>=1):
    [F1=窗内最大来路距离, F2=该帧伴随 margin, F3=窗内最大重复认领数,
     F4=窗内平均 (1-sim0)]。"""
    rows = np.empty((len(cells), 4), np.float64)
    for i, ((gy, gx), (lo, hi)) in enumerate(zip(cells, windows)):
        dd = dist[lo - 1:hi - 1, gy, gx]
        t_star = int(dd.argmax())
        rows[i] = (dd[t_star], margin[lo - 1 + t_star, gy, gx],
                   claims[lo - 1:hi - 1, gy, gx].max(),
                   1.0 - sim0[lo:hi, gy, gx].mean())
    return rows


def signature_fire(dist, claims, dist_ge=4.0, claims_ge=2.0):
    """复制品签名谓词: 来路距离远 且 源格被重复认领 (对抗红线用)。"""
    return (dist >= dist_ge) & (claims >= claims_ge)


# ====================================================================================
#  selftest: 对抗红线四场景 + 窗聚合 + LOO 管线, 纯 numpy 干跑
# ====================================================================================
def _mk_scene(rng, T=12, H=G1.H_LAT, W=G1.W_LAT, D=128, noise=0.02):
    """背景 = 共享分量 (背景互相有点像, margin 别虚高) + 每格独立分量 + 逐帧微噪。"""
    g = rng.randn(1, 1, D).astype(np.float32)
    bg = 0.8 * g + rng.randn(H, W, D).astype(np.float32)
    feat = np.repeat(bg[None], T, 0) + noise * rng.randn(T, H, W, D).astype(np.float32)
    return bg, feat


def selftest():
    print('[cic selftest] start')
    rng = np.random.RandomState(0)
    T, H, W, D = 12, G1.H_LAT, G1.W_LAT, 128

    # ---- 红线 (1): 合法双静物 distractor 对 (共享外观 + 位置上下文分量) ----
    _, feat = _mk_scene(np.random.RandomState(1), T, H, W, D)
    obj = rng.randn(D).astype(np.float32) * 1.5
    o1, o2 = (5, 5), (20, 30)
    for (oy, ox) in (o1, o2):
        pos = 0.6 * rng.randn(3, 3, D).astype(np.float32)   # 每物体独立位置上下文
        feat[:, oy:oy + 3, ox:ox + 3] = obj + pos \
            + 0.02 * rng.randn(T, 3, 3, D).astype(np.float32)
    dist, margin, claims, _ = match_maps(feat)
    cells = [(oy + a, ox + b) for (oy, ox) in (o1, o2) for a in range(3) for b in range(3)]
    dv = np.array([dist[:, y, x] for y, x in cells])
    cv = np.array([claims[:, y, x] for y, x in cells])
    assert dv.max() == 0.0, f'distractor 距离应为 0, got max={dv.max()}'
    fire = signature_fire(dv, cv)
    assert not fire.any(), 'distractor 对不应触发复制品签名'
    print(f'[cic selftest] 红线1 双静物: dist_max={dv.max():.1f} '
          f'claims_max={cv.max():.0f} fire={int(fire.sum())} OK')

    # ---- 红线 (2): 被搬运物体 (每帧 +1 格) 单认领 + 距离 1, 不触发 ----
    _, feat = _mk_scene(np.random.RandomState(2), T, H, W, D)
    V = rng.randn(3, 3, D).astype(np.float32) * 1.5
    oy, ox0 = 12, 6
    for t in range(T):
        feat[t, oy:oy + 3, ox0 + t:ox0 + t + 3] = V + 0.02 * rng.randn(3, 3, D)
    dist, margin, claims, _ = match_maps(feat)
    dvals, cvals = [], []
    for t in range(1, T):
        for a in range(3):
            for b in range(3):
                y, x = oy + a, ox0 + t + b
                dvals.append(dist[t - 1, y, x])
                cvals.append(claims[t - 1, y, x])
    dvals, cvals = np.array(dvals), np.array(cvals)
    assert np.median(dvals) <= 2.0, f'搬运距离中位应<=2, got {np.median(dvals)}'
    assert dvals.max() <= 2.0, f'搬运距离应<=2, got {dvals.max()}'
    fire = signature_fire(dvals, cvals)
    assert not fire.any(), '搬运物体不应触发复制品签名'
    print(f'[cic selftest] 红线2 搬运: dist_med={np.median(dvals):.1f} '
          f'claims_max={cvals.max():.0f} fire={int(fire.sum())} OK')

    # ---- 红线 (3): 复制品 = 原物位置被认领>=2 且其中一个认领距离大 (要抓的签名) ----
    _, feat = _mk_scene(np.random.RandomState(3), T, H, W, D)
    Vo = rng.randn(3, 3, D).astype(np.float32) * 1.5
    (oy, ox), (cy, cx), a_t = (6, 8), (22, 38), 6
    pos_c = 0.3 * rng.randn(3, 3, D).astype(np.float32)   # 复制品的位置上下文分量
    feat[:, oy:oy + 3, ox:ox + 3] = Vo + 0.02 * rng.randn(T, 3, 3, D)      # 原物常驻
    # 复制品出现: 外观=原物 Vo (要抓的复制), 叠位置上下文 (静止后自认领应赢过认原物)
    feat[a_t:, cy:cy + 3, cx:cx + 3] = Vo + pos_c + 0.02 * rng.randn(T - a_t, 3, 3, D)
    dist, margin, claims, sim0 = match_maps(feat)
    dup_cells = [(cy + a, cx + b) for a in range(3) for b in range(3)]
    org_cells = [(oy + a, ox + b) for a in range(3) for b in range(3)]
    dd = np.array([dist[a_t - 1, y, x] for y, x in dup_cells])
    dc = np.array([claims[a_t - 1, y, x] for y, x in dup_cells])
    oc = np.array([claims[a_t - 1, y, x] for y, x in org_cells])
    true_d = np.hypot(cy - oy, cx - ox)
    assert (dd >= true_d - 3).all(), f'复制格来路距离应~{true_d:.0f}, got min={dd.min()}'
    assert (dc >= 2).all(), f'复制格 F3 应>=2, got min={dc.min()}'
    assert (oc >= 2).all(), f'原物格 F3 应>=2, got min={oc.min()}'
    assert signature_fire(dd, dc).all(), '复制格应全部触发签名'
    # 出现后静止 (t>a_t): 复制格自认领, 距离回 0 (瞬态) -> 窗聚合必须抓出现帧
    dd_later = np.array([dist[a_t, y, x] for y, x in dup_cells])
    assert dd_later.max() == 0.0, f'复制品静止后距离应回 0, got {dd_later.max()}'
    # 窗聚合: dup 窗 [a_t-1, a_t+2) vs 背景随机窗, F1/F3 可分
    bg_cells = G1.sample_bg_cells(dup_cells + org_cells, n_bg=150, seed=0)
    wlo, whi = G1P.dup_window(a_t, 3, t_lat=T)
    Xd = cic_cell_features(dist, margin, claims, sim0, dup_cells,
                           [(wlo, whi)] * len(dup_cells))
    Xb = cic_cell_features(dist, margin, claims, sim0, bg_cells,
                           G1P.bg_windows(len(bg_cells), 3, t_lat=T,
                                          rng=np.random.RandomState(0)))
    auc_f1 = G1.rank_auc(Xd[:, 0], Xb[:, 0])
    auc_f3 = G1.rank_auc(Xd[:, 2], Xb[:, 2])
    assert auc_f1 > 0.95 and auc_f3 > 0.9, (auc_f1, auc_f3)
    print(f'[cic selftest] 红线3 复制品: dist_min={dd.min():.1f} (true {true_d:.1f}) '
          f'F3_dup_min={dc.min():.0f} F3_org_min={oc.min():.0f} '
          f'窗聚合 AUC(F1)={auc_f1:.3f} AUC(F3)={auc_f3:.3f} OK')

    # ---- 红线 (4)/(F2): 平坦纹理区 margin 低 + 过筛认领 << 未过筛 (垃圾计数被压) ----
    _, feat = _mk_scene(np.random.RandomState(4), T, H, W, D)
    flat_v = rng.randn(D).astype(np.float32)
    fy, fx, fs = 10, 20, 8
    feat[:, fy:fy + fs, fx:fx + fs] = flat_v + 0.05 * rng.randn(T, fs, fs, D)
    f = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)
    flat = f.reshape(T, H * W, D)
    S = flat[1] @ flat[0].T
    src = S.argmax(1)
    raw_cnt = np.bincount(src, minlength=H * W)          # 不过筛的认领计数
    dist, margin, claims, _ = match_maps(feat)
    fmask = np.zeros((H, W), bool)
    fmask[fy:fy + fs, fx:fx + fs] = True
    flat_margin = margin[0][fmask]
    bg_margin = margin[0][~fmask]
    raw_flat_max = raw_cnt.reshape(H, W)[fmask].max()
    filt_flat_max = claims[0][fmask].max()
    assert np.median(flat_margin) < np.median(bg_margin), '平坦区 margin 应显著低'
    assert filt_flat_max < raw_flat_max, \
        f'过筛认领应压掉平坦区垃圾计数: {filt_flat_max} !< {raw_flat_max}'
    fire = signature_fire(dist[0][fmask], claims[0][fmask])
    assert not fire.any(), '平坦纹理区不应触发复制品签名'
    print(f'[cic selftest] 红线4 平坦纹理: margin_med={np.median(flat_margin):.3f} '
          f'(bg {np.median(bg_margin):.3f}) claims raw_max={raw_flat_max} '
          f'-> filt_max={filt_flat_max:.0f} fire=0 OK')

    # ---- LOO 管线: 信号种在 CIC 维 (三层 x 4 维) -> C/B 高, A 盲 ----
    rng2 = np.random.RandomState(7)
    Xa_rows, Xc_rows, y_rows, uid_rows = [], [], [], []
    for ci in range(8):
        n_pos, n_neg = 30, 60
        y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
        Xa = rng2.randn(len(y), 12)                      # A: 纯噪声 (盲)
        Xc = rng2.randn(len(y), 12) * 0.5                # 三层 x 4 维
        for k in range(0, 12, 4):
            Xc[:, k] += y * 3.0                          # 各层 F1 来路距离
            Xc[:, k + 2] += y * 2.0                      # 各层 F3 重复认领
        Xa_rows.append(Xa); Xc_rows.append(Xc); y_rows.append(y)
        uid_rows.extend([f'case{ci}'] * len(y))
    Xa, Xc = np.vstack(Xa_rows), np.vstack(Xc_rows)
    y, uid = np.concatenate(y_rows), np.array(uid_rows)
    _, fp = G1P.make_lr_backend(force_numpy=True)
    auc_a, _, _ = G1P.loo_probe(Xa, y, uid, fp)
    auc_b, _, _ = G1P.loo_probe(np.hstack([Xa, Xc]), y, uid, fp)
    auc_c, _, _ = G1P.loo_probe(Xc, y, uid, fp)
    print(f'[cic selftest] LOO 管线: A={auc_a:.3f} B={auc_b:.3f} C={auc_c:.3f}')
    assert auc_c > 0.95 and auc_b > 0.95 and abs(auc_a - 0.5) < 0.1
    print('[cic selftest] SELFTEST_OK')


# ====================================================================================
#  GPU 抽特征 -> 逐格特征缓存 (fusion_ablation.extract 同款采样, 前 34 例 rng 流一致)
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
    cic_blocks = args.cic_blocks.split(',')
    keep_blocks = list(dict.fromkeys(A_BLOCKS + cic_blocks))
    print(f'[cic] extract: {len(cases)} cases, sigmas={sigmas}, '
          f'A_blocks={A_BLOCKS}, cic_blocks={cic_blocks}', flush=True)

    rows_a = {(b, s): [] for b in A_BLOCKS for s in sigmas}
    rows_c = {(b, s): [] for b in cic_blocks for s in sigmas}
    y_rows, uid_rows, pool_rows = [], [], []
    per_case = {}
    for ci, c in enumerate(cases):
        uid = c['uid']
        # ---- 与 fusion_ablation.extract / g1p.run 完全同款采样 (rng 调用顺序一致) ----
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
        crec = per_case.setdefault(uid, dict(
            pool=c['pool'], d0=c['d0'], dend=c['dend'],
            lt_span=[int(lt_lo), int(lt_hi)], dup_window=[int(wlo), int(whi)],
            n_dup_cells=len(dup_cells), n_bg_cells=len(bg_cells), stats={}))
        for sigma in sigmas:
            store = P.forward_with_hooks(models, fn_norm, emb, sigma, args.fps,
                                         device, dtype, edm)
            for kk in list(store):
                if kk not in keep_blocks:
                    del store[kk]
            # ---- A: 现有 12 维 (block10/12/16 各 4 维, g1p 同款) ----
            for b in A_BLOCKS:
                feat = store[b].float().numpy()
                _, Hf, Wf, _ = feat.shape
                assert (Hf, Wf) == (G1.H_LAT, G1.W_LAT), (uid, b, Hf, Wf)
                sim_prev, nb_max, sim0 = G1P.selfsim_maps(feat)
                Xd = G1P.cell_features(sim_prev, nb_max, sim0, dup_cells, dwins)
                Xb = G1P.cell_features(sim_prev, nb_max, sim0, bg_cells, bwins)
                rows_a[(b, sigma)].append(np.vstack([Xd, Xb]))
            # ---- C: 新认亲 4 维 x 各 cic 层 (t->t-1 全图匹配) ----
            for b in cic_blocks:
                feat = store[b].float().numpy()
                _, Hf, Wf, _ = feat.shape
                assert (Hf, Wf) == (G1.H_LAT, G1.W_LAT), (uid, b, Hf, Wf)
                dist, margin, claims, sim0 = match_maps(feat)
                Xd = cic_cell_features(dist, margin, claims, sim0, dup_cells, dwins)
                Xb = cic_cell_features(dist, margin, claims, sim0, bg_cells, bwins)
                rows_c[(b, sigma)].append(np.vstack([Xd, Xb]))
                crec['stats'][f'cic_{b}|{sigma}'] = dict(
                    f1_auc=G1.rank_auc(Xd[:, 0], Xb[:, 0]),
                    f3_auc=G1.rank_auc(Xd[:, 2], Xb[:, 2]),
                    dup_f1_mean=float(Xd[:, 0].mean()), bg_f1_mean=float(Xb[:, 0].mean()),
                    dup_f3_mean=float(Xd[:, 2].mean()), bg_f3_mean=float(Xb[:, 2].mean()))
            del store
        y_rows.append(np.r_[np.ones(len(dup_cells), int), np.zeros(len(bg_cells), int)])
        uid_rows.extend([uid] * (len(dup_cells) + len(bg_cells)))
        pool_rows.extend([c['pool']] * (len(dup_cells) + len(bg_cells)))
        st = crec['stats'][f'cic_{cic_blocks[-1]}|{sigmas[-1]}']
        print(f'  [{ci + 1}/{len(cases)}] {uid}: dup={len(dup_cells)} bg={len(bg_cells)} '
              f'win=[{wlo},{whi}) {cic_blocks[-1]}_f1_auc={st["f1_auc"]:.3f} '
              f'f3_auc={st["f3_auc"]:.3f} done', flush=True)

    y = np.concatenate(y_rows)
    payload = dict(y=y, uid=np.array(uid_rows), pool=np.array(pool_rows),
                   meta=np.array(json.dumps(dict(
                       pools=args.pools, sigmas=sigmas, a_blocks=A_BLOCKS,
                       cic_blocks=cic_blocks, seed=args.seed, n_cases=len(cases),
                       n_rows=int(len(y)))), dtype=object))
    for (b, s), xs in rows_a.items():
        payload[f'X_{b}_{s}'] = np.vstack(xs)
    for (b, s), xs in rows_c.items():
        payload[f'X_cic_{b}_{s}'] = np.vstack(xs)
    for k in payload:
        if k.startswith('X_'):
            assert len(payload[k]) == len(y), (k, payload[k].shape, y.shape)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez_compressed(args.cache, **payload)
    json.dump(per_case, open(os.path.join(os.path.dirname(args.cache),
                                          'cic_per_case.json'), 'w'), indent=1)
    print(f'[cic] cell feature cache -> {args.cache} ({len(y)} rows)', flush=True)


# ====================================================================================
#  CPU 融合 LOO: A_repro(旧缓存精确对账) / A34 / A / B / C / 逐层
# ====================================================================================
def old_cache_repro(sigmas, fit_predict, old_cache=OLD_CACHE):
    """旧 34 例缓存上复算 12 维基线 (应精确复现 fusion_ablation 的 0.8047@0.4)。"""
    if not os.path.exists(old_cache):
        print(f'[cic] 旧缓存不存在, 跳过精确对账: {old_cache}', flush=True)
        return {}
    d = np.load(old_cache, allow_pickle=True)
    y, uid = d['y'], d['uid']
    out = {}
    for s in sigmas:
        X = np.hstack([d[f'X_{b}_{s}'] for b in A_BLOCKS])
        auc, mca, nf = G1P.loo_probe(X, y, uid, fit_predict)
        out[s] = dict(loo_auc=auc, mean_case_auc=mca, n_folds=nf)
        print(f'  [repro] A_repro(旧缓存34例)|{s}: loo_auc={auc:.4f} '
              f'(基线参考 {BASELINE_REF})', flush=True)
    return out


def fuse_loo(cache_path, sigmas, out_path, gain_gate=GAIN_GATE_DEFAULT,
             force_numpy_lr=False):
    d = np.load(cache_path, allow_pickle=True)
    y, uid, pool = d['y'], d['uid'], d['pool']
    meta = json.loads(str(d['meta']))
    cic_blocks = meta['cic_blocks']
    backend, fit_predict = G1P.make_lr_backend(force_numpy=force_numpy_lr)
    n_cases = len(set(uid.tolist()))
    old34 = np.isin(pool, list(OLD_POOLS))
    print(f'[cic] {len(y)} rows ({int(y.sum())} dup / {int((1 - y).sum())} bg), '
          f'{n_cases} cases ({len(set(uid[old34].tolist()))} 旧池), '
          f'cic_blocks={cic_blocks}, backend={backend}', flush=True)

    repro = old_cache_repro(sigmas, fit_predict)

    configs = {}
    for s in sigmas:
        Xa = np.hstack([d[f'X_{b}_{s}'] for b in A_BLOCKS])
        Xcs = {b: d[f'X_cic_{b}_{s}'] for b in cic_blocks}
        Xc = np.hstack([Xcs[b] for b in cic_blocks])
        configs[f'A34|{s}'] = (Xa[old34], y[old34], uid[old34])
        configs[f'A|{s}'] = (Xa, y, uid)
        configs[f'B|{s}'] = (np.hstack([Xa, Xc]), y, uid)
        configs[f'C|{s}'] = (Xc, y, uid)
        for b in cic_blocks:
            configs[f'C_{b}|{s}'] = (Xcs[b], y, uid)

    results = {}
    for name, (X, yy, uu) in configs.items():
        auc, mca, nf = G1P.loo_probe(X, yy, uu, fit_predict)
        results[name] = dict(loo_auc=auc, mean_case_auc=mca, n_folds=nf,
                             n_dims=int(X.shape[1]), n_rows=int(len(yy)),
                             backend=backend)
        print(f'  [loo] {name}: dims={X.shape[1]} rows={len(yy)} '
              f'loo_auc={auc:.4f} mean_case={mca:.4f} folds={nf}', flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(dict(cache=cache_path, sigmas=sigmas, cic_blocks=cic_blocks,
                   baseline_ref=BASELINE_REF, gain_gate=gain_gate,
                   repro_old_cache={str(k): v for k, v in repro.items()},
                   results=results),
              open(out_path, 'w'), indent=1)
    print(f'[cic] -> {out_path}', flush=True)

    # ---- 榜单 ----
    print(f'\n[CIC] 对照榜单 (同 g1p LOO 协议: {n_cases} 例逐例留出, LR; '
          f'A=现有12维 B=12+认亲{4 * len(cic_blocks)}维 C=认亲单独 '
          f'C_blockXX=逐层4维)', flush=True)
    print(f'{"rank":<6}{"config":<14}{"sigma":<7}{"dims":<6}{"rows":<8}{"loo_auc":<10}'
          f'{"mean_case_auc":<15}{"n_folds":<8}', flush=True)
    rank = sorted(results.items(), key=lambda kv: -kv[1]['loo_auc'])
    for i, (k, v) in enumerate(rank):
        name, s = k.rsplit('|', 1)
        print(f'{i + 1:<6}{name:<14}{s:<7}{v["n_dims"]:<6}{v["n_rows"]:<8}'
              f'{v["loo_auc"]:<10.4f}{v["mean_case_auc"]:<15.4f}{v["n_folds"]:<8}',
              flush=True)

    # ---- 结论行 ----
    verdicts = {}
    for s in sigmas:
        a = results[f'A|{s}']['loo_auc']
        b = results[f'B|{s}']['loo_auc']
        cc = results[f'C|{s}']['loo_auc']
        gain = b - a
        verdicts[s] = (a, b, cc, gain)
        per_layer = ' '.join(f'{blk}={results[f"C_{blk}|{s}"]["loo_auc"]:.4f}'
                             for blk in cic_blocks)
        print(f'[CIC] CONCLUSION sigma={s}: A(12d)={a:.4f} '
              f'B({12 + 4 * len(cic_blocks)}d)={b:.4f} '
              f'C({4 * len(cic_blocks)}d)={cc:.4f} | B-A={gain:+.4f} | '
              f'逐层认亲4维: {per_layer}', flush=True)
    # 主判决口径 = 基线所在 sigma=0.4; 双 sigma 均报
    s_main = 0.4 if 0.4 in verdicts else sigmas[-1]
    a, b, cc, gain = verdicts[s_main]
    best_gain = max(v[3] for v in verdicts.values())
    if gain >= gain_gate:
        verdict = (f'可并入 ICH 检测子 (B-A>={gain_gate:.2f}; 口径含叙事价值: '
                   f'ICH 反向认亲 x TIA 正向接力共享匹配原语)')
    elif best_gain >= gain_gate:
        verdict = f'主口径未达标但另一 sigma 达标 (best B-A={best_gain:+.4f}), 边缘可并入'
    else:
        verdict = f'不值得并入 (B-A<{gain_gate:.2f})'
    a34 = results.get(f'A34|{s_main}', {}).get('loo_auc', float('nan'))
    print(f'[CIC] VERDICT (主口径 sigma={s_main}, gate={gain_gate}, '
          f'基线参考 {BASELINE_REF} A34_new={a34:.4f}): A={a:.4f} B={b:.4f} '
          f'C={cc:.4f} B-A={gain:+.4f} -> {verdict}', flush=True)
    print('CIC_DONE', flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pools', default=POOLS_DEFAULT)
    ap.add_argument('--it2v', default='/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json')
    ap.add_argument('--model-dir', default='/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--sigmas', default='0.2,0.4')
    ap.add_argument('--cic-blocks', default=CIC_BLOCKS_DEFAULT)
    ap.add_argument('--gain-gate', type=float, default=GAIN_GATE_DEFAULT)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--win', type=int, default=3)
    ap.add_argument('--n-bg', type=int, default=200)
    ap.add_argument('--n-dup-max', type=int, default=200)
    ap.add_argument('--edge', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--force-numpy-lr', action='store_true')
    ap.add_argument('--fuse-only', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.cache is None:
        args.cache = os.path.join(args.out_dir, CACHE_NAME)
    if not os.path.exists(args.cache):
        assert not args.fuse_only, f'--fuse-only 但缓存不存在: {args.cache}'
        extract(args)
    else:
        print(f'[cic] cache exists, skip extract: {args.cache}', flush=True)
    fuse_loo(args.cache, [float(s) for s in args.sigmas.split(',')],
             os.path.join(args.out_dir, REPORT_NAME),
             gain_gate=args.gain_gate, force_numpy_lr=args.force_numpy_lr)


if __name__ == '__main__':
    main()
