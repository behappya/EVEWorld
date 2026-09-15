#!/usr/bin/env python3
"""ICH-D 固化检测子拟合 (71 号 B1 D 臂预案): 40 例全量训 LR, 权重固化供 ICHDModule。

背景: 学习型检测子 (C 臂 conv 头 + BCE) step158 判决失败 (det 分离 +0.005 <
预注册线 0.05) -> 检测子改为离线拟合 + 固化 (buffer, 永不更新), 训练只学擦除。

特征 = v1 16 维 (LOO-AUC 0.8261@sigma0.4, cic_match_report.json):
  [block10 4d, block12 4d, block16 4d] g1p novelty + [block22 4d] CIC 认亲
  列序与 selfcase_cic_match.fuse_loo 的 B 配置严格一致:
    hstack([X_block10, X_block12, X_block16, X_cic])。
训练行 = sigma 0.2 与 0.4 两档合并 (前向时 EDM sigma 随机, 合并更接近部署分布);
标准化参数 (mu/sd) 与 LR 权重一并保存。

输出: <cic_match>/ich_d_frozen_lr.npz {w(16), b(1), mu(16), sd(16), meta_json}
  + 打印: 全量训练内 AUC (合并/分 sigma) + dup/bg sigmoid 均值分离
  + 手算 sigmoid(w·(x-mu)/sd+b) 与 sklearn predict_proba 逐位对账 (装载正确性)。
CPU 本机可跑。
"""
import argparse
import json
import os

import numpy as np

import selfcase_g1 as G1

CIC_DIR = '/data/datasets/gagi/eve_v2_outputs/selfcase/cic_match'
CACHE_DEFAULT = os.path.join(CIC_DIR, 'cic_cell_features.npz')
OUT_DEFAULT = os.path.join(CIC_DIR, 'ich_d_frozen_lr.npz')
A_BLOCKS = ['block10', 'block12', 'block16']
SIGMAS = [0.2, 0.4]


def build_matrix(d, sigma):
    """v1 B 配置 16 维, 列序 = [b10, b12, b16] novelty + [b22] cic。"""
    return np.hstack([d[f'X_{b}_{sigma}'] for b in A_BLOCKS] + [d[f'X_cic_{sigma}']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=CACHE_DEFAULT)
    ap.add_argument('--out', default=OUT_DEFAULT)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta_in = json.loads(str(d['meta']))
    assert meta_in['cic_block'] == 'block22', meta_in
    y1 = d['y']
    Xs = {s: build_matrix(d, s) for s in SIGMAS}
    X = np.vstack([Xs[s] for s in SIGMAS])
    y = np.concatenate([y1] * len(SIGMAS))
    print(f'[ich-d fit] {meta_in["n_cases"]} cases, rows/sigma={len(y1)} '
          f'({int(y1.sum())} dup / {int((1 - y1).sum())} bg), merged rows={len(y)}, '
          f'dims={X.shape[1]}')

    mu, sd = X.mean(0), X.std(0) + 1e-8
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit((X - mu) / sd, y)
    w = clf.coef_.ravel().astype(np.float64)
    b = clf.intercept_.astype(np.float64)

    # ---- 对账 1: 手算 sigmoid(w·z+b) 与 sklearn predict_proba 逐位一致 ----
    z = (X - mu) / sd
    p_manual = 1.0 / (1.0 + np.exp(-(z @ w + b[0])))
    p_sklearn = clf.predict_proba(z)[:, 1]
    assert np.allclose(p_manual, p_sklearn, atol=1e-10), '手算与 sklearn 不一致'
    print('[ich-d fit] manual sigmoid == sklearn predict_proba (atol 1e-10) OK')

    # ---- 对账 2: 训练内 AUC + dup/bg sigmoid 分离 (固化后运行时应复现该量级) ----
    stats = {}
    auc_all = G1.rank_auc(p_manual[y == 1], p_manual[y == 0])
    stats['merged'] = dict(train_auc=auc_all,
                           m_dup=float(p_manual[y == 1].mean()),
                           m_bg=float(p_manual[y == 0].mean()))
    print(f'[ich-d fit] merged: train_auc={auc_all:.4f} '
          f'M_dup={stats["merged"]["m_dup"]:.4f} M_bg={stats["merged"]["m_bg"]:.4f} '
          f'sep={stats["merged"]["m_dup"] - stats["merged"]["m_bg"]:+.4f}')
    for s in SIGMAS:
        zs = (Xs[s] - mu) / sd
        p = 1.0 / (1.0 + np.exp(-(zs @ w + b[0])))
        auc = G1.rank_auc(p[y1 == 1], p[y1 == 0])
        stats[str(s)] = dict(train_auc=auc, m_dup=float(p[y1 == 1].mean()),
                             m_bg=float(p[y1 == 0].mean()))
        print(f'[ich-d fit] sigma={s}: train_auc={auc:.4f} '
              f'M_dup={stats[str(s)]["m_dup"]:.4f} M_bg={stats[str(s)]["m_bg"]:.4f} '
              f'sep={stats[str(s)]["m_dup"] - stats[str(s)]["m_bg"]:+.4f} '
              f'(LOO 参考 B_16d|{s})')

    meta = dict(cache=args.cache, sigmas=SIGMAS, feature_order=A_BLOCKS + ['cic_block22'],
                n_cases=meta_in['n_cases'], n_rows_merged=int(len(y)), stats=stats,
                loo_ref_b16d_04=0.8261)
    np.savez(args.out, w=w, b=b, mu=mu, sd=sd,
             meta=np.array(json.dumps(meta), dtype=object))
    print(f'[ich-d fit] frozen LR -> {args.out}')
    print('ICH_D_FIT_OK')


if __name__ == '__main__':
    main()
