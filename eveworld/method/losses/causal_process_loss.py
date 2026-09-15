#!/usr/bin/env python3
"""EVE 方法 · 因果过程忠实损失(方法 §五,集群侧 torch)。

组合三项(可各自开关做消融):
  L_cpc   : 真实过程去噪误差 < 作弊过程(margin ranking)   —— 方法 C
  L_dyn   : 生成/真实转移在 LAM 下可解释(动力学一致)      —— 方法 A
  L_prog  : 单调进度 + 接触门控                              —— 方法 B

与 GigaWorld-0 的 EDM/去噪训练解耦:本模块只消费
  - denoise_error_fn(z, cond): 给定 latent 序列与条件,返回标量去噪误差(用现有 EDMLoss 包一层)
  - lam: 可选的 LatentActionModel(见 lam/latent_action_model.py)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .lazy_negatives import make_negatives


def cpc_loss(denoise_error_fn, z, cond, neg_kinds, margin=0.1):
    """margin ranking:让作弊过程的去噪误差比真实高至少 margin。"""
    e_pos = denoise_error_fn(z, cond)                      # (B,)
    negs = make_negatives(z, neg_kinds)
    loss = z.new_zeros(())
    logs = {}
    for k, zneg in negs.items():
        e_neg = denoise_error_fn(zneg, cond)
        l = F.relu(margin - (e_neg - e_pos)).mean()
        loss = loss + l
        logs[f"cpc/{k}"] = float(l.detach())
    return loss / max(1, len(negs)), logs


def dyn_consistency_loss(lam, z):
    """LAM 前向重建误差:sum_t ||z_{t+1} - forward(z_t, inv(z_t,z_{t+1}))||。
    真实视频上作为自监督;生成视频上作为反捷径正则(偷懒转移误差高)。"""
    B, C, T, H, W = z.shape
    zt, ztp = z[:, :, :-1], z[:, :, 1:]
    a = lam.inverse(zt, ztp)              # 潜在动作
    pred = lam.forward_dyn(zt, a)         # 预测下一帧
    return F.mse_loss(pred, ztp)


def progress_loss(progress_head, z, contact_bin=None):
    """单调进度 + 接触门控。progress_head(z)->(B,T) in [0,1]。"""
    B, C, T, H, W = z.shape
    p = progress_head(z)                                   # (B,T)
    # 单调性:惩罚下降
    mono = F.relu(p[:, :-1] - p[:, 1:]).mean()
    loss = mono
    logs = {"prog/mono": float(mono.detach())}
    # 接触门控:接触前进度不应过高
    if contact_bin is not None:
        gate = torch.zeros_like(p)
        for b in range(B):
            cb = int(contact_bin[b] * (T - 1))
            gate[b, :cb] = 1.0
        pre = F.relu(p - 0.3) * gate      # 接触前 p>0.3 受罚
        pg = pre.mean()
        loss = loss + pg
        logs["prog/gate"] = float(pg.detach())
    return loss, logs


def total_loss(cfg, denoise_error_fn, z, cond, lam=None, progress_head=None, contact_bin=None):
    """按 cfg 权重组合。cfg: dict(w_cpc,w_dyn,w_prog,neg_kinds,margin)。"""
    total = z.new_zeros(()); logs = {}
    if cfg.get("w_cpc", 0) > 0:
        l, lg = cpc_loss(denoise_error_fn, z, cond, cfg.get("neg_kinds", ["teleport", "excision", "freeze_jump"]), cfg.get("margin", 0.1))
        total = total + cfg["w_cpc"] * l; logs.update(lg)
    if cfg.get("w_dyn", 0) > 0 and lam is not None:
        l = dyn_consistency_loss(lam, z); total = total + cfg["w_dyn"] * l; logs["dyn"] = float(l.detach())
    if cfg.get("w_prog", 0) > 0 and progress_head is not None:
        l, lg = progress_loss(progress_head, z, contact_bin); total = total + cfg["w_prog"] * l; logs.update(lg)
    logs["loss_causal_total"] = float(total.detach())
    return total, logs
