#!/usr/bin/env python3
"""EVE 方法 · 偷懒负样本构造(CPC 的核心,方法 §五-C)。

对 clean 视频 latent 序列 z (B,C,T,H,W),确定性地构造"过程作弊"版本 z^-。
这些负样本无需真值动作,是"干预一致性"的可实现近似。

依赖 torch(集群侧)。本文件可被训练循环 import。
纯张量操作,无外部模型依赖。
"""
from __future__ import annotations
import torch


def teleport(z: "torch.Tensor", frac: float = 0.4) -> "torch.Tensor":
    """把终态帧粘到早期帧:目标物体无接触地提前出现。"""
    B, C, T, H, W = z.shape
    out = z.clone()
    k = max(1, int(T * frac))
    out[:, :, :k] = z[:, :, -1:].expand(-1, -1, k, -1, -1)
    return out


def excision(z: "torch.Tensor", lo: float = 0.3, hi: float = 0.7) -> "torch.Tensor":
    """切除"接触->搬运"中间段,首尾拼接后重采样回原长(缺必要阶段)。"""
    B, C, T, H, W = z.shape
    a, b = int(T * lo), int(T * hi)
    kept = torch.cat([z[:, :, :a], z[:, :, b:]], dim=2)
    idx = torch.linspace(0, kept.shape[2] - 1, T, device=z.device).round().long()
    return kept[:, :, idx]


def shuffle_mid(z: "torch.Tensor", lo: float = 0.25, hi: float = 0.85) -> "torch.Tensor":
    """把中间时间块打乱(因果顺序颠倒)。"""
    B, C, T, H, W = z.shape
    a, b = int(T * lo), int(T * hi)
    out = z.clone()
    perm = torch.randperm(b - a, device=z.device) + a
    out[:, :, a:b] = z[:, :, perm]
    return out


def freeze_jump(z: "torch.Tensor", frac: float = 0.55) -> "torch.Tensor":
    """前半冻结初始态,某帧突然跳到接近终态(终态过早+不连续)。"""
    B, C, T, H, W = z.shape
    out = z.clone()
    k = int(T * frac)
    out[:, :, :k] = z[:, :, :1].expand(-1, -1, k, -1, -1)
    out[:, :, k:] = z[:, :, -1:].expand(-1, -1, T - k, -1, -1)
    return out


NEG_FUNCS = {"teleport": teleport, "excision": excision,
             "shuffle": shuffle_mid, "freeze_jump": freeze_jump}


def make_negatives(z: "torch.Tensor", kinds=("teleport", "excision", "freeze_jump")):
    """返回 dict{kind: z^-}。训练时对每个 batch 在线构造 2~3 类。"""
    return {k: NEG_FUNCS[k](z) for k in kinds}


def random_control(z: "torch.Tensor") -> "torch.Tensor":
    """随机置换全部帧 —— 消融用:若随机负样本也一样有效,说明不是因果信号。"""
    B, C, T, H, W = z.shape
    return z[:, :, torch.randperm(T, device=z.device)]
