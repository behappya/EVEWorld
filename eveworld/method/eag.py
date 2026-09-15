#!/usr/bin/env python3
"""EVE · EAG 可执行性引导采样(方案27 §四 I2)。

核心: 采样每一步, 用 LAD 对当前 x0 预测(ẑ0, 整段 latent)算"转移可执行性能量"
E(ẑ0), 取 ∂E/∂(noisy latent) 梯度, 以小权重把去噪方向推向"帧间转移合法" ->
物体不能无运动源移动。只改采样, 不重训 backbone。

能量设计(吸收 max 聚合诊断): 偷懒 = 少数几个巨大的非法跳变(转移误差尖峰),
不是平均误差高。所以能量用 transition_error 的 **soft-max(top-k 平均, 可微)**
盯最大跳变, 而非 mean(会被零运动复制帧稀释)。

用法: 见 EAGGuidance.energy / eag_gradient。集成进采样见 pipeline_eag.py。
"""
from __future__ import annotations
import torch


class EAGGuidance:
    """把预训 LAD 包成采样期引导的能量函数 + 梯度。"""

    def __init__(self, lam, topk: int = 3, tau: float = 0.5, weight: float = 0.0):
        """
        lam:   预训好的 LatentActionModel(eval, requires_grad_(False))。
        topk:  soft-max 聚合取前 k 个最大转移误差(盯非法跳变尖峰)。
        tau:   soft-max 温度(越小越接近硬 max)。
        weight: 引导强度(0 = 关闭; 采样时按 sigma 调度)。
        """
        self.lam = lam
        self.topk = topk
        self.tau = tau
        self.weight = weight
        for p in self.lam.parameters():
            p.requires_grad_(False)
        self.lam.eval()

    def energy(self, z0: torch.Tensor) -> torch.Tensor:
        """转移可执行性能量 E(z0). z0:(B,C,T,H,W). 返回标量(B 平均)。
        用 top-k soft 聚合盯最大非法跳变。"""
        te = self.lam.transition_error(z0)            # (B, T-1)
        # soft top-k: 对每个样本取前 k 大转移误差的加权平均(权重 = softmax(te/tau))
        B = te.shape[0]
        k = min(self.topk, te.shape[1])
        topv, _ = te.topk(k, dim=1)                   # (B,k)
        w = torch.softmax(topv / self.tau, dim=1)     # (B,k)
        e_per = (w * topv).sum(dim=1)                 # (B,)  ~ soft-max
        return e_per.mean()

    @torch.enable_grad()
    def gradient(self, z0: torch.Tensor) -> torch.Tensor:
        """返回 ∂E/∂z0, 形状同 z0。仅对 z0 求导(LAD 冻结, 无 transformer 二次前向)。"""
        z0 = z0.detach().float().requires_grad_(True)
        e = self.energy(z0)
        grad, = torch.autograd.grad(e, z0, create_graph=False, retain_graph=False)
        return grad.detach()

    def sigma_weight(self, sigma: float, sigma_max: float = 80.0) -> float:
        """按噪声级调度引导强度: 高噪声早期 x0 预测不可靠, 弱引导; 低噪声后期强引导。
        用 1/(1+sigma) 单调调度。
        注意(实测): 能量地形陡峭, 单步相对步长 w 需在 ~0.01-0.03 量级才稳定下降,
        w>=0.1 会翻过极小点发散。故 self.weight 建议设 0.02-0.05(经 1/(1+sigma) 衰减后
        有效步长落在安全区), 采样多步累积修正。"""
        return self.weight * (1.0 / (1.0 + float(sigma)))


def apply_eag_to_x0(z0_pred: torch.Tensor, guidance: EAGGuidance, sigma: float,
                    sigma_max: float = 80.0) -> torch.Tensor:
    """把 EAG 引导作用在 x0 预测上(universal-guidance on x0):
       z0' = z0 - w(sigma) * ∇_z0 E(z0)。
    返回修正后的 x0。修正 x0 再让 scheduler 走 step, 等价于把去噪方向往"转移合法"推。
    """
    w = guidance.sigma_weight(sigma, sigma_max)
    if w <= 0:
        return z0_pred
    grad = guidance.gradient(z0_pred)                # ∂E/∂z0, 梯度下降降能量
    # 步长: 单位方向 * (w * ||z0|| 的每样本比例)。用相对步长(相对 x0 自身 L2),
    # w 就是"每步最多移动 x0 的百分之几", 天然有界、不发散。
    B = grad.shape[0]
    g = grad.reshape(B, -1)
    gn = g / (g.norm(dim=1, keepdim=True) + 1e-8)     # 单位方向(整体 L2=1)
    gn = gn.reshape_as(grad)
    z0_norm = z0_pred.reshape(B, -1).norm(dim=1).view(B, 1, 1, 1, 1)
    return z0_pred - w * z0_norm * gn                # 沿降能方向移动 w·||z0||
