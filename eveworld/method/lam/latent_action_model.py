#!/usr/bin/env python3
"""EVE 方法 · 潜在动作模型 LAM(方法 §五-A,集群侧 torch)。

Genie 式:inverse(z_t,z_{t+1})->离散潜在动作码;forward_dyn(z_t,a)->z_{t+1}。
在 VAE latent 空间操作(GigaWorld-0 用 Wan VAE,latent_channels=16)。
先在大规模无标注操作视频上自监督预训,再在 GR1 real 微调。
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class ConvEnc(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, cout)

    def forward(self, x):   # x:(N,cin,H,W)
        return self.fc(self.net(x).flatten(1))


class LatentActionModel(nn.Module):
    def __init__(self, latent_ch=16, action_dim=32, codebook=64):
        super().__init__()
        self.inv_enc = ConvEnc(latent_ch * 2, action_dim)     # 从 (z_t,z_{t+1}) 推动作
        self.codebook = nn.Parameter(torch.randn(codebook, action_dim))
        self.fwd = nn.Sequential(                              # (z_t,a) -> z_{t+1}
            nn.Conv2d(latent_ch, 128, 3, 1, 1), nn.SiLU())
        self.act_proj = nn.Linear(action_dim, 128)
        self.head = nn.Conv2d(128, latent_ch, 3, 1, 1)

    def _flatten_time(self, zt, ztp):
        B, C, T, H, W = zt.shape
        return (zt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W),
                ztp.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), (B, T, H, W))

    def inverse(self, zt, ztp):
        """返回量化后的潜在动作 (B,T,action_dim)。"""
        a, b, (B, T, H, W) = self._flatten_time(zt, ztp)
        raw = self.inv_enc(torch.cat([a, b], dim=1))          # (B*T, action_dim)
        # 最近邻量化到码本(直通估计)
        d = torch.cdist(raw, self.codebook)
        idx = d.argmin(1)
        q = self.codebook[idx]
        q = raw + (q - raw).detach()
        return q.view(B, T, -1)

    def forward_dyn(self, zt, a):
        """给定 z_t (B,C,T,H,W) 与动作 a (B,T,action_dim) 预测 z_{t+1}。"""
        B, C, T, H, W = zt.shape
        x = zt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h = self.fwd(x)
        av = self.act_proj(a.reshape(B * T, -1))[:, :, None, None]
        h = h + av
        out = self.head(h)
        return out.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    def transition_error(self, z):
        """逐转移重建误差 (B,T-1):偷懒转移误差高。用于评测第 2 层 / 反捷径正则。"""
        zt, ztp = z[:, :, :-1], z[:, :, 1:]
        pred = self.forward_dyn(zt, self.inverse(zt, ztp))
        return ((pred - ztp) ** 2).mean(dim=(1, 3, 4))
