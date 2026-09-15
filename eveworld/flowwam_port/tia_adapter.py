#!/usr/bin/env python3
"""TIA adapter 移植（源: eveworld/tia_transport/cic_transport_transformer.py）。

数学与 giga 版逐行一致：相邻帧低秩特征局部窗口匹配 -> 注意力搬运(fold/mass
归一) -> 零初始化 output_proj + tanh 残差。仅去掉 giga 的 sequence-parallel
守卫（FlowWAM 训练为数据并行）。

TIAInjection: 以 monkey-patch 方式把 adapter 插到双流 block 循环第 l_star
块之后（作用于 RGB token），并捕获搬运后特征供 L_TIA (eq:cic) 使用；
梯度正常回传，训练/推理通用。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import diffsynth.pipelines.wan_video_dual_stream as ds


class TIAAdapter(nn.Module):
    """Adjacent-frame local matching followed by zero-init residual feedback."""

    def __init__(
        self,
        channels: int,
        rank: int = 64,
        window_radius: int = 3,
        temperature: float = 0.07,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank > channels:
            raise ValueError(f"rank must be in [1, {channels}], got {rank}")
        self.channels = int(channels)
        self.rank = int(rank)
        self.window_radius = int(window_radius)
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        self.input_proj = nn.Linear(channels, rank, bias=False)
        self.output_proj = nn.Linear(rank, channels, bias=False)
        nn.init.zeros_(self.output_proj.weight)
        self._last_stats: dict[str, torch.Tensor] = {}

    @property
    def kernel_size(self) -> int:
        return 2 * self.window_radius + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected (B,T,H,W,D), got {tuple(x.shape)}")
        if x.shape[1] <= 1:
            return x
        batch, frames, height, width, _ = x.shape
        pairs = batch * (frames - 1)
        locations = height * width
        kernel = self.kernel_size
        candidates = kernel * kernel

        # adapter 参数保持 fp32(优化更稳), token 可能是 bf16 -> 显式升精度做匹配
        low = self.input_proj(x.float())
        prev_value = low[:, :-1].permute(0, 1, 4, 2, 3).reshape(pairs, self.rank, height, width)
        curr_value = low[:, 1:].permute(0, 1, 4, 2, 3).reshape(pairs, self.rank, height, width)
        prev_norm = F.normalize(prev_value.float(), dim=1, eps=1e-6)
        curr_norm = F.normalize(curr_value.float(), dim=1, eps=1e-6)

        curr_patches = F.unfold(curr_norm, kernel_size=kernel, padding=self.window_radius)
        curr_patches = curr_patches.reshape(pairs, self.rank, candidates, locations)
        prev_queries = prev_norm.flatten(2)
        logits = torch.einsum("nrl,nrkl->nkl", prev_queries, curr_patches) / self.temperature

        valid = F.unfold(
            torch.ones(1, 1, height, width, device=x.device, dtype=torch.float32),
            kernel_size=kernel, padding=self.window_radius,
        ).reshape(1, candidates, locations) > 0
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1)

        prev_flat = prev_value.float().flatten(2)
        contributions = prev_flat.unsqueeze(2) * attention.unsqueeze(1)
        transported = F.fold(
            contributions.reshape(pairs, self.rank * candidates, locations),
            output_size=(height, width), kernel_size=kernel, padding=self.window_radius,
        )
        mass = F.fold(
            attention, output_size=(height, width),
            kernel_size=kernel, padding=self.window_radius,
        ).clamp_min_(1e-6)
        transported = transported / mass
        delta = (transported - curr_value.float()).permute(0, 2, 3, 1)
        update = self.residual_scale * torch.tanh(self.output_proj(delta))
        update = update.reshape(batch, frames - 1, height, width, self.channels).to(dtype=x.dtype)

        with torch.no_grad():
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=1)
            self._last_stats = {
                "match_peak": attention.max(dim=1).values.detach().mean(),
                "match_entropy": entropy.detach().mean(),
                "update_rms": update.detach().float().square().mean().sqrt(),
                "feature_rms": x[:, 1:].detach().float().square().mean().sqrt(),
            }
        return torch.cat((x[:, :1], x[:, 1:] + update), dim=1)

    @torch.no_grad()
    def sentinel_stats(self) -> dict[str, float]:
        stats = {
            "input_proj_norm": self.input_proj.weight.detach().float().norm().item(),
            "output_proj_norm": self.output_proj.weight.detach().float().norm().item(),
        }
        for key in ("match_peak", "match_entropy", "update_rms", "feature_rms"):
            value = self._last_stats.get(key)
            stats[key] = float("nan") if value is None else float(value.item())
        fr = stats["feature_rms"]
        stats["relative_update"] = stats["update_rms"] / fr if math.isfinite(fr) and fr > 0 else float("nan")
        return stats


class TIAInjection:
    """with TIAInjection(adapter, l_star, grid): model_fn(...) 期间生效。

    在第 l_star 个 block 输出后对 RGB token 应用 adapter，并保存搬运后
    特征 (B,T,GH,GW,D) 于 self.post_features（带梯度，供 L_TIA）。
    """

    def __init__(self, adapter: TIAAdapter, l_star: int, grid: tuple[int, int, int],
                 n_blocks: int = 30):
        self.adapter = adapter
        self.l_star = int(l_star)
        self.grid = grid  # (T, GH, GW)
        self.n_blocks = int(n_blocks)  # 每次 forward 的块数; 计数取模使多次
        #  forward(多步去噪/CFG 双通)共享同一上下文时 adapter 每个 forward 都命中
        self.post_features: torch.Tensor | None = None
        self._orig = None
        self._idx = 0

    def __enter__(self):
        self._orig = ds._dual_stream_block_fn
        self._idx = 0
        self.post_features = None
        inj = self

        def wrapped(block, rtok, ftok, *a, **k):
            r, f = inj._orig(block, rtok, ftok, *a, **k)
            if inj._idx % inj.n_blocks == inj.l_star:
                T, GH, GW = inj.grid
                b = r.shape[0]
                x = r.reshape(b, T, GH, GW, -1)
                x = inj.adapter(x)
                inj.post_features = x
                r = x.reshape(b, T * GH * GW, -1)
            inj._idx += 1
            return r, f

        ds._dual_stream_block_fn = wrapped
        return self

    def __exit__(self, *exc):
        ds._dual_stream_block_fn = self._orig
        return False


def tia_infonce_loss(
    post_features: torch.Tensor,
    gt_cells: list,
    temperature: float = 0.07,
    window_radius: int = 3,
) -> torch.Tensor:
    """论文 eq:cic: 搬运后归一化特征上, 真实下一帧目标格 vs 局部候选 InfoNCE。

    post_features: (B,T,GH,GW,D)（B 维目前按 1 处理, 多样本在外层循环）。
    gt_cells: 长度 T 的 (gy,gx) 或 None（token 网格坐标）。
    """
    x = post_features[0].float()
    T, GH, GW, D = x.shape
    fn = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    losses = []
    r = window_radius
    for t in range(1, T):
        c0, c1 = gt_cells[t - 1], gt_cells[t]
        if c0 is None or c1 is None:
            continue
        q = fn[t - 1, c0[0], c0[1]]
        y0, y1 = max(0, c1[0] - r), min(GH, c1[0] + r + 1)
        x0, x1 = max(0, c1[1] - r), min(GW, c1[1] + r + 1)
        cand = fn[t, y0:y1, x0:x1].reshape(-1, D)
        logits = cand @ q / temperature
        pos = (c1[0] - y0) * (x1 - x0) + (c1[1] - x0)
        losses.append(F.cross_entropy(logits.unsqueeze(0),
                                      torch.tensor([pos], device=x.device)))
    if not losses:
        return post_features.new_zeros(())
    return torch.stack(losses).mean()
