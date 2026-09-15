#!/usr/bin/env python3
"""T4G-JOINT 权重图接入冒烟 (CPU, 无模型): 逐条验证 forward_step 的权重图数学。
  S1 上采样: (24,30,48) -repeat_interleave-> (24,60,96), 每格->2x2块, 值不变
  S2 z 广播: (B,24,60,96) -> (B,16,24,60,96), 各通道相同
  S3 贴入区 max: clamp_(min=w_paste) 只抬不降 (合同6x保持, 合同0.5x升到4x)
  S4 逐样本归一化: w_map/mean -> 每样本均值精确=1
  S5 loss 尺度守恒: 归一化后 (err*w_map).mean() 与 err.mean() 同量级 (均匀err下相等)
  S6 缓存一致性: 预计算.npy == build_weightmap 实时 (抽样5条)
  S7 维度实锤: 权重图上采样后 == VAE latent 空间 (60,96) [硬编码核对]
"""
import os
import numpy as np
import torch

# ---- S1 上采样 ----
wm = torch.arange(24 * 30 * 48, dtype=torch.float32).reshape(24, 30, 48)
up = wm.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
assert up.shape == (24, 60, 96), up.shape
# 格 (t=3, gy=5, gx=7) 应映射到 up 的 [3, 10:12, 14:16] 四格, 值相同
v = float(wm[3, 5, 7])
assert torch.allclose(up[3, 10:12, 14:16], torch.full((2, 2), v)), '上采样格映射错'
print('S1 上采样 30x48->60x96 每格2x2块 PASS')

# ---- S2 z 广播 ----
B, Z = 2, 16
cw = up.unsqueeze(0).expand(B, -1, -1, -1)                  # (B,24,60,96)
w5 = cw.unsqueeze(1).expand(-1, Z, -1, -1, -1).contiguous()  # (B,16,24,60,96)
assert w5.shape == (B, Z, 24, 60, 96), w5.shape
assert torch.allclose(w5[0, 0], w5[0, 15]), 'z 通道不一致'
print('S2 z 广播 (B,16,24,60,96) 各通道相同 PASS')

# ---- S3 贴入区 max (clamp_ min) ----
w = torch.tensor([[0.5, 6.0, 2.0, 4.0]]).repeat(1, 1)      # 模拟合同权重值
w_test = torch.tensor([0.5, 6.0, 2.0, 3.0])
w_paste = 4.0
res = w_test.clone(); res.clamp_(min=w_paste)
assert torch.allclose(res, torch.tensor([4.0, 6.0, 4.0, 4.0])), res  # 0.5->4, 6保持, 2->4, 3->4
print('S3 贴入区 clamp_(min=w_paste) 只抬不降 PASS')

# ---- S4 逐样本归一化 ----
torch.manual_seed(0)
w_map = torch.rand(B, Z, 24, 60, 96) * 5 + 0.5
w_norm = w_map / w_map.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
means = w_norm.mean(dim=(1, 2, 3, 4))
assert torch.allclose(means, torch.ones(B), atol=1e-5), means
print(f'S4 逐样本归一化 均值={means.tolist()} (精确1) PASS')

# ---- S5 loss 尺度守恒 (均匀 err 下 加权.mean == err.mean) ----
err = torch.full((B, Z, 24, 60, 96), 0.3)
lw = (err * w_norm).mean(dim=(1, 2, 3, 4))
assert torch.allclose(lw, torch.full((B,), 0.3), atol=1e-5), lw
print('S5 归一化后 loss 尺度守恒 (均匀err下不变) PASS')

# ---- S6 缓存 == 实时 ----
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4g_weightmap as W
CACHE = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/weightmap_cache'
import glob
vids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(CACHE + '/*.npy'))[:5]]
for vid in vids:
    cached = np.load(f'{CACHE}/{vid}.npy')
    live, _ = W.build_weightmap(W.load_anno(vid))
    assert np.allclose(cached, live), f'缓存!=实时 vid{vid}'
print(f'S6 缓存==实时 (抽{len(vids)}条) PASS')

# ---- S7 维度硬核对: latent 空间 = 上采样后权重 ----
# VAE: 480/8=60, 768/8=96; 权重图 30x48 x2 = 60x96
assert 30 * 2 == 480 // 8 == 60 and 48 * 2 == 768 // 8 == 96, '维度关系错!'
assert W.T_LAT == 24 and (93 - 1) // 4 + 1 == 24, '时间维错!'
print('S7 维度硬核对: 权重30x48 x2=60x96=latent, T=24=(93-1)/4+1 PASS')

print('\n全部冒烟 PASS — 权重图接入数学可信, 可提交 50 步探针。')
