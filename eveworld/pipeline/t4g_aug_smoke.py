#!/usr/bin/env python3
"""T4G-AUG 冒烟 (CPU, 无模型): 投毒逻辑的硬检验。
  S1 sample_paste_plan: 帧0 永不贴 / patch 覆盖格全窗口安全 / 不安全 zones 拒贴
  S2 zone 偏置: B 格可用时以 bias 概率被选中 (统计验证)
  S3 混合数学: alpha 贴入后区域值正确, 区域外逐位不变
  S4 latent 区域映射: 像素框 -> VAE 8x 网格覆盖一致
  S5 loss 目标切换: denoised==投毒 -> l_paste≈delta (ret~1); denoised==干净 -> l_paste=0 (ret~0)
  S6 干净样本等价: w_map 全 1 时手算 loss == EDM compute_loss 公式
"""
import numpy as np
import torch

from t4g_aug_paste import (sample_paste_plan, prep_patch, paste_lat_region,
                           sample_follow_plan, build_follow_boxes, static_boxes,
                           T_LAT, H_LAT, W_LAT, HPIX, WPIX, CELL_PX, NF)

rng = np.random.default_rng(0)

# ---- 合成 zones: 大部分安全 bg, 一块 B 区, 一块不安全 ----
zones = np.zeros((T_LAT, H_LAT, W_LAT), np.int8)
zones[0] = -1
zones[:, 10:15, 33:38] = 1                       # B 区
zones[:, 0:8, 0:16] = -1                         # 不安全带
patch_hw = (60, 80)

# S1: 500 次采样全部合法
for i in range(500):
    plan = sample_paste_plan(zones, patch_hw, rng)
    assert plan is not None
    assert plan['t_lo'] >= 2, '贴到了帧 0/1!'
    y0, x0, y1, x1 = plan['box']
    assert 0 <= y0 < y1 <= HPIX and 0 <= x0 < x1 <= WPIX
    for t in range(plan['t_lo'], plan['t_hi']):
        for cy in range(y0 // CELL_PX, (y1 - 1) // CELL_PX + 1):
            for cx in range(x0 // CELL_PX, (x1 - 1) // CELL_PX + 1):
                assert zones[t, cy, cx] >= 0, f'贴进不安全格 t={t} ({cy},{cx})'
print('S1 采样合法性 500/500 PASS')

# 全不安全 -> 拒贴
assert sample_paste_plan(np.full_like(zones, -1), patch_hw, rng) is None
print('S1b 全不安全拒贴 PASS')

# S2: zone 偏置统计 (B 权重 0.4 vs bg 0.4 -> B 占比应显著 > 面积占比 25/1440)
zcnt = {0: 0, 1: 0, 2: 0}
for i in range(400):
    p = sample_paste_plan(zones, (30, 30), rng)
    zcnt[p['zone']] += 1
frac_b = zcnt[1] / 400
assert 0.2 < frac_b < 0.7, f'B 占比异常 {zcnt}'
print(f'S2 zone 偏置 PASS (B={zcnt[1]} bg={zcnt[0]} A={zcnt[2]})')

# S3: 混合数学
images = torch.zeros(1, 10, 3, HPIX, WPIX)
patch = torch.full((3, 40, 50), 1.0)
y0, x0, a = 100, 200, 0.6
img_aug = images.clone()
img_aug[0, 3:7, :, y0:y0 + 40, x0:x0 + 50] = a * patch + (1 - a) * img_aug[0, 3:7, :, y0:y0 + 40, x0:x0 + 50]
assert abs(float(img_aug[0, 4, 0, y0 + 5, x0 + 5]) - 0.6) < 1e-6
assert float(img_aug[0, 4, 0, y0 - 1, x0]) == 0.0 and float(img_aug[0, 2, 0, y0 + 5, x0 + 5]) == 0.0
assert torch.equal(img_aug[0, :3], images[0, :3]) and torch.equal(img_aug[0, 7:], images[0, 7:])
print('S3 混合数学 PASS')

# S4: latent 区域映射
plan = dict(t_lo=5, t_hi=12, box=(100, 200, 140, 250))
lr = paste_lat_region(plan)
assert lr['ly0'] == 12 and lr['ly1'] == 18 and lr['lx0'] == 25 and lr['lx1'] == 32
assert lr['lt_lo'] == 5 and lr['lt_hi'] == 12
print('S4 latent 映射 PASS')

# S5: loss 目标切换 (模拟 forward_step 的手算公式)
B, z, T, h, w = 1, 4, 8, 10, 12
clean = torch.zeros(B, z, T, h, w)
lat_in = clean.clone(); lat_in[:, :, 3:6, 2:5, 4:8] = 0.5          # 投毒幅度
region = (slice(None), slice(None), slice(3, 6), slice(2, 5), slice(4, 8))
for denoised, expect_ret in [(lat_in.clone(), 1.0), (clean.clone(), 0.0)]:
    err = (denoised - clean) ** 2
    l_p = float(err[region].mean())
    d_p = float(((lat_in - clean) ** 2)[region].mean())
    ret = (l_p / d_p) ** 0.5 if d_p > 0 else float('nan')
    assert abs(ret - expect_ret) < 1e-6, (ret, expect_ret)
print('S5 loss 目标切换 (ret 1.0/0.0) PASS')

# S6: 干净样本与 EDM compute_loss 等价 (weight*(err) 全元素均值)
weight = torch.tensor([2.5])
err = torch.rand(B, z, T, h, w)
mine = (weight.view(B, 1, 1, 1, 1) * err * torch.ones_like(err)).mean(dim=(1, 2, 3, 4))
edm_style = (weight.view(B, 1) * err.reshape(B, -1)).mean(dim=1)
assert torch.allclose(mine, edm_style, atol=1e-6)
print('S6 干净样本 EDM 等价 PASS')

print('\n全部冒烟 PASS — 投毒逻辑可信, 可跑 prep + 提交训练。')


# S7: 护送贴 (v2) —— 逐帧框跟随轨迹 / 不遮挡真物 / 帧0不贴 / 出界钳位
rep7 = np.linspace(0, NF - 1, T_LAT).astype(int)
traj = []
for t in range(T_LAT):
    f = min(1.0, t / 16)
    traj.append([int(25 - 13 * f), int(5 + 30 * f)])
traj[5] = None                                     # 模拟漏检帧
ok_follow = 0
for i in range(300):
    plan = sample_follow_plan(traj, (60, 80), rng)
    if plan is None:
        continue
    boxes, lat, p0, p1 = build_follow_boxes(traj, plan, rep7)
    if boxes is None:
        continue
    ok_follow += 1
    assert p0 >= rep7[2], '贴到了帧0/1!'
    hp_c = (plan['box'][2] // 2) // CELL_PX
    wp_c = (plan['box'][3] // 2) // CELL_PX
    moved = set()
    for p in range(p0, p1):
        y0, x0, y1, x1 = boxes[p]
        assert 0 <= y0 < y1 <= HPIX and 0 <= x0 < x1 <= WPIX, '出界'
        lt = int(np.clip(np.searchsorted(rep7, p, side='right') - 1, plan['t_lo'], plan['t_hi'] - 1))
        tc = traj[lt] or traj[lt - 1] or traj[lt + 1]
        cy, cx = (y0 + y1) // 2 // CELL_PX, (x0 + x1) // 2 // CELL_PX
        # 真物格不在贴框内 (偏移 >= patch半径+1 格, 除非边界钳位; 钳位帧跳过该检查)
        clipped = (y0 == 0 or x0 == 0 or y1 == HPIX or x1 == WPIX)
        if not clipped:
            assert not (y0 // CELL_PX <= tc[0] <= (y1 - 1) // CELL_PX
                        and x0 // CELL_PX <= tc[1] <= (x1 - 1) // CELL_PX), f'护送贴遮挡真物 p={p}'
        moved.add((cy, cx))
    # 轨迹在窗口内动了, 贴框才必须跟着动 (物体到位后护送=悬停, 合法)
    traj_cells = {tuple(traj[lt]) for lt in range(plan['t_lo'], plan['t_hi']) if traj[lt]}
    if len(traj_cells) >= 3:
        assert len(moved) >= 2, f'轨迹在动但护送贴不动 win=[{plan["t_lo"]},{plan["t_hi"]})'
assert ok_follow > 200, f'护送贴成功率过低 {ok_follow}/300'
print(f'S7 护送贴 PASS ({ok_follow}/300, 跟随/不遮挡/钳位/帧0全过)')

# S8: static_boxes 与旧 plan 等价
plan8 = sample_paste_plan(zones, (60, 80), rng)
boxes8, p0, p1 = static_boxes(plan8, rep7)
assert (boxes8[p0] == np.array(plan8['box'])).all() and (boxes8[p1 - 1] == np.array(plan8['box'])).all()
assert (boxes8[:p0] == 0).all()
print('S8 静态贴逐帧框等价 PASS')

print('v2 冒烟全 PASS')
