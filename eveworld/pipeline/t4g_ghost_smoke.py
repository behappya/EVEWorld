#!/usr/bin/env python3
"""T4G-GHOST/EMPTY-MAP 冒烟 (CPU, 无模型): 用合成 anno+motion+dev 验证探针逻辑本身。
硬检验:
  S1 build_zones: B_pre 只含到达前+静止 B 格; B_post 只含到达后; 目标邻域被排除
  S2 时间纹 bins: 递增 dev 场 -> bins 单调递增 (探针能看见 ramp)
  S3 path_dt: 距访问时刻越近 dev 越高的场 -> dt1 > dt8 (探针能测涂抹)
  S4 cell_dev: (z,T,60,96) -> (T,30,48) 形状 + 数值正确 (常数场)
  S5 window_static: 运动窗口豁免正确 (手臂扫过帧 t 时 [t-D,t+D] 全豁免)
  S6 zone_means 空 zone -> nan 不崩
"""
import numpy as np
import torch

from t4g_ghost_probe import build_zones, zone_means, cell_dev, T_LAT, H_LAT, W_LAT
from t4g_empty_map import window_static

# ---- 合成 anno: 物体 (25,5)->(12,35) 直线, t_arr=16, B 框 [10..14]x[33..37] ----
t_arr = 16
per = []
for t in range(T_LAT):
    f = min(1.0, t / t_arr)
    cy, cx = int(25 - 13 * f), int(5 + 30 * f)
    per.append({'target_cell': [cy, cx],
                'b_cells': [[y, x] for y in range(10, 15) for x in range(33, 38)],
                'detected': True})
anno = {'t_arrival': t_arr, 'per_lat_frame': per, 'distractor_cells': [[5, 5]],
        'a_cells': [[y, x] for y in range(23, 28) for x in range(3, 8)]}

# ---- 合成 motion: 物体路径格在经过时刻运动高, 其余静止 ----
motion = np.ones((T_LAT, H_LAT, W_LAT), np.float32) * 0.5
for t in range(T_LAT):
    c = per[t]['target_cell']
    motion[t, max(0, c[0] - 1):c[0] + 2, max(0, c[1] - 1):c[1] + 2] = 20.0

zones = build_zones(anno, motion)

# S1: B_pre 帧域正确
assert zones['B_pre'], 'B_pre 空!'
assert all(t < 0.75 * t_arr + 1 for (t, y, x) in zones['B_pre']), 'B_pre 混入晚帧'
assert all(t >= t_arr for (t, y, x) in zones['B_post']), 'B_post 混入早帧'
assert all((y, x) in {(yy, xx) for yy in range(10, 15) for xx in range(33, 38)}
           for (t, y, x) in zones['B_pre']), 'B_pre 出框'
# 到达时刻目标在 B 内, 其邻域格不应出现在 B_pre (cur_nb 排除)
for (t, y, x) in zones['B_pre']:
    c = per[t]['target_cell']
    assert max(abs(y - c[0]), abs(x - c[1])) > 2, f'B_pre 含目标邻域 t={t}'
print('S1 build_zones 分区正确 PASS')

# S2: 递增场 -> bins 递增
dev = np.zeros((T_LAT, H_LAT, W_LAT), np.float32)
for t in range(T_LAT):
    dev[t] = t / T_LAT
zm = zone_means(dev, zones)
b = [zm[f'Bbin_bin{i}'] for i in range(4) if zm[f'Bbin_bin{i}'] == zm[f'Bbin_bin{i}']]
assert len(b) >= 3 and all(b[i] < b[i + 1] for i in range(len(b) - 1)), f'bins 非递增: {b}'
print(f'S2 时间纹递增可见 PASS {["%.3f" % x for x in b]}')

# S3: 靠近访问时刻 dev 高 -> dt 剖面递减
dev2 = np.zeros_like(dev)
vis = {}
for t in range(T_LAT):
    c = per[t]['target_cell']
    vis.setdefault((c[0], c[1]), []).append(t)
for (cy, cx), ts in vis.items():
    for t in range(T_LAT):
        dt = min(abs(t - x) for x in ts)
        dev2[t, cy, cx] = max(0.0, 1.0 - 0.12 * dt)
zm2 = zone_means(dev2, zones)
p = [zm2[f'path_dt{d}'] for d in range(1, 9) if zm2[f'path_dt{d}'] == zm2[f'path_dt{d}']]
assert len(p) >= 4 and p[0] > p[-1], f'path 剖面非递减: {p}'
print(f'S3 路径涂抹剖面可测 PASS dt1={p[0]:.3f} dt_last={p[-1]:.3f}')

# S4: cell_dev 形状+数值
x0 = torch.zeros(16, T_LAT, 60, 96)
xh = torch.full((16, T_LAT, 60, 96), 0.25)
d = cell_dev(xh, x0)
assert d.shape == (T_LAT, H_LAT, W_LAT) and abs(float(d.mean()) - 0.25) < 1e-6, d.shape
print('S4 cell_dev 池化正确 PASS')

# S5: window_static 豁免
mo = np.zeros((T_LAT, 4, 4), np.float32)
mo[10, 1, 1] = 99.0                       # t=10 该格动
for delta in (0, 2):
    import t4g_empty_map as EM
    old = EM.T_LAT
    pm = window_static(mo, theta=3.0, delta=delta)
    banned = [t for t in range(T_LAT) if not pm[t, 1, 1]]
    assert banned == list(range(10 - delta, 10 + delta + 1)), (delta, banned)
assert window_static(mo, 3.0, 1)[:, 0, 0].all(), '静止格被误豁免'
print('S5 window_static 时间余量豁免正确 PASS')

# S6: 空 zone nan
zm3 = zone_means(dev, {'empty': []})
assert zm3['empty'] != zm3['empty'], 'nan 处理失败'
print('S6 空 zone nan PASS')

print('\n全部冒烟 PASS — 探针逻辑自身可信, 可提交 GPU 作业。')
