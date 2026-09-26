#!/usr/bin/env python3
"""T4G-WEIGHTMAP: 三段合同 loss 权重图 (44 号阶段2 v3)。

纯查表 (零训练时检测): vid -> t4g_anno JSON -> 24x30x48 权重格图。
区域 (逐格取 max, 之后逐样本归一化到均值 1):
  物体轨迹区   target_cell ±obj_r 格 (逐帧)          w_obj=4.0   治 morph/形变
  抓取窗       t_grasp±tw 帧 × A位±2格               w_trans=6.0 段1→2 交接(消失案发)
  放置窗       t_release±tw 帧 × B位±2格             w_trans=6.0 段3 粘手
  该空区       到达前B区 + 拿走后A原位               w_empty=2.0 守住空的精度
  背景         其余                                  w_bg=0.5    全覆盖但低配
漏检帧 detected=false -> 该帧全 1.0 (宁可少重点不打错)。
gate_enabled=false -> 关 B 相关(放置窗/该空B), 物体轨迹/抓取窗照常。
build_weightmap 返回 (权重图, 段位); build_weightmap_regions 是其等价实现, 额外返回
逐区域布尔掩码 (REGION_KEYS: obj/trans/b_dest/grip/empty/distractor/bg), 供权重图变体复用。

段位估计器 (纯轨迹, 无 GDINO):
  t_grasp   = target_cell 首次离初始位 >2 格
  t_release = 到达 B 后 (t>=t_arrival) 位置连续 stab_k 帧稳定(<=1格抖) 的首帧; 无则末段
"""
import json
import os

import numpy as np

ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
ASSETS_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets'
GRIPPER_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/gripper_anno'
T_LAT, H_LAT, W_LAT = 24, 30, 48
CELL_PX = 16

W_OBJ, W_TRANS, W_EMPTY, W_BG = 4.0, 6.0, 2.0, 0.5
W_BDEST, W_GRIP = 3.0, 3.0            # B 目的地到达前该空(高危终态生成) / 空爪该空(高危生成形变)
W_DISTRACTOR = 2.0                    # 干扰物 permanence: 全程该静止不变 (防形变/消失)
WIN_T, STAB_K = 2, 3
OBJ_MARGIN, TRANS_MARGIN = 1, 2      # 物体框外扩格数 (物体区+1, 抓放窗+2含夹爪)
DEFAULT_HALF = (2, 2)                 # 无框回退半尺寸 (5x5格)

# 区域组分解 (供 build_weightmap_regions / 权重图变体使用)
MARK = 1.0                            # 区域掩码戳印值 (掩码只用于"该格属哪组")
REGION_KEYS = ('obj', 'trans', 'b_dest', 'grip', 'empty', 'distractor', 'bg')
REGION_VALUES = dict(obj=W_OBJ, trans=W_TRANS, empty=W_EMPTY, b_dest=W_BDEST,
                     grip=W_GRIP, distractor=W_DISTRACTOR, bg=W_BG)
# 合成时的落值顺序: 级别升序 (同格被多组标记时高档覆盖低档, 等价于逐格 max)
REGION_ORDER = tuple(sorted(REGION_KEYS, key=lambda k: REGION_VALUES[k]))


def smooth_traj(anno, jump_thr=6):
    """清洗逐帧 target_cell: 中值滤波(宽3) + 离群剔除(与两邻居都 >jump_thr 的置 None)。
    返回 (traj 清洗后 list[cell|None], reliable list[bool], residual_jumps)。"""
    per = anno['per_lat_frame']
    raw = [tuple(p['target_cell']) if p['target_cell'] else None for p in per]
    n = len(raw)
    # 离群剔除: 与前后有效邻居都远的单点
    cleaned = list(raw)
    for t in range(n):
        if raw[t] is None:
            continue
        nb = [raw[t + d] for d in (-2, -1, 1, 2) if 0 <= t + d < n and raw[t + d]]
        if nb and all(abs(raw[t][0] - c[0]) + abs(raw[t][1] - c[1]) > jump_thr for c in nb):
            cleaned[t] = None                       # 孤立跳点 = 检测误配, 剔除
    # 中值滤波 (宽3, 分坐标)
    sm = list(cleaned)
    for t in range(n):
        win = [cleaned[t + d] for d in (-1, 0, 1) if 0 <= t + d < n and cleaned[t + d]]
        if len(win) >= 2:
            sm[t] = (int(np.median([c[0] for c in win])), int(np.median([c[1] for c in win])))
    reliable = [c is not None for c in sm]
    jumps = sum(1 for t in range(1, n) if sm[t] and sm[t - 1]
                and abs(sm[t][0] - sm[t - 1][0]) + abs(sm[t][1] - sm[t - 1][1]) > jump_thr)
    return sm, reliable, jumps


def traj_destination(traj, t_release):
    """物体落点 = t_release 起连续几帧的中值位置 (回退末段有效帧中值)。"""
    lo = t_release if t_release is not None else T_LAT - 4
    seg = [c for c in traj[max(0, lo):] if c]
    if not seg:
        seg = [c for c in traj if c][-4:]
    if not seg:
        return None
    return (int(np.median([c[0] for c in seg])), int(np.median([c[1] for c in seg])))


def stable_b_region(per, t_arr, min_frac=0.4):
    """到达前 B 区稳定众数: 出现在 >=min_frac 到达前帧的 b_cells (滤逐帧检测跳变)。"""
    pre = range(max(1, t_arr))
    cnt = {}
    n = 0
    for t in pre:
        n += 1
        for c in per[t]['b_cells']:
            cnt[tuple(c)] = cnt.get(tuple(c), 0) + 1
    if n == 0:
        return set()
    return {c for c, k in cnt.items() if k >= min_frac * n}


def robust_origin(traj, k=6):
    """前 k 个有效帧的中值作原位 (抗单点); 要求这些点空间一致 (max两两距<=4) 才可信。"""
    early = [c for c in traj[:k] if c]
    if len(early) < 2:
        return None, False
    ok = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in early for b in early) <= 4
    origin = (int(np.median([c[0] for c in early])), int(np.median([c[1] for c in early])))
    return origin, ok


def estimate_t_grasp(anno, sustain=2, thr=2):
    """t_grasp = target_cell 离初始位 >thr 格且**持续 sustain 帧不回**的首帧 (滤单帧抖动)。"""
    per = anno['per_lat_frame']
    c0 = anno.get('target_cell_0') or (per[0]['target_cell'] if per[0]['target_cell'] else None)
    if c0 is None:
        return None
    for t in range(1, T_LAT - sustain + 1):
        win = [per[t + k]['target_cell'] for k in range(sustain)]
        if all(c is not None and abs(c[0] - c0[0]) + abs(c[1] - c0[1]) > thr for c in win):
            return t
    return None


def _stamp(wmap, t, cells, val, r=0):
    for (gy, gx) in cells:
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy, xx = gy + dy, gx + dx
                if 0 <= yy < H_LAT and 0 <= xx < W_LAT:
                    wmap[t, yy, xx] = max(wmap[t, yy, xx], val)


def _stamp_box(wmap, t, cell, half, val, margin=0):
    """以 cell 为中心, 盖 (2*(hh+margin)+1) x (2*(hw+margin)+1) 的物体尺寸框。"""
    gy, gx = cell
    hh, hw = half[0] + margin, half[1] + margin
    for yy in range(max(0, gy - hh), min(H_LAT, gy + hh + 1)):
        for xx in range(max(0, gx - hw), min(W_LAT, gx + hw + 1)):
            wmap[t, yy, xx] = max(wmap[t, yy, xx], val)


def load_gripper(vid):
    """读夹爪检测缓存 -> per_lat_gripper (list[list[{center,cells,score}]]) 或 None。"""
    fp = os.path.join(GRIPPER_DIR, f'{vid}.json')
    if os.path.exists(fp):
        return json.load(open(fp)).get('per_lat_gripper')
    return None


def object_halfsize(vid):
    """从 aug_assets 帧0 框读物体半尺寸 (格); 无框回退 DEFAULT_HALF。"""
    fp = os.path.join(ASSETS_DIR, f'{vid}.npz')
    if os.path.exists(fp):
        d = np.load(fp)
        if 'box' in d:
            y0, x0, y1, x1 = [float(v) for v in d['box']]
            hh = max(1, int(round((y1 - y0) / 2 / CELL_PX)))
            hw = max(1, int(round((x1 - x0) / 2 / CELL_PX)))
            return (hh, hw)
    return DEFAULT_HALF


def _compose_weightmap(regions):
    """把逐区域掩码合成权重图: 按 REGION_VALUES 级别升序落值
    (同格被多组标记时高档覆盖低档, 与老实现逐格 max 等价)。"""
    w = np.full((T_LAT, H_LAT, W_LAT), W_BG, np.float32)
    for k in REGION_ORDER:
        w[regions[k]] = np.float32(REGION_VALUES[k])
    return w


def build_weightmap_regions(anno, t_grasp=None, t_release=None):
    """返回 (w, seg, regions): 权重图 + 段位 dict + 逐区域布尔掩码。

    regions: dict[k] -> bool (T_LAT,H_LAT,W_LAT), k ∈ REGION_KEYS:
      obj   物体轨迹区 (含降级锚)      trans 抓取窗/放置窗
      b_dest 到达前 B 目的地该空       grip  空爪检测框
      empty 拿走后物体原位该空         distractor 干扰物
      bg    其余 (未被任何组标记)
    同一格可被多组标记 (老实现逐格取 max); 各区域级别互不冲突, 升序落值即等价。
    轨迹先清洗(smooth_traj); 不可靠视频降级(只标 B该空+原位, 不标物体轨迹/转换窗)。"""
    per = anno['per_lat_frame']
    gate = anno.get('gate_enabled', False)
    t_arr = anno.get('t_arrival')
    if t_arr == 0:                                   # t_arr=0 退化标注, 关 B 相关
        gate = False; t_arr = None

    traj, reliable, jumps = smooth_traj(anno)
    origin, origin_ok = robust_origin(traj)
    # 可靠性: 清洗后残余跳变少 + 原位一致 -> 允许标物体轨迹/转换窗
    traj_reliable = origin_ok and jumps <= 3

    # t_grasp/t_release 用清洗轨迹重估 (离 robust origin 持续位移)
    if traj_reliable and origin is not None:
        tg = None
        for t in range(1, T_LAT - 1):
            win = [traj[t + k] for k in range(2)]
            if all(c is not None and abs(c[0] - origin[0]) + abs(c[1] - origin[1]) > 2 for c in win):
                tg = t; break
        t_grasp = tg if t_grasp is None else t_grasp
    else:
        t_grasp = None
    if t_release is None:
        t_release = estimate_t_release_traj(traj, t_arr, t_grasp) if traj_reliable else None

    m = {k: np.zeros((T_LAT, H_LAT, W_LAT), np.float32) for k in REGION_KEYS if k != 'bg'}
    half = object_halfsize(anno['vid'])          # 物体真实框半尺寸 (格)

    # --- 目的地 B: 优先用【物体轨迹落点】(gate 无关, 覆盖模糊容器); 回退 GDINO 稳定众数 ---
    dest = traj_destination(traj, t_release) if traj_reliable else None
    b_cut = t_release if t_release is not None else t_arr     # 物体到达 B 的时刻

    # --- 该空区 ---
    # B 到达前该空 (W_BDEST=3x, 终态提前生成高发): 目的地在物体到达前一直该空
    if dest is not None and b_cut is not None:
        for t in range(0, b_cut):
            _stamp_box(m['b_dest'], t, dest, half, MARK)
    elif gate and t_arr is not None:                         # 回退: GDINO B 众数
        stable_b = stable_b_region(per, t_arr)
        for t in range(t_arr):
            _stamp(m['b_dest'], t, list(stable_b), MARK)
    # 拿走后物体原位该空
    if traj_reliable and t_grasp is not None and origin is not None:
        for t in range(t_grasp + WIN_T, T_LAT):
            _stamp_box(m['empty'], t, origin, half, MARK)

    # --- 空爪 (W_GRIP=3x): 逐帧检测的爪框 (防生成/形变) ---
    grip = load_gripper(anno['vid'])
    if grip is not None:
        for t in range(min(T_LAT, len(grip))):
            for g in grip[t]:
                _stamp(m['grip'], t, [tuple(c) for c in g['cells']], MARK)

    # --- 干扰物 permanence (恒可标, 不依赖轨迹): 全程该静止不变 ---
    for c in anno.get('distractor_cells', []):
        for t in range(T_LAT):
            _stamp_box(m['distractor'], t, tuple(c), half, MARK)

    # --- 降级视频最小物体锚: 帧0原位标物体(段1静止段), 保底监督 ---
    if not traj_reliable:
        o0 = anno.get('target_cell_0')
        if o0:
            for t in range(min(4, T_LAT)):       # 前几帧物体必在原位
                _stamp_box(m['obj'], t, tuple(o0), half, MARK, margin=OBJ_MARGIN)

    if traj_reliable:
        # --- 物体轨迹区 (w_obj): 清洗轨迹逐帧, 物体真实框大小 +margin ---
        for t in range(T_LAT):
            if traj[t]:
                _stamp_box(m['obj'], t, traj[t], half, MARK, margin=OBJ_MARGIN)
        # --- 抓取窗 (w_trans): t_grasp±WIN_T × 原位 & 当帧物体 (物体框+夹爪余量) ---
        if t_grasp is not None:
            for t in range(max(0, t_grasp - WIN_T), min(T_LAT, t_grasp + WIN_T + 1)):
                if origin:
                    _stamp_box(m['trans'], t, origin, half, MARK, margin=TRANS_MARGIN)
                if traj[t]:
                    _stamp_box(m['trans'], t, traj[t], half, MARK, margin=TRANS_MARGIN)
        # --- 放置窗 (w_trans): t_release±WIN_T × 落点 & 当帧物体 (gate 无关) ---
        if t_release is not None:
            for t in range(max(0, t_release - WIN_T), min(T_LAT, t_release + WIN_T + 1)):
                if dest is not None:
                    _stamp_box(m['trans'], t, dest, half, MARK, margin=TRANS_MARGIN)
                if traj[t]:
                    _stamp_box(m['trans'], t, traj[t], half, MARK, margin=TRANS_MARGIN)

    regions = {k: (v > 0.5) for k, v in m.items()}
    covered = np.zeros((T_LAT, H_LAT, W_LAT), bool)
    for v in regions.values():
        covered |= v
    regions['bg'] = ~covered                                  # 其余 = 未被任何组标记的格

    seg = dict(t_grasp=t_grasp, t_release=t_release, t_arrival=t_arr, gate=gate,
               traj_reliable=traj_reliable, jumps=jumps, origin=origin, traj=traj)
    return _compose_weightmap(regions), seg, regions


def build_weightmap(anno, t_grasp=None, t_release=None):
    """返回 (T_LAT,30,48) float32 权重图 + 段位 dict (向后兼容薄包装)。
    轨迹先清洗(smooth_traj); 不可靠视频降级(只标 B该空+原位, 不标物体轨迹/转换窗)。
    需要逐区域掩码时用 build_weightmap_regions。"""
    w, seg, _ = build_weightmap_regions(anno, t_grasp=t_grasp, t_release=t_release)
    return w, seg


def estimate_t_release_traj(traj, t_arr, t_grasp):
    """清洗轨迹版: 到达后 (t>=lo) 连续 STAB_K 帧稳定的首帧。
    lo 必在 t_grasp 之后 (松手不可能早于抓取), 且要求该稳定位【远离原位】(否则是抓取前的静止)。"""
    cand = [x for x in (t_arr, (t_grasp + 3) if t_grasp is not None else None) if x is not None]
    lo = max(cand) if cand else T_LAT // 2
    origin = None
    for c in traj[:6]:
        if c:
            origin = c; break
    for t in range(max(lo, 1), T_LAT - STAB_K + 1):
        win = [traj[t + k] for k in range(STAB_K)]
        if all(c is not None for c in win):
            ref = win[0]
            stable = all(abs(c[0] - ref[0]) + abs(c[1] - ref[1]) <= 1 for c in win)
            far = origin is None or abs(ref[0] - origin[0]) + abs(ref[1] - origin[1]) > 3
            if stable and far:            # 停稳 且 已远离原位 = 真放置点
                return t
    return None


def load_anno(vid):
    return json.load(open(os.path.join(ANNO_DIR, f'{vid}.json')))
