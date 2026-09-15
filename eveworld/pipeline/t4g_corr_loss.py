#!/usr/bin/env python3
"""EVE · Track4Gen 式对应监督 —— 两道实证 loss 纯函数实现 (43 号 §6 精确公式)。

43 号实证链把方案从"拍脑袋"逼成"有证据": 旧四道 (L_id/L_cnt/L_neg/L_stat) 中,
基于**特征相似度**检测复制的 L_cnt/L_neg 被 E5/E6 实测判死 (复制品变形, 绝对相似度重叠,
sep_dup 全层全 sigma 为负)。E7 用**前后帧自相似变化**重构出 L_change (gap+0.45 判活)。
故本模块**删 L_cnt/L_neg/L_stat(第0帧全局版)**, 只留两道实证支撑的:

  L_id     (block22, 前后帧接力, 治瞬移/形变; E4 EPE0.55)
  L_change (block25, 前后帧自相似, 治复制/重生; E7 gap+0.45, 主病主力)

输入
----
  fn_*: 已单位化特征 (T,H,W,D)  —— 单条视频, 由 prepare() 归一化。L_id 用 block22 特征,
        L_change 用 block25 特征 (两道各取所需层, 43 号 §2 选层)。
  anno: 该视频缓存标注 dict (t4g_anno/<vid>.json)。字段语义 (t4g_detect.py):
        per_lat_frame[t] = {target_cell:[gy,gx]或None(漏检), b_cells:[[gy,gx]...], detected:bool}
        t_arrival(int或None, 目标中心首次入 B 框的 latent 帧), distractor_cells(同类静止干扰物中心格)。

输出 (compute_corr_losses)
--------------------------
  dict(
    losses = {L_id, L_change},          # 0 维 tensor (与 feat 连图, 供反传)
    counts = {n_id, n_change},          # 各自"生效帧数"(参与均值的帧数, 裸值哨兵用)
    gate   = {b_active_frames, b_released_frames},   # 框重叠闸状态 (B 施加 vs 放开帧数)
    diag   = {id_win_skipped, static_cells_mean},    # 诊断 (无梯度)
  )

第一性设计要点 (43 号 §6 四检查点, 见 t4g_corr_smoke.py 硬检验)
-----------------------------------------------------------------
  ① 框重叠闸: B 格仅 t<t_arrival 施加 L_change; t>=t_arrival(框重叠/物体到达)放开
     —— 否则罚合法放置 (用户强调, 必须对)。
  ② L_id 前后帧局部: 参照=feat[t-1, obj_{t-1}] (非第0帧全局); 上一帧位置局部窗口内 softmax-CE。
  ③ tol 标定: 正常静止 self-sim 0.80 → 变化 0.20 → relu(0.20-tol)=0 不罚; 复制变化 0.64 → 罚。
  ④ 参照系全前后帧 (修 42 号第0帧全局错误)。

设计决定 (43 号 §6 未逐字覆盖的细节, 列出+理由)
------------------------------------------------
  A. L_change 的 static 集用**补集**实现: static_t = 全格 − 动态区。动态区 = 物体足迹
     (当前+上一帧 target_cell 各 3×3) ∪ (t>=t_arrival 时的 B 格)。这样 "背景/distractor"
     自然全部落入 static (匹配 E7 floor=0.80 在随机背景格测得), 且复制出现在**任意**该静
     位置都能抓 (主病复制 67% 出现位置不定, 只约束标注格会漏)。distractor 天然含在补集里。
  B. **上一帧物体格也从 static 排除**: 物体从 c 离开 (c 在 t-1 有物、t 无物) 的合法变化,
     恰体现在"帧 t 的自相似 = feat[t,c] vs feat[t-1,c]", 排除上一帧足迹正好覆盖这次离开
     (EPE0.55 → 逐格平滑移动, r=1 邻域足够)。故 a_cells(原位)无需单列, 补集已治原位重生。
  C. L_change 帧要求**前后帧均 detected**: 漏检帧无法可靠排除物体足迹, 跳过 (宁可少覆盖,
     不误罚合法物体运动)。检出率高 (多数 24/24), 代价可接受。
  D. B 闸判据 = (t_arrival is not None and t < t_arrival): t_arrival 仅在 gate_enabled 时
     由 detect 置值, 故 None 隐含闸禁用/无法定时 → B 全程放开 (宁可漏罚, 不误罚合法放置)。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---- 43 号 §6 起步超参 (不擅自改; trainer/env 可覆盖) ----
TAU = 0.07          # L_id softmax 温度
TOL = 0.2           # L_change 静止容差 (E7: 静止地板 self-sim 0.80 → 正常变化 0.20)
WIN_R_ID = 3        # L_id 局部窗口半径 (7×7; EPE0.55 → 物体 <1 格/帧, r=3 足够含目标)
OBJ_EXCL_R = 1      # L_change 物体足迹排除半径 (3×3, 覆盖物体格及上一帧离开位)


# ------------------------------------------------------------------------------------
#  小工具 (纯坐标, 无梯度)
# ------------------------------------------------------------------------------------
def _valid_cells(cells, H, W):
    out = []
    for c in cells or []:
        gy, gx = int(c[0]), int(c[1])
        if 0 <= gy < H and 0 <= gx < W:
            out.append((gy, gx))
    return out


def _neighborhood(cell, H, W, r):
    gy, gx = int(cell[0]), int(cell[1])
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < H and 0 <= nx < W:
                out.append((ny, nx))
    return out


def prepare(feat, eps=1e-8):
    """L2 归一化特征 (沿 D), 返回 (fn, meta)。fn 与 feat 连图 (归一化可导)。"""
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + eps)
    return fn, dict(T=T, H=H, W=W, D=D)


# ------------------------------------------------------------------------------------
#  ① 身份 loss L_id  (block22, 前后帧接力, 治瞬移/形变)
# ------------------------------------------------------------------------------------
def loss_id(fn, anno, meta, tau=TAU, win_r=WIN_R_ID):
    """前后帧接力 softmax-CE: 上一帧物体格特征 q, 在上一帧位置的局部窗口内,
    应最像**当前帧** GDINO 物体格。

    参照 q = fn[t-1, target_cell_{t-1}] (非第0帧全局!)。候选 = prev 位置 win_r 邻域窗口
    − distractor_cells (排除合法同款干扰; 但目标格必留)。目标 = target_cell_t。τ 温度。
    前后帧任一漏检 / 目标移出窗口 → 跳过。返回 (loss, n_frames, diag)。
    """
    H, W = meta['H'], meta['W']
    device = fn.device
    per = anno['per_lat_frame']
    Tn = min(meta['T'], len(per))
    distractor = set(_valid_cells(anno.get('distractor_cells', []), H, W))
    fn_flat = fn.reshape(fn.shape[0], H * W, -1)

    terms = []
    n_skip = 0
    for t in range(1, Tn):
        fp, fc = per[t - 1], per[t]
        if not (fp.get('detected', False) and fc.get('detected', False)):
            continue
        pc, cc = fp.get('target_cell'), fc.get('target_cell')
        if pc is None or cc is None:
            continue
        pgy, pgx, cgy, cgx = int(pc[0]), int(pc[1]), int(cc[0]), int(cc[1])
        if not (0 <= pgy < H and 0 <= pgx < W and 0 <= cgy < H and 0 <= cgx < W):
            continue
        # 上一帧位置的局部窗口, 排除 distractor (目标格例外必留)
        win = _neighborhood((pgy, pgx), H, W, r=win_r)
        cand = [c for c in win if (c not in distractor or c == (cgy, cgx))]
        if (cgy, cgx) not in cand:
            n_skip += 1                                   # 目标移出窗口 (瞬移超窗) -> 无法监督
            continue
        q = fn[t - 1, pgy, pgx]                            # (D,) 上一帧物体指纹 (已单位化)
        idx = torch.tensor([c[0] * W + c[1] for c in cand], device=device, dtype=torch.long)
        logits = (fn_flat[t][idx] @ q) / tau              # (n,) 当前帧窗口内各格余弦/τ
        logp = F.log_softmax(logits, dim=0)
        terms.append(-logp[cand.index((cgy, cgx))])
    diag = dict(id_win_skipped=n_skip)
    if not terms:
        return fn.new_zeros(()), 0, diag
    return torch.stack(terms).mean(), len(terms), diag


# ------------------------------------------------------------------------------------
#  ② 变化 loss L_change  (block25, 前后帧自相似, 治复制/重生; 主力)
# ------------------------------------------------------------------------------------
def loss_change(fn, anno, meta, tol=TOL, obj_r=OBJ_EXCL_R):
    """该静组前后帧自相似违背: L = mean_{c∈static_t} relu( (1 − cos(feat[t,c],feat[t-1,c])) − tol )。

    static_t = 全格 − 物体足迹(当前+上一帧 target_cell 各 obj_r 邻域) − (t>=t_arrival 时的 B 格)。
    时间闸 (框重叠闸): B 格仅 (t_arrival 有值 且 t<t_arrival) 时留在 static 受约束;
    t>=t_arrival(框重叠/物体到达) 或 t_arrival=None 时从 static 剔除 (放开, 不罚合法放置)。
    前后帧任一漏检 → 跳过 (无法排物体足迹)。t=0 无前帧, 从 t=1 起。
    返回 (loss, n_frames, gate)。
    """
    H, W = meta['H'], meta['W']
    device = fn.device
    per = anno['per_lat_frame']
    Tn = min(meta['T'], len(per))
    t_arr = anno.get('t_arrival', None)
    t_arr = int(t_arr) if t_arr is not None else None
    fn_flat = fn.reshape(fn.shape[0], H * W, -1)

    terms = []
    b_active_frames = 0            # B 受约束帧 (t<t_arrival)
    b_released_frames = 0          # B 放开帧 (t>=t_arrival / 无 t_arrival)
    static_total = 0
    for t in range(1, Tn):
        fp, fc = per[t - 1], per[t]
        if not (fp.get('detected', False) and fc.get('detected', False)):
            continue
        pc, cc = fp.get('target_cell'), fc.get('target_cell')
        if pc is None or cc is None:
            continue
        static = torch.ones(H * W, dtype=torch.bool, device=device)
        # 物体足迹: 当前 + 上一帧 target_cell 各 obj_r 邻域 -> 剔除 (合法运动, 含离开位)
        for cell in ((int(pc[0]), int(pc[1])), (int(cc[0]), int(cc[1]))):
            for (yy, xx) in _neighborhood(cell, H, W, r=obj_r):
                static[yy * W + xx] = False
        # 框重叠闸: B 格
        b = _valid_cells(fc.get('b_cells', []), H, W)
        b_constrained = (t_arr is not None) and (t < t_arr)
        if b:
            if b_constrained:
                b_active_frames += 1                       # 留在 static (到达前, 该静)
            else:
                b_released_frames += 1                      # 剔除 (到达后放开, 不罚合法放置)
                for (yy, xx) in b:
                    static[yy * W + xx] = False
        idx = static.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        cos = (fn_flat[t][idx] * fn_flat[t - 1][idx]).sum(-1)   # (n,) 前后帧自相似
        terms.append(F.relu((1.0 - cos) - tol).mean())
        static_total += int(idx.numel())
    gate = dict(b_active_frames=b_active_frames, b_released_frames=b_released_frames,
                static_cells_mean=(static_total / len(terms)) if terms else 0.0)
    if not terms:
        return fn.new_zeros(()), 0, gate
    return torch.stack(terms).mean(), len(terms), gate


# ------------------------------------------------------------------------------------
#  编排: 一次算两道 (trainer 调用)。L_id 吃 block22 特征, L_change 吃 block25 特征。
# ------------------------------------------------------------------------------------
def compute_corr_losses(feat_id, feat_change, anno, tau=TAU, tol=TOL,
                        win_r=WIN_R_ID, obj_r=OBJ_EXCL_R):
    """feat_id/feat_change (T,H,W,D) + anno -> {losses, counts, gate, diag}。

    单层测试时可传同一特征给两参 (smoke 用)。feat 需 float (余弦精度)。
    """
    fn_id, meta_id = prepare(feat_id)
    fn_ch, meta_ch = prepare(feat_change)
    L_id, n_id, diag = loss_id(fn_id, anno, meta_id, tau=tau, win_r=win_r)
    L_change, n_change, gate = loss_change(fn_ch, anno, meta_ch, tol=tol, obj_r=obj_r)
    return dict(
        losses=dict(L_id=L_id, L_change=L_change),
        counts=dict(n_id=n_id, n_change=n_change),
        gate=gate,
        diag=dict(id_win_skipped=diag['id_win_skipped'],
                  static_cells_mean=gate['static_cells_mean']),
    )
