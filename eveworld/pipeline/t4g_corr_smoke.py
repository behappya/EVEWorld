#!/usr/bin/env python3
"""EVE · Track4Gen 式对应监督 —— CPU 冒烟 (工作机无 GPU, 真跑 43 号 §6 四硬检验)。

跑法 (train venv):
  /data/datasets/gagi/envs/giga_world_train_venv/bin/python \
      eveworld/pipeline/t4g_corr_smoke.py

四硬检验 (43 号 §6 检查点, 必过):
  [1] 框重叠闸 (用户强调, 核心): 同一 B 变化, t_arrival 不同 ->
        到达前 (t<t*) B 有复制  -> L_change > 0 (B 受约束);
        到达后 (t>=t*) B 变化    -> L_change 对 B = 0 (B 放开, 不罚合法放置)。
  [2] L_id 前后帧局部: 恒定特征 -> L_id≈0; 目标移动 -> 窗口 argmax = 当前帧目标格。
  [3] tol 标定: 静止小变化 0.20 -> relu(0.20-0.20)=0 不罚; 大变化 0.50 -> relu(0.30)>0 罚。
  [4] 双层 hook: tiny 模型同时抓 block22 + block25, 各输出 (B,T,H,W,D);
        两道 loss 反传, 梯度流到 block0..block25 (block26/27 在 block25 之后, 无梯度)。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch

from eveworld.pipeline.t4g_corr_loss import (
    compute_corr_losses, loss_change, loss_id, prepare,
)

D = 16
H, W, T = 6, 8, 5


def _basis(i, d=D):
    v = torch.zeros(d)
    v[i] = 1.0
    return v


E0 = _basis(0)   # 物体身份方向
E1 = _basis(1)   # 背景方向 (与 E0 正交)


def _unit_mix(a, b, ca):
    """返回单位向量 = ca*a + sqrt(1-ca^2)*b (a,b 正交单位); 与 a 的余弦 = ca。"""
    cb = (1.0 - ca * ca) ** 0.5
    return ca * a + cb * b


# ====================================================================================
#  [1] 框重叠闸硬检验 (核心)
# ====================================================================================
def _gate_scene():
    """恒定背景 E1 + 物体 E0@(2,3) 全程 + B 格 [(5,7),(5,6)]; B 格 frame0=E1, frame>=1=E0
    (仅 B 在 0->1 突变, 其余全静)。返回 (feat, b_cells)。"""
    b_cells = [[5, 7], [5, 6]]
    frame0 = E1.view(1, 1, D).expand(H, W, D).clone()
    frame0[2, 3] = E0
    for (gy, gx) in b_cells:
        frame0[gy, gx] = E1
    later = frame0.clone()
    for (gy, gx) in b_cells:
        later[gy, gx] = E0                      # B 在 frame0->1 突变 (模拟复制/放置)
    feat = torch.stack([frame0] + [later] * (T - 1), dim=0).contiguous()
    return feat, b_cells


def _gate_anno(t_arrival, b_cells):
    return dict(
        target_cell_0=[2, 3],
        distractor_cells=[],
        t_arrival=t_arrival,
        per_lat_frame=[dict(target_cell=[2, 3], b_cells=b_cells, detected=True) for _ in range(T)],
    )


def test_1_box_overlap_gate():
    print('\n[1] 框重叠闸硬检验 (用户强调, 核心) ...')
    feat, b_cells = _gate_scene()
    fn, meta = prepare(feat)

    # 到达前: t_arrival=4 -> B 在 t=1(<4) 受约束, 突变被罚
    L_before, n_b, g_b = loss_change(fn, _gate_anno(4, b_cells), meta)
    # 到达后: t_arrival=1 -> B 在 t=1(>=1) 放开, 突变不罚 (其余全静) -> L_change=0
    L_after, n_a, g_a = loss_change(fn, _gate_anno(1, b_cells), meta)

    print('    到达前 t*=4: L_change=%.4e (nf=%d)  gate: B_active=%d B_released=%d'
          % (L_before.item(), n_b, g_b['b_active_frames'], g_b['b_released_frames']))
    print('    到达后 t*=1: L_change=%.4e (nf=%d)  gate: B_active=%d B_released=%d'
          % (L_after.item(), n_a, g_a['b_active_frames'], g_a['b_released_frames']))
    assert L_before.item() > 1e-3, '到达前 B 有复制 -> L_change 应 > 0'
    assert L_after.item() < 1e-6, '到达后 B 变化 (框重叠) -> L_change 对 B 应 = 0 (放开)'
    assert g_b['b_active_frames'] == 3 and g_b['b_released_frames'] == 1, 't*=4 应 3 帧约束 1 帧放开'
    assert g_a['b_active_frames'] == 0 and g_a['b_released_frames'] == 4, 't*=1 应全放开'
    print('    OK: 到达前罚复制 / 到达后放开合法放置; 闸帧计数正确')


# ====================================================================================
#  [2] L_id 前后帧局部
# ====================================================================================
def test_2_id_local_relay():
    print('\n[2] L_id 前后帧接力局部 ...')
    # (a) 恒定: 物体不动 @(2,3) -> L_id≈0
    fconst = E1.view(1, 1, D).expand(H, W, D).clone()
    fconst[2, 3] = E0
    feat_c = fconst.view(1, H, W, D).expand(T, H, W, D).contiguous()
    anno_c = dict(distractor_cells=[[0, 0]],
                  per_lat_frame=[dict(target_cell=[2, 3], b_cells=[], detected=True) for _ in range(T)])
    fn_c, meta = prepare(feat_c)
    Lc, nc, _ = loss_id(fn_c, anno_c, meta)
    print('    恒定不动:   L_id=%.4e (nf=%d)' % (Lc.item(), nc))
    assert Lc.item() < 1e-2 and nc == T - 1, '恒定特征 L_id 应≈0'

    # (b) 移动: 物体 @(2,2+t) 逐帧右移 -> 窗口 argmax = 当前帧目标格, L_id 小
    frames = []
    per = []
    for t in range(T):
        f = E1.view(1, 1, D).expand(H, W, D).clone()
        f[2, 2 + t] = E0
        frames.append(f)
        per.append(dict(target_cell=[2, 2 + t], b_cells=[], detected=True))
    feat_m = torch.stack(frames, dim=0).contiguous()
    anno_m = dict(distractor_cells=[], per_lat_frame=per)
    fn_m, meta = prepare(feat_m)
    Lm, nm, _ = loss_id(fn_m, anno_m, meta)
    # 手工验 argmax: t=2, q=fn[1,(2,3)]=E0, 当前帧窗口内余弦最大格应为 (2,4)
    t = 2
    win = [(2, 2 + t + dx) for dx in (-1, 0, 1)]           # (2,3)(2,4)(2,5)
    coss = {c: float((fn_m[t, c[0], c[1]] * fn_m[t - 1, 2, 2 + t - 1]).sum()) for c in win}
    argmax_cell = max(coss, key=coss.get)
    print('    逐帧右移:   L_id=%.4e (nf=%d)  t=2 窗口余弦=%s argmax=%s (目标=(2,4))'
          % (Lm.item(), nm, {k: round(v, 2) for k, v in coss.items()}, argmax_cell))
    assert Lm.item() < 1e-2, '目标移动但被追上 -> L_id 应小'
    assert argmax_cell == (2, 4), '当前帧窗口 argmax 应 = 当前帧目标格 (2,4)'
    print('    OK: 恒定 L_id≈0; 移动时 argmax = 当前帧目标格 (前后帧接力追上)')


# ====================================================================================
#  [3] tol 标定
# ====================================================================================
def _witness_scene(ca):
    """全格恒定 E1 + 物体 E0@(0,0); 见证静止格 (3,4): frame0=E0, frame>=1 = 与 E0 余弦 ca。
    (物体在 (0,0) 会被足迹排除; (3,4) 是纯静止格, 用来标定 tol。)"""
    frame0 = E1.view(1, 1, D).expand(H, W, D).clone()
    frame0[0, 0] = E0
    frame0[3, 4] = E0
    later = frame0.clone()
    later[3, 4] = _unit_mix(E0, E1, ca)          # 见证格前后帧余弦 = ca -> 变化 = 1-ca
    feat = torch.stack([frame0] + [later] * (T - 1), dim=0).contiguous()
    return feat


def _witness_anno():
    return dict(distractor_cells=[], t_arrival=None,
                per_lat_frame=[dict(target_cell=[0, 0], b_cells=[], detected=True) for _ in range(T)])


def test_3_tol_calibration():
    print('\n[3] tol 标定 (tol=0.20) ...')
    # 小变化: cos=0.80 -> 1-cos=0.20 -> relu(0.20-0.20)=0 不罚
    fn_s, meta = prepare(_witness_scene(ca=0.80))
    Ls, ns, _ = loss_change(fn_s, _witness_anno(), meta)
    # 大变化: cos=0.50 -> 1-cos=0.50 -> relu(0.50-0.20)=0.30 罚
    fn_b, meta = prepare(_witness_scene(ca=0.50))
    Lb, nb, _ = loss_change(fn_b, _witness_anno(), meta)
    print('    静止小变化 cos=0.80 (变化0.20): L_change=%.4e (nf=%d)' % (Ls.item(), ns))
    print('    大变化     cos=0.50 (变化0.50): L_change=%.4e (nf=%d)' % (Lb.item(), nb))
    assert Ls.item() < 1e-5, '变化0.20=tol -> relu=0 不罚'
    assert Lb.item() > 1e-4, '变化0.50>tol -> relu>0 罚'
    print('    OK: 0.20 变化不罚 (=tol 地板); 0.50 变化被罚')


# ====================================================================================
#  [4] 双层 hook (block22 + block25) + 梯度
# ====================================================================================
def _build_tiny_model(num_blocks=28, model_channels=64):
    from giga_models import GigaWorld0Transformer3DModel
    m = GigaWorld0Transformer3DModel(
        max_img_h=16, max_img_w=16, max_frames=8,
        in_channels=17, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        block_config='FA-CA-MLP',
        model_channels=model_channels, num_blocks=num_blocks, num_heads=4,
        mlp_ratio=2.0, crossattn_emb_channels=64, adaln_lora_dim=32,
        natten_parameters=None, moe_parameters=None,
    )
    m.eval()
    return m


def test_4_double_hook_grad():
    print('\n[4] 双层 hook block22 + block25 + 梯度 ...')
    torch.manual_seed(0)
    model = _build_tiny_model()
    B, Tl, Hl, Wl = 1, 6, 8, 8
    x = torch.randn(B, 17, Tl, Hl, Wl)
    timesteps = torch.rand(B, 1, Tl, 1, 1)
    crossattn = torch.randn(B, 8, 64)
    padding_mask = torch.zeros(B, 1, Hl, Wl)

    store = {}
    handles = []
    for name in ('block22', 'block25'):
        h = model.blocks[name].register_forward_hook(
            lambda _m, _i, out, n=name: store.__setitem__(n, out))
        handles.append(h)
    try:
        model(x=x, timesteps=timesteps, crossattn_emb=crossattn, padding_mask=padding_mask, fps=16)
    finally:
        for h in handles:
            h.remove()
    assert 'block22' in store and 'block25' in store, '双层 hook 应各抓到一次'
    f22, f25 = store['block22'], store['block25']
    Hf, Wf = Hl // 2, Wl // 2
    print('    block22 输出 shape =', tuple(f22.shape), ' block25 输出 shape =', tuple(f25.shape),
          '(期望 (1,6,4,4,%d))' % model.model_channels)
    assert f22.dim() == 5 and f22.shape[:4] == (B, Tl, Hf, Wf), 'block22 输出非 (B,T,H,W,D)'
    assert f25.dim() == 5 and f25.shape[:4] == (B, Tl, Hf, Wf), 'block25 输出非 (B,T,H,W,D)'

    anno = dict(
        target_cell_0=[1, 1], distractor_cells=[[0, 0]], t_arrival=3,
        per_lat_frame=[dict(target_cell=[1, 1], b_cells=[[Hf - 1, 0]], detected=True) for _ in range(Tl)],
    )
    res = compute_corr_losses(f22[0].float(), f25[0].float(), anno)
    L = res['losses']
    total = L['L_id'] + L['L_change']
    print('    对应 loss: L_id=%.4e(nf=%d)  L_change=%.4e(nf=%d)  gate: active=%d released=%d'
          % (L['L_id'].item(), res['counts']['n_id'], L['L_change'].item(), res['counts']['n_change'],
             res['gate']['b_active_frames'], res['gate']['b_released_frames']))
    model.zero_grad(set_to_none=True)
    total.backward()
    g0 = sum(p.grad.abs().sum().item() for p in model.blocks['block0'].parameters() if p.grad is not None)
    g22 = sum(p.grad.abs().sum().item() for p in model.blocks['block22'].parameters() if p.grad is not None)
    g25 = sum(p.grad.abs().sum().item() for p in model.blocks['block25'].parameters() if p.grad is not None)
    g27 = [p.grad for p in model.blocks['block27'].parameters() if p.grad is not None]
    print('    grad_abs_sum: block0=%.4e block22=%.4e block25=%.4e  block27_params_with_grad=%d'
          % (g0, g22, g25, len(g27)))
    assert g25 > 0 and g22 > 0 and g0 > 0, 'block0/22/25 应有非零梯度'
    assert len(g27) == 0, 'block27 (在 block25 之后, 不喂 loss) 不应有梯度'
    print('    OK: 双层 hook 各抓 (B,T,H,W,D); 两道 loss 梯度只沿 block0..block25 反传')


def main():
    print('=' * 84)
    print(' Track4Gen 式对应监督 CPU 冒烟 (43 号 §6 两道 loss, 无 GPU)')
    print('   torch=%s  cuda_available=%s' % (torch.__version__, torch.cuda.is_available()))
    print('=' * 84)
    test_1_box_overlap_gate()
    test_2_id_local_relay()
    test_3_tol_calibration()
    test_4_double_hook_grad()
    print('\n' + '=' * 84)
    print(' CPU 冒烟全部通过 (1-4)。框重叠闸 / L_id 接力 / tol 标定 / 双层 hook 硬检验 OK。')
    print('=' * 84)


if __name__ == '__main__':
    main()
