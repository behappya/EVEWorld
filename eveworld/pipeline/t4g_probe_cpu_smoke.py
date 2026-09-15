#!/usr/bin/env python3
"""EVE · Track4Gen 探针 CPU 冒烟 (工作机无 GPU)。

用最小随机权重 GigaWorld0Transformer3DModel (小参数版, 不加载预训练) 验证三件事:
  1. hook 能抓到全部 num_blocks 层输出, 且每层 shape = (B, T, H, W, D);
  2. 分析核 (光流 + 追踪 + 指标) 在真实视频帧 + 随机特征上跑通不报错, 指标有限;
  3. 正确性硬检验: 特征在时间轴上恒定 (所有帧相同) 时
       - 静组稳定性 ≈ 1.0
       - 追踪预测位移 ≈ 0 (query 停在原格)。

注意: NATTEN 邻域注意力只在 CUDA Hopper/Blackwell 上可用 (见 neighborhood_attn.py),
      故冒烟模型必须用 natten_parameters=None 回退 torch 注意力, 才能在 CPU 跑。

跑法 (工作机 conda giga_models 环境):
  /home/jovyan/miniconda/envs/giga_models/bin/python t4g_probe_cpu_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import torch

import t4g_probe as T

VIDEO = os.environ.get('SMOKE_VIDEO', '/data/datasets/gagi/gr1_finetune_data/raw_data/13.mp4')
NUM_BLOCKS = 28


def build_tiny_model():
    """最小随机权重模型: 28 层, 小通道, torch 注意力 (无 NATTEN), 无 MoE。"""
    from giga_models import GigaWorld0Transformer3DModel
    model = GigaWorld0Transformer3DModel(
        max_img_h=16, max_img_w=16, max_frames=8,
        in_channels=17, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        block_config='FA-CA-MLP',
        model_channels=64, num_blocks=NUM_BLOCKS, num_heads=4,
        mlp_ratio=2.0, crossattn_emb_channels=64, adaln_lora_dim=32,
        natten_parameters=None,   # <- 关键: 无 NATTEN, CPU 可跑
        moe_parameters=None,
    )
    model.eval()
    return model


def test_hook_shapes():
    print('\n[1] hook 抓 28 层 + shape=(B,T,H,W,D) ...')
    model = build_tiny_model()
    B, Tlat, Hlat, Wlat = 1, 6, 8, 8
    x = torch.randn(B, 17, Tlat, Hlat, Wlat)                 # (B, in_ch, T, h, w)
    timesteps = torch.rand(B, 1, Tlat, 1, 1)                 # flatten -> B*T
    crossattn = torch.randn(B, 16, 64)                       # (B, L, ctx_dim)
    padding_mask = torch.zeros(B, 1, Hlat, Wlat)

    shapes = {}
    handles = []
    for name, block in model.blocks.items():
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, n=name: shapes.__setitem__(n, tuple(out.shape))))
    with torch.no_grad():
        model(x=x, timesteps=timesteps, crossattn_emb=crossattn, padding_mask=padding_mask, fps=16)
    for h in handles:
        h.remove()

    assert len(shapes) == NUM_BLOCKS, f'期望 {NUM_BLOCKS} 层, 实抓 {len(shapes)}'
    exp = (B, Tlat, Hlat // 2, Wlat // 2, 64)                # patch_spatial=2 -> H/2, W/2; D=model_channels
    for n, s in shapes.items():
        assert len(s) == 5, f'{n} 输出非 5 维: {s}'
        assert s == exp, f'{n} shape={s} != 期望{exp}'
    print(f'    OK: {len(shapes)} 层, 每层输出 (B,T,H,W,D)={exp}  (feature 网格 T={Tlat},H={Hlat//2},W={Wlat//2},D=64)')
    # 返回一层真实(随机权重)特征供分析核测试
    feat = None
    handles = []
    store = {}
    for name, block in model.blocks.items():
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, n=name: store.__setitem__(n, out.detach()[0])))
    with torch.no_grad():
        model(x=x, timesteps=timesteps, crossattn_emb=crossattn, padding_mask=padding_mask, fps=16)
    for h in handles:
        h.remove()
    return store['block13']   # (T, H, W, D)


def test_analysis_core_real_frames(feat):
    print('\n[2] 分析核在 真实视频帧 + 随机特征 上跑通 ...')
    T_lat, H, W, D = feat.shape
    n_pix = 25
    frames = T.sample_frames_like_training(VIDEO, num_frames=n_pix, height=240, width=384)
    print(f'    真实帧: {frames.shape} <- {os.path.basename(VIDEO)}')
    flows = T.dense_flow_sequence(frames)
    tm = T.motion_grid(flows, H, W)
    dyn, static = T.classify_dyn_static(tm, T.DEFAULT_P_HI, T.DEFAULT_P_LO)
    print(f'    光流网格 ({H}x{W})  动组={len(dyn)}  静组={len(static)}')
    m = T.compute_layer_metrics(feat, flows, dyn, static, n_pix, device='cpu')
    print(f'    指标(随机特征): median_epe={m["median_epe"]:.3f} cell  '
          f'static_stability={m["static_stability"]:.3f}')
    assert np.isfinite(m['median_epe']), 'EPE 非有限'
    assert np.isfinite(m['static_stability']), 'STAB 非有限'
    print('    OK: 分析核跑通, 指标有限')
    return frames, flows, dyn, static


def test_correctness_static(frames, flows):
    print('\n[3] 正确性硬检验: 时间轴恒定特征 -> 稳定性≈1.0, 追踪位移≈0 ...')
    T_lat, H, W, D = 6, 4, 4, 64
    base = torch.randn(1, H, W, D)
    feat_const = base.expand(T_lat, H, W, D).contiguous()    # 所有帧相同
    tm = T.motion_grid(flows, H, W)
    dyn, static = T.classify_dyn_static(tm, T.DEFAULT_P_HI, T.DEFAULT_P_LO)
    if len(static) == 0:
        static = [(0, 0)]
    if len(dyn) == 0:
        dyn = [(H // 2, W // 2)]

    stab, per = T.static_stability(feat_const, static, device='cpu')
    print(f'    静组稳定性(恒定特征) = {stab:.6f}  (期望 ≈ 1.0)')
    assert stab > 0.999, f'静组稳定性应≈1.0, 实得 {stab}'

    pred = T.soft_argmax_track(feat_const, dyn, device='cpu')  # (T, N, 2)
    disp = np.sqrt(((pred[1:] - pred[0:1]) ** 2).sum(-1))      # 相对首帧位移
    med_disp = float(np.median(disp))
    print(f'    追踪预测位移(恒定特征) 中位 = {med_disp:.6f} cell  (期望 ≈ 0)')
    assert med_disp < 1e-3, f'恒定特征追踪应停在原格, 实得位移 {med_disp}'
    print('    OK: 稳定性≈1.0 且 追踪位移≈0, 分析核正确')


def main():
    print('=' * 78)
    print(' Track4Gen 探针 CPU 冒烟 (无 GPU)')
    print(f'   torch={torch.__version__}  cuda_available={torch.cuda.is_available()}')
    print('=' * 78)
    feat = test_hook_shapes()
    frames, flows, dyn, static = test_analysis_core_real_frames(feat)
    test_correctness_static(frames, flows)
    print('\n' + '=' * 78)
    print(' CPU 冒烟全部通过 (3/3)。')
    print('=' * 78)


if __name__ == '__main__':
    main()
