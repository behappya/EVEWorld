#!/usr/bin/env python3
"""ICH-D 训练 (71 号 B1 固化预案): 固化 LR 检测子 + zero-init 擦除子, 只训擦除+主干。

背景: 学习型检测子 (C 臂 conv 头 + BCE + pos_weight) step158 判决失败 (det 分离
+0.005 << 预注册线 0.05)。D 臂改为: 检测子离线拟合并**固化** (buffer, 永不更新),
训练信号全部留给擦除子; λ_det=0, BCE 撤掉。

结构:
  检测特征 16 维 = [block10/12/16 novelty 各 4 维 (照抄 C 臂 transient_features)]
                 + [block22 CIC 认亲 4 维 (selfcase_cic_match v1, LOO 0.8261@0.4)]
    认亲 4 维 (t->t-1 全图匹配, 逐帧滑窗聚合, 口径照抄 cic_cell_features):
      [F1 窗内最大来路距离, F2 该帧伴随 margin, F3 窗内最大重复认领数,
       F4 窗内平均 (1-sim0)]
  检测子 = 固化 LR (t4g_ich_d_fit.py 产物: w/b/mu/sd 注册为 **buffer**),
    逐格标准化 + 线性打分 + sigmoid -> M (B,24,30,48); 全程 no_grad (M 无梯度)。
  擦除子 = Δ = -sigmoid(gate)·M·W_out(h22), W_out zero-init (cic_transport 同款)。

**注入点 = block22 输出处 (对 71 号 "block16 后注入" 的必要偏离)**: block22 认亲
特征在前向中于 block16 之后才产生, 单次前向里 block16 处因果上拿不到 16 维 M;
注入点后移到特征可得的最早位置 block22 (TIA transport 同点位先例), zero-init
恒等语义不变。

认亲前向成本 (预算核算, 结论=无需 stride2 下采样):
  逐帧对 S = flat[t] @ flat[t-1]^T: (1440,2048)x(2048,1440) ≈ 8.5 GFLOP, 23 对
  x B=1 ≈ 200 GFLOP (<0.5s, transformer 单前向的零头); 显存逐帧 S fp32 8.3MB
  + dist/margin/claims (23,1440)x3 ≈ 0.4MB, 全 no_grad 不进图。

训练: T4GICHDTrainer(T4GSelfCaseTrainer) —— 总 loss = A1 修复损失 (原样), 无 BCE;
可训 = gate + W_out + 主干。哨兵 [ichd-sentinel] 观测固化 M 的 det_pos/det_neg
(step1 即应显示 ~+0.27 量级分离 = 固化权重装载正确的运行时验证)。
`--selftest` CPU 单测: 固化装载对账 / 零初始化恒等 / 认亲特征红线 / EMA 兼容。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import nn

from .t4g_aug_paste import T_LAT, H_LAT
from .t4g_ich_trainer import ICHModule
from .t4g_selfcase_trainer import T4GSelfCaseTrainer

W_LAT = 48

DEFAULT_WEIGHTS = '/data/datasets/gagi/eve_v2_outputs/selfcase/cic_match/ich_d_frozen_lr.npz'
NOVELTY_BLOCKS = ('block10', 'block12', 'block16')
CIC_BLOCK = 'block22'


# ====================================================================================
#  ICH-D 模块
# ====================================================================================
class ICHDModule(nn.Module):
    """固化 LR 检测子 (buffer) + zero-init 擦除子; 可训参数只有 gate + w_out。"""

    def __init__(self, channels, weights_path=DEFAULT_WEIGHTS, win=3):
        super().__init__()
        self.novelty_blocks = NOVELTY_BLOCKS
        self.cic_block = CIC_BLOCK
        self.win = int(win)
        d = np.load(weights_path, allow_pickle=True)
        self.frozen_meta = json.loads(str(d['meta'])) if 'meta' in d else {}
        assert d['w'].shape == (16,) and d['mu'].shape == (16,) and d['sd'].shape == (16,)
        # 固化检测子: buffer 非 Parameter, 永不进 optimizer; 全部 >=1 维 (EMA 兼容)
        self.register_buffer('lr_w', torch.from_numpy(d['w']).float())
        self.register_buffer('lr_b', torch.from_numpy(np.atleast_1d(d['b'])).float())
        self.register_buffer('lr_mu', torch.from_numpy(d['mu']).float())
        self.register_buffer('lr_sd', torch.from_numpy(d['sd']).float())
        # 擦除子 (唯一可训): gate 必须 1 维 (EMAModel 0 维存档坑, C 臂实锤)
        self.gate = nn.Parameter(torch.zeros(1))
        self.w_out = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.w_out.weight)                  # cic_transport 同款 zero-init
        self._feats = {}
        self.last_m = None                                 # (B,T,H,W) fp32, no_grad
        # ---- 双态门控 (72 号 §5, 事故驱动修订): 默认全关 = D 臂原语义 ----
        # sigma_band: (lo,hi) 或 None; 当前 σ 带外 -> suppress 恒等 (不检不擦)。
        # m_gate: >0 -> Δ 只作用于 M>m_gate 的格 (稀疏保守门)。
        # cur_sigma: 调用方每前向设置 (trainer 显式 / pre-hook 从 timesteps 反解)。
        self.sigma_band = None
        self.m_gate = 0.0
        self.cur_sigma = None

    def set_gates(self, sigma_band=None, m_gate=0.0):
        self.sigma_band = tuple(sigma_band) if sigma_band else None
        self.m_gate = float(m_gate)

    def set_sigma(self, sigma):
        self.cur_sigma = None if sigma is None else float(sigma)

    def sigma_in_band(self):
        """σ 带判定; 未知 σ 或未启带门 -> 视为带内 (回退 D 臂原语义)。"""
        if self.sigma_band is None or self.cur_sigma is None:
            return True
        return self.sigma_band[0] <= self.cur_sigma <= self.sigma_band[1]

    def reset(self):
        self._feats = {}
        self.last_m = None

    def observe(self, name, out):
        self._feats[name] = out

    @staticmethod
    @torch.no_grad()
    def cic_features(feat, win=3):
        """feat (B,T,H,W,D) fp32 -> (B,T,H,W,4) 认亲特征 (口径照抄 selfcase_cic_match:
        match_maps + cic_cell_features, 窗口=逐帧滑动 dup_window(t))。
        方向固定 t -> t-1; 认领过筛阈值 = 当帧 margin 中位数 (torch.quantile 线性插值
        = np.median, 与离线拟合特征精确同口径)。"""
        B, T, H, W, _ = feat.shape
        hw = H * W
        fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        flat = fn.reshape(B, T, hw, -1)
        ys = (torch.arange(hw, device=feat.device) // W).float()
        xs = (torch.arange(hw, device=feat.device) % W).float()
        dist = feat.new_zeros(B, T - 1, hw)
        margin = feat.new_zeros(B, T - 1, hw)
        claims = feat.new_zeros(B, T - 1, hw)
        for t in range(1, T):                              # 逐帧对, S 只占 8.3MB
            s_mat = torch.bmm(flat[:, t], flat[:, t - 1].transpose(1, 2))  # (B,hw,hw)
            v2, idx = s_mat.topk(2, dim=-1)
            src = idx[..., 0]                              # (B,hw) argmax 源格
            m = v2[..., 0] - v2[..., 1]                    # top1 - top2 margin
            dist[:, t - 1] = torch.hypot(ys[None] - ys[src], xs[None] - xs[src])
            margin[:, t - 1] = m
            med = torch.quantile(m, 0.5, dim=-1, keepdim=True)
            for b in range(B):
                sel = src[b][m[b] > med[b, 0]]
                cnt = torch.bincount(sel, minlength=hw).float()
                claims[b, t - 1] = cnt[src[b]]
        sim0 = (fn * fn[:, :1]).sum(-1).reshape(B, T, hw)
        rows = []
        for t in range(T):
            lo = max(1, min(t - 1, T - 1))                 # g1p.dup_window 同款
            hi = max(lo + 1, min(lo + win, T))
            dd = dist[:, lo - 1:hi - 1]
            f1, t_star = dd.max(dim=1)
            f2 = margin[:, lo - 1:hi - 1].gather(1, t_star.unsqueeze(1)).squeeze(1)
            f3 = claims[:, lo - 1:hi - 1].max(dim=1).values
            f4 = (1.0 - sim0[:, lo:hi]).mean(dim=1)
            rows.append(torch.stack([f1, f2, f3, f4], dim=-1))
        return torch.stack(rows, dim=1).reshape(B, T, H, W, 4)

    @torch.no_grad()
    def detect(self):
        """16 维特征 -> 标准化 -> 固化线性打分 -> M=sigmoid (B,T,H,W); 无梯度。"""
        feats = [ICHModule.transient_features(self._feats[n].detach().float(), self.win)
                 for n in self.novelty_blocks]
        feats.append(self.cic_features(self._feats[self.cic_block].detach().float(),
                                       self.win))
        f16 = torch.cat(feats, dim=-1)                     # (B,T,H,W,16) fp32
        z = (f16 - self.lr_mu) / self.lr_sd
        logits = (z * self.lr_w).sum(-1) + self.lr_b
        m = torch.sigmoid(logits)
        self.last_m = m
        return m

    def suppress(self, name, out):
        """block22 输出处: Δ = -sigmoid(gate)·M_eff·W_out(h22), W_out zero-init 恒等。
        门控 (若启用): σ 带外 -> 恒等直通 (不检不擦, last_m=None);
        m_gate>0 -> Δ 只作用于 M>m_gate 的格 (稀疏保守)。"""
        if not self.sigma_in_band():
            self._feats = {}
            self.last_m = None
            return out
        self.observe(name, out)
        m = self.detect().unsqueeze(-1)                    # (B,T,H,W,1) fp32 无梯度
        if self.m_gate > 0:
            m = m * (m > self.m_gate)
        m = m.to(out.dtype)
        delta = -torch.sigmoid(self.gate).to(out.dtype) * m * self.w_out(out)
        self._feats = {}
        return out + delta


# ====================================================================================
#  trainer
# ====================================================================================
class T4GICHDTrainer(T4GSelfCaseTrainer):
    """A1 修复损失 (原样, 无 BCE/λ_det) + 固化 M 观测哨兵 [ichd-sentinel]。"""

    def get_models(self, model_config):
        model = super().get_models(model_config)
        weights_path = os.environ.get('T4G_ICH_D_WEIGHTS',
                                      model_config.get('t4g_ich_d_weights', DEFAULT_WEIGHTS))
        win = int(os.environ.get('T4G_ICH_WIN', model_config.get('t4g_ich_win', 3)))
        tf = model['transformer']
        for b in NOVELTY_BLOCKS + (CIC_BLOCK,):
            assert b in tf.blocks, f'未知 block: {b}'
        ich = ICHDModule(channels=tf.config.model_channels, weights_path=weights_path,
                         win=win)
        for n, p in list(ich.named_parameters()) + list(ich.named_buffers()):
            assert p.dim() >= 1, f'ICH-D {n} 不得为 0 维 (EMAModel.state_dict 会崩)'
        ich.to(self.dtype)
        # LR buffer 保持 fp32 (标准化/打分精度; bf16 的 sd 倒数会放大量化误差)
        for name in ('lr_w', 'lr_b', 'lr_mu', 'lr_sd'):
            getattr(ich, name).data = getattr(ich, name).data.float()
        tf.ich_d = ich
        self._ichd_hook_done = False
        self._ichd_stat_buffer = []
        self._ichd_logged_step = -1
        st = ich.frozen_meta.get('stats', {}).get('0.4', {})
        self.print('[t4g-ich-d] frozen LR <- %s | 16d (novelty %s + cic %s) | '
                   'fit@0.4: train_auc=%s M_dup=%s M_bg=%s | trainable=gate+w_out(%d) '
                   '| inject=%s (block16 因果不可行, 后移至认亲特征可得点)'
                   % (weights_path, ','.join(NOVELTY_BLOCKS), CIC_BLOCK,
                      st.get('train_auc'), st.get('m_dup'), st.get('m_bg'),
                      ich.w_out.weight.numel() + 1, CIC_BLOCK))
        return model

    def _ensure_ichd_hooks(self):
        if self._ichd_hook_done:
            return
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        tf = model['transformer']
        ich = tf.ich_d
        for name in NOVELTY_BLOCKS:
            tf.blocks[name].register_forward_hook(
                lambda _m, _i, out, _n=name: ich.observe(_n, out))
        tf.blocks[CIC_BLOCK].register_forward_hook(
            lambda _m, _i, out: ich.suppress(CIC_BLOCK, out))
        self._ichd_hook_done = True
        self.print('[t4g-ich-d] hooks -> observe %s + suppress %s (%s)'
                   % (list(NOVELTY_BLOCKS), CIC_BLOCK,
                      type(tf.blocks[CIC_BLOCK]).__name__))

    @staticmethod
    def build_case_mask(paste_lat, has_paste, t_lat=T_LAT, h_lat=H_LAT, w_lat=W_LAT):
        """同 C 臂: paste_lat (VAE ÷8) -> 特征网格 (÷2, ceil 上界) 病例掩码。"""
        from .t4g_ich_trainer import T4GICHTrainer
        return T4GICHTrainer.build_case_mask(paste_lat, has_paste, t_lat, h_lat, w_lat)

    def forward_step(self, batch_dict):
        self._ensure_ichd_hooks()
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        ich = model['transformer'].ich_d
        ich.reset()
        losses = super().forward_step(batch_dict)          # A1 修复损失, 无 BCE
        m = ich.last_m                                     # (B,T,H,W) fp32, no_grad
        assert m is not None, 'ICH-D hook 未触发'
        mask = self.build_case_mask(batch_dict['paste_lat'],
                                    batch_dict['has_paste']).to(m.device)
        # ---- [ichd-sentinel] 固化 M 观测 (布局同 C 臂, bce 槽位挪作 M 全图均值) ----
        # [0]n_case [1]Σdet_pos [2]Σdet_neg [3]n_samples [4]Σm_mean [5]Σdet_neg_clean
        # [6]Σgate [7]Σwout_norm [8]n_calls
        with torch.no_grad():
            stats = torch.zeros(9, device=m.device)
            for b in range(m.shape[0]):
                mb = mask[b] > 0.5
                stats[3] += 1
                if mb.any():
                    stats[0] += 1
                    stats[1] += float(m[b][mb].mean())
                    stats[2] += float(m[b][~mb].mean())
                else:
                    stats[5] += float(m[b].mean())
            stats[4] += float(m.mean())
            stats[6] += float(torch.sigmoid(ich.gate.detach().float()))
            stats[7] += float(ich.w_out.weight.detach().float().norm())
            stats[8] += 1
        gathered = self.accelerator.gather(stats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._ichd_stat_buffer.append(gathered)
        return losses

    def print_step(self):
        super().print_step()                               # [aug-sentinel] 原样
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        if self.cur_step == self._ichd_logged_step:
            return
        self._ichd_logged_step = self.cur_step
        agg = (torch.stack(self._ichd_stat_buffer).sum(0)
               if self._ichd_stat_buffer else torch.zeros(9))
        self._ichd_stat_buffer = []

        def _r(n, d):
            return (agg[n] / agg[d]).item() if agg[d] > 0 else float('nan')

        det_neg_clean = (agg[5] / (agg[3] - agg[0])).item() if agg[3] > agg[0] else float('nan')
        msg = ('[ichd-sentinel] step=%d cases=%d/%d | det_pos=%.4f det_neg=%.4f '
               'det_neg_clean=%.4f | m_mean=%.4f | gate=%.3f wout_norm=%.4e '
               '| frozen16d inject=%s'
               % (self.cur_step, int(agg[0]), int(agg[3]), _r(1, 0), _r(2, 0),
                  det_neg_clean, _r(4, 8), _r(6, 8), _r(7, 8), CIC_BLOCK))
        self.logger.info(msg)
        print(msg, flush=True)


# ====================================================================================
#  selftest (CPU): 固化装载对账 / 零初始化恒等 / 认亲红线 / EMA 兼容
# ====================================================================================
def selftest():
    import functools as ft
    import operator as op
    import tempfile
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    print('[ich-d selftest] start')
    B, T, H, W, D = 1, T_LAT, H_LAT, W_LAT, 32

    # 合成固化权重 (真实 fit 产物同 schema)
    w = rng.randn(16); b = np.array([0.3]); mu = rng.randn(16) * 0.1; sd = rng.rand(16) + 0.5
    tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    np.savez(tmp.name, w=w, b=b, mu=mu, sd=sd,
             meta=np.array(json.dumps(dict(stats={})), dtype=object))
    ich = ICHDModule(channels=D, weights_path=tmp.name)

    # 1) EMA 兼容: 参数+buffer 全 >=1 维且 reduce(shape) 可算
    for n, p in list(ich.named_parameters()) + list(ich.named_buffers()):
        assert p.dim() >= 1, n
        assert ft.reduce(op.mul, p.shape) == p.numel()
    print('[ich-d selftest] EMA save compat OK (params+buffers, no 0-dim)')

    # 2) 认亲特征红线: 复制品出现 -> F1 来路距离 ~ 原物距离, F3 认领 >= 2
    #    背景带共享分量 (照抄 cic_match._mk_scene): 背景互相有点像 -> margin 中位
    #    数非饱和, 认领过筛 (margin>中位) 才有区分力
    g = torch.randn(1, 1, 1, 1, D)
    base = 0.8 * g + torch.randn(1, 1, H, W, D)
    feat = base.repeat(1, T, 1, 1, 1) + 0.02 * torch.randn(1, T, H, W, D)
    vo = torch.randn(D) * 1.5
    (oy, ox), (cy, cx), a_t = (6, 8), (22, 38), 8
    feat[0, :, oy, ox] = vo + 0.02 * torch.randn(T, D)          # 原物常驻
    feat[0, a_t:, cy, cx] = vo + 0.02 * torch.randn(T - a_t, D)  # 复制品出现
    f4 = ICHDModule.cic_features(feat.float())
    assert f4.shape == (1, T, H, W, 4)
    true_d = float(np.hypot(cy - oy, cx - ox))
    got_d = float(f4[0, a_t, cy, cx, 0])
    got_claims = float(f4[0, a_t, cy, cx, 2])
    bg_d = float(f4[0, a_t, cy + 3, cx + 3, 0])
    print(f'[ich-d selftest] cic 红线: dup F1={got_d:.1f} (true {true_d:.1f}) '
          f'F3={got_claims:.0f} bg F1={bg_d:.1f}')
    assert got_d >= true_d - 3 and got_claims >= 2 and bg_d <= 2.0

    # 3) 固化打分对账: detect() 与 numpy 手算 sigmoid(w·z+b) 逐位一致
    for n in NOVELTY_BLOCKS:
        ich.observe(n, feat)
    ich.observe(CIC_BLOCK, feat)
    m = ich.detect()
    assert m.shape == (1, T, H, W)
    f16 = torch.cat([ICHModule.transient_features(feat.float()) for _ in NOVELTY_BLOCKS]
                    + [f4], dim=-1).numpy()
    z = (f16 - mu) / sd
    m_ref = 1.0 / (1.0 + np.exp(-(z @ w + b[0])))
    assert np.allclose(m.numpy(), m_ref, atol=1e-5), '固化打分与手算不一致'
    print(f'[ich-d selftest] frozen scoring == numpy manual (atol 1e-5) OK; '
          f'M_dup_cell={float(m[0, a_t, cy, cx]):.3f} M_bg={float(m[0, a_t, 3, 3]):.3f}')

    # 4) 零初始化恒等 + M 无梯度 + gate/w_out 有梯度路径
    ich.reset()
    for n in NOVELTY_BLOCKS:
        ich.observe(n, feat)
    out = ich.suppress(CIC_BLOCK, feat)
    assert torch.equal(out, feat), '零初始化下 Δ 必须为 0'
    assert not ich.last_m.requires_grad, 'M 必须无梯度 (固化)'
    x = feat.clone().requires_grad_(True)
    ich.reset()
    for n in NOVELTY_BLOCKS:
        ich.observe(n, x)
    y = ich.suppress(CIC_BLOCK, x)
    y.square().mean().backward()
    assert ich.w_out.weight.grad is not None and torch.isfinite(ich.w_out.weight.grad).all()
    print('[ich-d selftest] zero-init identity + M no-grad + w_out grad path OK')
    os.unlink(tmp.name)
    print('[ich-d selftest] SELFTEST_OK')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    assert args.selftest, '本文件仅支持 --selftest 直跑; 训练经 config runners 进入'
    selftest()
