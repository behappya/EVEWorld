#!/usr/bin/env python3
"""ICH 训练 (71 号 B1 G3): Instance Conservation Head = 学习型检测子 + zero-init 抑制子。

结构 (挂在主干 block 上, 层选择由 G1' 数据裁决: 中层 10-16 最可读, 非草案的 25):
  检测子: forward hook 抓 blocks {10,12,16} (臂B) 或 {16} (臂A) 输出 (B,T,H,W,D),
    逐格算 4 维瞬态特征/层 (照抄 selfcase_g1p.cell_features 定义, 窗口逐帧滑动):
      [窗内最低同位前后帧 cos, 窗内平均同位 cos, 窗内最低 3x3 邻域最大匹配, 窗内平均与帧0同位 cos]
    拼接 (4L 通道) 过 2 层 3x3 conv 头 -> 涌现 logits -> M=sigmoid (B,24,30,48)。
  抑制子: 在最后一个选中 block (block16) 输出处加负残差
      Δ = -sigmoid(gate) ⊙ M ⊙ W_out(h16),  W_out zero-init (照抄 cic_transport
      nn.init.zeros_ 安全带) -> 初始前向与 A1 逐位等价。

训练 (T4GICHTrainer, 子类 T4GSelfCaseTrainer=T4GAugTrainer 行为):
  总 loss = A1 自病例修复损失 (原样) + λ_det × BCE(M, 病例掩码)。
  病例掩码由 batch 的 paste_lat 生成 (病例格=1, 非病例样本全 0);
  paste_lat 是 VAE ÷8 (60x96) 坐标, 特征网格 30x48 再 ÷2 (patchify 2x), 上界取 ceil。
  λ_det=1.0, warmup 50 步内线性 0→1。可训参数 = ICH + 主干全参 (与 A1 同)。

设计要点 (对抗检查后):
  - hook 挂 accelerate.prepare 之后的 (若启用 AC) CheckpointWrapper 外层 (照抄
    t4g_corr._ensure_hook): 输出与自动求导连图, 且 hook 只在原始前向执行一次,
    避开 41 号 ISOA 的 "共享 dict 状态被 checkpoint 重放清零" 坑;
    ICH 计算不在 checkpoint 区内 -> Δ 梯度路径正常。
  - 检测特征默认在 detach 后的 block 输出上算 (fp32): BCE 只训 conv 头, 主干梯度
    与 A1 完全同源 (修复损失 + Δ 路径); env T4G_ICH_DET_ATTACH=1 可切端到端。
  - 哨兵 [ich-sentinel]: det_pos/det_neg (病例格/背景格 M 均值, 应分离) +
    det_neg_clean (干净样本 M, 应低) + bce/λ/gate/wout_norm; 父类 [aug-sentinel]
    (ret_proxy 等) 原样保留。

`--selftest` 为 CPU 单测 (特征维度/零初始化恒等/掩码生成), 提交前干跑。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as TF
from torch import nn

from .t4g_aug_paste import T_LAT, H_LAT
from .t4g_selfcase_trainer import T4GSelfCaseTrainer

W_LAT = 48

DEFAULT_ICH_BLOCKS = 'block10,block12,block16'


# ====================================================================================
#  ICH 模块
# ====================================================================================
class ICHModule(nn.Module):
    """检测子 (4L 通道瞬态特征 + 2 层 3x3 conv 头) + 抑制子 (zero-init 负残差)。"""

    def __init__(self, channels, block_names, hidden=96, win=3, det_detach=True):
        super().__init__()
        self.block_names = tuple(block_names)
        assert len(self.block_names) >= 1
        self.win = int(win)
        self.det_detach = bool(det_detach)
        c_in = 4 * len(self.block_names)
        self.det_conv1 = nn.Conv2d(c_in, hidden, 3, padding=1)
        self.det_conv2 = nn.Conv2d(hidden, 1, 3, padding=1)
        # 注意: 必须是 1 维 (1,) 而非 0 维标量 —— giga_train EMAModel.state_dict()
        # 的 world_size>1 分支对 0 维参数做 reduce(mul, torch.Size([])) 抛
        # "reduce() of empty iterable" (8 卡首档 step25 存档必崩, 已实锤)。
        self.gate = nn.Parameter(torch.zeros(1))           # sigmoid(0)=0.5
        self.w_out = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.w_out.weight)                  # cic_transport 同款 zero-init
        self._feats = {}                                   # 本次前向抓到的 block 输出
        self.last_logits = None                            # (B,T,H,W) fp32, 连图

    def num_det_params(self):
        return sum(p.numel() for p in self.det_conv1.parameters()) + \
            sum(p.numel() for p in self.det_conv2.parameters())

    def reset(self):
        self._feats = {}
        self.last_logits = None

    @staticmethod
    def transient_features(feat, win=3):
        """feat: (B,T,H,W,D) float32 -> (B,T,H,W,4) float32。
        照抄 selfcase_g1p 逐格 4 维定义; 每帧 t 的窗口 = g1p.dup_window(t, win)
        (即 [t-1, t-1+win) 裁到可算 t-1 自相似的 [1, T))。"""
        B, T, H, W, _ = feat.shape
        assert T >= 2, 'ICH 需要 >=2 个 latent 帧'
        fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_prev = (fn[:, 1:] * fn[:, :-1]).sum(-1)        # (B,T-1,H,W)
        sim0 = (fn * fn[:, :1]).sum(-1)                    # (B,T,H,W)
        cur, prev = fn[:, 1:], fn[:, :-1]
        shifted = []                                       # 9 个平移相似图, 全 out-of-place
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cy = slice(max(0, -dy), H - max(0, dy))
                cx = slice(max(0, -dx), W - max(0, dx))
                py = slice(max(0, dy), H - max(0, -dy))
                px = slice(max(0, dx), W - max(0, -dx))
                s = (cur[:, :, cy, cx] * prev[:, :, py, px]).sum(-1)
                # 回填到 (H,W) 全图, 越界邻居填 -2 (排除出 max), 连图安全 (无原位写)
                shifted.append(TF.pad(
                    s, (max(0, -dx), max(0, dx), max(0, -dy), max(0, dy)), value=-2.0))
        nb = torch.stack(shifted, dim=0).amax(dim=0)       # (B,T-1,H,W)
        rows = []
        for t in range(T):
            lo = max(1, min(t - 1, T - 1))                 # g1p.dup_window 同款
            hi = max(lo + 1, min(lo + win, T))
            sp = sim_prev[:, lo - 1:hi - 1]
            rows.append(torch.stack([
                sp.amin(dim=1), sp.mean(dim=1),
                nb[:, lo - 1:hi - 1].amin(dim=1),
                sim0[:, lo:hi].mean(dim=1)], dim=-1))      # (B,H,W,4)
        return torch.stack(rows, dim=1)                    # (B,T,H,W,4)

    def observe(self, name, out):
        """非末位 block 的 hook: 只缓存特征 (连图, 不 detach)。"""
        self._feats[name] = out

    def detect(self):
        """用缓存特征算涌现 logits (B,T,H,W); fp32 输出并缓存供 BCE。"""
        feats = []
        for n in self.block_names:
            h = self._feats[n]
            assert h is not None, f'ICH: block 特征缺失 {n}'
            src = h.detach() if self.det_detach else h
            feats.append(self.transient_features(src.float(), self.win))
        f4 = torch.cat(feats, dim=-1)                      # (B,T,H,W,4L) fp32
        B, T, H, W, C = f4.shape
        x = f4.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        x = x.to(self.det_conv1.weight.dtype)
        logits = self.det_conv2(TF.silu(self.det_conv1(x))).reshape(B, T, H, W)
        self.last_logits = logits.float()
        return logits

    def suppress(self, name, out):
        """末位 block 的 hook: 检测 + 负残差 Δ = -sigmoid(gate)·M·W_out(h)。
        W_out zero-init -> 初始 Δ 恒等于 0 (前向与无 ICH 逐位一致)。"""
        self.observe(name, out)
        logits = self.detect()
        m = torch.sigmoid(logits).to(out.dtype).unsqueeze(-1)   # (B,T,H,W,1)
        delta = -torch.sigmoid(self.gate).to(out.dtype) * m * self.w_out(out)
        self._feats = {}                                   # 释放引用 (logits 已缓存)
        return out + delta


# ====================================================================================
#  trainer
# ====================================================================================
class T4GICHTrainer(T4GSelfCaseTrainer):
    """A1 修复损失 (原样, 父类) + λ_det × BCE(M, 病例掩码) + [ich-sentinel]。"""

    # ------------------------------- models -------------------------------
    def get_models(self, model_config):
        model = super().get_models(model_config)
        blocks_s = os.environ.get('T4G_ICH_BLOCKS',
                                  model_config.get('t4g_ich_blocks', DEFAULT_ICH_BLOCKS))
        self.ich_blocks = tuple(s.strip() for s in str(blocks_s).split(',') if s.strip())
        hidden = int(os.environ.get('T4G_ICH_HIDDEN', model_config.get('t4g_ich_hidden', 96)))
        win = int(os.environ.get('T4G_ICH_WIN', model_config.get('t4g_ich_win', 3)))
        det_detach = os.environ.get('T4G_ICH_DET_ATTACH', '0') != '1'
        self.lambda_det = float(os.environ.get('T4G_LAMBDA_DET',
                                               model_config.get('t4g_lambda_det', 1.0)))
        self.det_warmup = int(os.environ.get('T4G_DET_WARMUP',
                                             model_config.get('t4g_det_warmup', 50)))
        self.det_pos_weight = float(os.environ.get('T4G_DET_POS_WEIGHT',
                                                   model_config.get('t4g_det_pos_weight', 1.0)))

        tf = model['transformer']
        for b in self.ich_blocks:
            assert b in tf.blocks, f'未知 block: {b} (可用 {list(tf.blocks)[:4]}...)'
        ich = ICHModule(channels=tf.config.model_channels, block_names=self.ich_blocks,
                        hidden=hidden, win=win, det_detach=det_detach)
        for n, p in ich.named_parameters():                # EMA 多卡存档不兼容 0 维参数
            assert p.dim() >= 1, f'ICH 参数 {n} 不得为 0 维 (EMAModel.state_dict 会崩)'
        ich.to(self.dtype)
        tf.ich = ich          # 注册为子模块 -> 参数进 optimizer/EMA/checkpoint
        self._ich_hook_done = False
        self._ich_stat_buffer = []
        self._ich_logged_step = -1
        self.print('[t4g-ich] blocks=%s hidden=%d win=%d det_detach=%s det_params=%d '
                   'wout_params=%d lambda_det=%.2f warmup=%d pos_weight=%.1f'
                   % (','.join(self.ich_blocks), hidden, win, det_detach,
                      ich.num_det_params(), ich.w_out.weight.numel(),
                      self.lambda_det, self.det_warmup, self.det_pos_weight))
        return model

    # ------------------------------- hooks --------------------------------
    def _ensure_ich_hooks(self):
        """惰性注册 (照抄 t4g_corr._ensure_hook): prepare 后解包取模块引用,
        hook 挂 (若 AC) CheckpointWrapper 外层 -> 连图且只在原始前向执行一次。
        末位 block 的 hook 返回修改后的输出 (抑制子写回主干)。"""
        if self._ich_hook_done:
            return
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        tf = model['transformer']
        ich = tf.ich
        last = self.ich_blocks[-1]
        for name in self.ich_blocks[:-1]:
            tf.blocks[name].register_forward_hook(
                lambda _m, _i, out, _n=name: ich.observe(_n, out))
        tf.blocks[last].register_forward_hook(
            lambda _m, _i, out: ich.suppress(last, out))
        self._ich_hook_done = True
        self.print('[t4g-ich] hooks -> observe %s + suppress %s (%s)'
                   % (list(self.ich_blocks[:-1]), last, type(tf.blocks[last]).__name__))

    # ------------------------------- mask ---------------------------------
    @staticmethod
    def build_case_mask(paste_lat, has_paste, t_lat=T_LAT, h_lat=H_LAT, w_lat=W_LAT):
        """paste_lat (B,6)=[lt_lo,lt_hi,ly0,ly1,lx0,lx1] (VAE ÷8 60x96 坐标) ->
        特征网格病例掩码 (B,24,30,48) {0,1}; 空间 ÷2 (patchify), 上界 ceil。"""
        B = len(has_paste)
        m = torch.zeros(B, t_lat, h_lat, w_lat)
        for b in range(B):
            if not int(has_paste[b]):
                continue
            lt0, lt1, ly0, ly1, lx0, lx1 = [int(v) for v in paste_lat[b]]
            fy0, fy1 = ly0 // 2, min(h_lat, (ly1 + 1) // 2)
            fx0, fx1 = lx0 // 2, min(w_lat, (lx1 + 1) // 2)
            m[b, lt0:lt1, fy0:fy1, fx0:fx1] = 1.0
        return m

    def _lambda_det_warmup(self):
        return self.lambda_det * min(1.0, self.cur_step / max(1, self.det_warmup))

    # ------------------------------ forward -------------------------------
    def forward_step(self, batch_dict):
        self._ensure_ich_hooks()
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        ich = model['transformer'].ich
        ich.reset()
        losses = super().forward_step(batch_dict)          # A1 修复损失 (前向触发 hook)
        logits = ich.last_logits                           # (B,T,H,W) fp32, 连图
        assert logits is not None, 'ICH hook 未触发 (前向未经过选中 block?)'
        mask = self.build_case_mask(batch_dict['paste_lat'],
                                    batch_dict['has_paste']).to(logits.device)
        assert logits.shape == mask.shape, (logits.shape, mask.shape)
        pw = None
        if self.det_pos_weight != 1.0:
            pw = torch.tensor(self.det_pos_weight, device=logits.device)
        bce = TF.binary_cross_entropy_with_logits(logits, mask, pos_weight=pw)
        lam = self._lambda_det_warmup()
        losses['l_det'] = lam * bce

        # ---- [ich-sentinel] 统计 (跨进程可加; 每 micro-step 对称 gather 一次) ----
        # 布局: [0]n_case [1]Σdet_pos [2]Σdet_neg(case) [3]n_samples [4]Σbce
        #       [5]Σdet_neg_clean [6]Σgate [7]Σwout_norm [8]n_calls
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            stats = torch.zeros(9, device=logits.device)
            for b in range(logits.shape[0]):
                mb = mask[b] > 0.5
                stats[3] += 1
                if mb.any():
                    stats[0] += 1
                    stats[1] += float(probs[b][mb].mean())
                    stats[2] += float(probs[b][~mb].mean())
                else:
                    stats[5] += float(probs[b].mean())
            stats[4] += float(bce.detach())
            stats[6] += float(torch.sigmoid(ich.gate.detach().float()))
            stats[7] += float(ich.w_out.weight.detach().float().norm())
            stats[8] += 1
        gathered = self.accelerator.gather(stats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._ich_stat_buffer.append(gathered)
        return losses

    # ------------------------------ sentinel ------------------------------
    def print_step(self):
        super().print_step()                               # [aug-sentinel] 原样
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        if self.cur_step == self._ich_logged_step:
            return
        self._ich_logged_step = self.cur_step
        agg = (torch.stack(self._ich_stat_buffer).sum(0)
               if self._ich_stat_buffer else torch.zeros(9))
        self._ich_stat_buffer = []

        def _r(n, d):
            return (agg[n] / agg[d]).item() if agg[d] > 0 else float('nan')

        n_case, n_all = int(agg[0]), int(agg[3])
        det_neg_clean = (agg[5] / (agg[3] - agg[0])).item() if agg[3] > agg[0] else float('nan')
        msg = ('[ich-sentinel] step=%d cases=%d/%d | det_pos=%.4f det_neg=%.4f '
               'det_neg_clean=%.4f | bce=%.4e lambda_det=%.3f | gate=%.3f '
               'wout_norm=%.4e | blocks=%s'
               % (self.cur_step, n_case, n_all, _r(1, 0), _r(2, 0), det_neg_clean,
                  _r(4, 8), self._lambda_det_warmup(), _r(6, 8), _r(7, 8),
                  ','.join(self.ich_blocks)))
        self.logger.info(msg)
        print(msg, flush=True)


# ====================================================================================
#  selftest (CPU, 无训练栈): 特征维度 / 零初始化恒等 / 掩码生成 / 植入信号可读
# ====================================================================================
def selftest():
    torch.manual_seed(0)
    print('[ich selftest] start')
    B, T, H, W, D = 1, T_LAT, H_LAT, W_LAT, 32

    # 1) 掩码生成: ÷2 映射 + ceil 上界 + 非病例全 0
    lat = torch.tensor([[5, 12, 11, 23, 40, 61], [0, 0, 0, 0, 0, 0]], dtype=torch.long)
    m = T4GICHTrainer.build_case_mask(lat, [1, 0])
    assert m.shape == (2, T, H, W)
    assert m[1].sum() == 0
    assert m[0, 4].sum() == 0 and m[0, 12].sum() == 0        # 帧窗 [5,12)
    assert m[0, 5, 5, 20] == 1 and m[0, 5, 5, 19] == 0       # x: [20, 31)
    assert m[0, 5, 11, 30] == 1 and m[0, 5, 12, 30] == 0     # y: [5, 12)
    assert m[0, 5, 4, 20] == 0
    exp = (12 - 5) * (12 - 5) * (31 - 20)
    assert int(m[0].sum()) == exp, (int(m[0].sum()), exp)

    # 2) 植入瞬态: 背景恒定+微噪, 病例格在 a_t 帧切换向量 -> f1 (min selfsim) 骤降
    base = torch.randn(1, 1, H, W, D)
    feat = base.repeat(1, T, 1, 1, 1) + 0.05 * torch.randn(1, T, H, W, D)
    a_t, gy, gx = 8, 10, 20
    feat[0, a_t:, gy, gx] = torch.randn(D) + 0.05 * torch.randn(T - a_t, D)
    f4 = ICHModule.transient_features(feat.float())
    assert f4.shape == (1, T, H, W, 4)
    dip = f4[0, a_t, gy, gx, 0]
    bg = f4[0, a_t, gy, gx + 5, 0]
    print(f'[ich selftest] planted dip f1={dip:.3f} vs bg f1={bg:.3f}')
    assert dip < 0.5 < bg

    # 3) 零初始化恒等: suppress 输出与输入逐位一致
    for names in (('block16',), ('block10', 'block12', 'block16')):
        ich = ICHModule(channels=D, block_names=names, hidden=96)
        for n in names[:-1]:
            ich.observe(n, feat)
        out = ich.suppress(names[-1], feat)
        assert torch.equal(out, feat), '零初始化下 Δ 必须为 0'
        assert ich.last_logits.shape == (1, T, H, W)
        print(f'[ich selftest] {len(names)}L det_params={ich.num_det_params()} '
              f'zero-init identity OK')

    # 4) BCE 维度 + 梯度可达 conv 头
    ich = ICHModule(channels=D, block_names=('block16',), hidden=96)
    _ = ich.suppress('block16', feat.requires_grad_(False))
    bce = TF.binary_cross_entropy_with_logits(ich.last_logits, m[:1])
    bce.backward()
    g = ich.det_conv1.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0

    # 5) EMA 多卡存档兼容: 无 0 维参数 (EMAModel.state_dict world_size>1 分支对
    #    torch.Size([]) 做 reduce(mul) 会崩 —— step25 首档实锤的回归护栏)
    import functools as _ft
    import operator as _op
    for n, p in ich.named_parameters():
        assert p.dim() >= 1, f'0 维参数 {n} 会崩 EMA 存档'
        assert _ft.reduce(_op.mul, p.shape) == p.numel()
    print('[ich selftest] EMA save compat OK (no 0-dim params)')
    print('[ich selftest] SELFTEST_OK')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    assert args.selftest, '本文件仅支持 --selftest 直跑; 训练经 config runners 进入'
    selftest()
