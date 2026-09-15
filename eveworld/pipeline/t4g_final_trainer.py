#!/usr/bin/env python3
"""终局合体臂 v2 (72 号 + 隔离判决修订): pretrain 底座纯自动闭环, 零结构改动。

隔离判决 (裸D 10.07 / 协议版 9.66 / 全开 13.27, 全差于 A1 6.83): 擦除子协同
训练 = 净负 (拐杖效应, 结构头吸走本该流入主干的纠正信号)。v2 **彻底移除
ICHModule** —— 无擦除子/无 hook 写回/零推理开销, checkpoint 就是裸 transformer。

组件:
  1. GT 干净 SFT 锚 (~3/4 样本, T4GCorrTransform 注入 vid);
  2. L_id (block22 InfoNCE, t4g_corr 成熟组件, σ∈[0.2,0.5], λ warmup→0.5;
     config t4g_lambdas='0.5,0.0' 关 L_change);
  3. A2 在线自滚雪球 (~1/4 样本, step1 起, λ_a2 前 50 步 warmup):
     GT latent 起步, K 轮大步幅 σ (geomspace hi→lo) 自我喂养造现场
     (前 K-1 轮 no_grad 且绕过 DS engine, 末轮带梯度走 engine);
     **冻结检测器 = 纯训练侧裁判** (16 维 LR, ICHDModule 实例仅作特征/打分器,
     不挂进模型/不进 optimizer/EMA/checkpoint) 对末轮 x̂0 指认涌现区 (M>0.7,
     末轮 σ=0.3 在标定带内); 指认格在线回填"出现前帧" x̂0 (detach) 作目标,
     修复损失**全部教给主干权重** (复制 A1 成功机制, 教材换全自动 on-policy)。

工程铁律 (smoke 三次死锁 py-spy 实锤后固化):
  - parse_losses 对 losses dict 每键各一次 gather -> 两分支**键集必须恒定**;
  - A2 的 no_grad 轮绕过 DeepSpeed engine (engine 调用计数各 rank 必须一致);
  - A2 分支补一次与 corr 同 shape 的零 stats gather (collective 序列对称)。

`--selftest` CPU 单测: σ 调度 / prefill 回填目标 / S5 (A2 关闭时键集恒等链)。
"""
from __future__ import annotations

import functools
import os

import numpy as np
import torch

from .t4g_corr_trainer import T4GCorrTrainer, T4GCorrTransform  # noqa: F401 (transform 注册)
from .t4g_ich_d_trainer import CIC_BLOCK, ICHDModule, NOVELTY_BLOCKS

T_LAT, H_LAT, W_LAT = 24, 30, 48

DEFAULT_WEIGHTS = '/data/datasets/gagi/eve_v2_outputs/selfcase/cic_match/ich_d_frozen_lr.npz'


# ====================================================================================
#  纯函数 (CPU 可单测)
# ====================================================================================
def a2_sigma_schedule(k, s_hi=3.0, s_lo=0.3):
    """K 轮大步幅 σ: 几何均匀覆盖高→低 (与 t4g_final_kprobe 同款)。"""
    return [float(s) for s in np.geomspace(s_hi, s_lo, k)]


def hit_upsample(hit):
    """hit (B,T,30,48) bool -> (B,1,T,60,96) float (latent 空间损失掩码)。"""
    up = hit.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return up.unsqueeze(1).float()


def pixel_prefill(pix, hit, feather=8):
    """像素精确回填 (align 对账 FAIL 后指令: case builder 同款逻辑的在线版):
    指认格的像素帧段用该格"出现前"代表像素帧的同位块回填 + 边缘羽化。
    pix: (B,3,NF,Hp,Wp) fp32; hit: (B,T,30,48) bool; 格=16px, latent 帧 t 的
    像素段 = rep 相邻中点区间。返回 (pix_fill, alpha (B,1,NF,Hp,Wp))。"""
    import torch.nn.functional as TF
    B, C, NF, Hp, Wp = pix.shape
    T, Hc, Wc = hit.shape[1:]
    ch, cw = Hp // Hc, Wp // Wc
    rep = np.linspace(0, NF - 1, T).astype(int)
    mids = (rep[:-1] + rep[1:]) // 2 + 1
    seg_lo = np.concatenate([[0], mids])
    seg_hi = np.concatenate([mids, [NF]])
    out = pix.clone()
    hard = torch.zeros(B, 1, NF, Hp, Wp, device=pix.device, dtype=pix.dtype)
    last_clean = torch.zeros(B, Hc, Wc, dtype=torch.long, device=pix.device)
    for t in range(T):
        for b, gy, gx in torch.nonzero(hit[:, t]).tolist():
            tp = int(last_clean[b, gy, gx])               # 出现前帧 (兜底帧0=ref)
            y0, x0 = gy * ch, gx * cw
            src = pix[b, :, rep[tp], y0:y0 + ch, x0:x0 + cw].unsqueeze(1)
            out[b, :, seg_lo[t]:seg_hi[t], y0:y0 + ch, x0:x0 + cw] = src
            hard[b, 0, seg_lo[t]:seg_hi[t], y0:y0 + ch, x0:x0 + cw] = 1.0
        cur = torch.full_like(last_clean, t)
        last_clean = torch.where(hit[:, t], last_clean, cur)
    # 羽化 (空间 box blur 逐帧; clamp*1.2 保内芯≈1, 边缘线性过渡)
    k = 2 * feather + 1
    kernel = torch.ones(1, 1, k, k, device=pix.device, dtype=pix.dtype) / (k * k)
    a2d = hard.reshape(B * NF, 1, Hp, Wp)
    a2d = TF.conv2d(TF.pad(a2d, (feather,) * 4, mode='replicate'), kernel)
    alpha = (a2d.reshape(B, 1, NF, Hp, Wp) * 1.2).clamp(0, 1)
    return alpha * out + (1 - alpha) * pix, alpha


def prefill_target(x0_det, hit):
    """在线擦除目标: 指认格回填"出现前帧"同位 x̂0 (逐格最近的无指认帧, 兜底帧0)。
    x0_det: (B,z,T,60,96) detached latent;  hit: (B,T,30,48) bool。
    返回 (target 同 x0 形状, hit_up (B,1,T,60,96) float)。"""
    B, T, H, W = hit.shape
    dev = hit.device
    tgt_idx = torch.zeros(B, T, H, W, dtype=torch.long, device=dev)
    last_clean = torch.zeros(B, H, W, dtype=torch.long, device=dev)   # 帧0=ref 恒净
    for t in range(T):
        cur = torch.full_like(last_clean, t)
        tgt_idx[:, t] = torch.where(hit[:, t], last_clean, cur)
        last_clean = torch.where(hit[:, t], last_clean, cur)
    idx_up = tgt_idx.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    hit_up = hit.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    z = x0_det.shape[1]
    idx_g = idx_up.unsqueeze(1).expand(B, z, T, idx_up.shape[-2], idx_up.shape[-1])
    target = x0_det.gather(2, idx_g)
    return target, hit_up.unsqueeze(1).float()


# ====================================================================================
#  trainer
# ====================================================================================
class T4GFinalTrainer(T4GCorrTrainer):
    """SFT(L_diff+L_id) 3/4 + A2 在线滚雪球 1/4; 冻结检测器=纯训练侧裁判 (v2)。"""

    CORR_STATS_DIM = 11                     # t4g_corr._compute_corr 的 stats 长度

    # ------------------------------- models -------------------------------
    def get_models(self, model_config):
        model = super().get_models(model_config)          # corr: L_id/annos/哨兵
        weights_path = os.environ.get('T4G_ICH_D_WEIGHTS',
                                      model_config.get('t4g_ich_d_weights', DEFAULT_WEIGHTS))
        self.a2_p = float(os.environ.get('T4G_A2_P', model_config.get('t4g_a2_p', 0.25)))
        self.a2_k = int(os.environ.get('T4G_A2_K', model_config.get('t4g_a2_k', 4)))
        self.a2_sigma_hi = float(os.environ.get('T4G_A2_SIGMA_HI',
                                                model_config.get('t4g_a2_sigma_hi', 3.0)))
        self.a2_sigma_lo = float(os.environ.get('T4G_A2_SIGMA_LO',
                                                model_config.get('t4g_a2_sigma_lo', 0.3)))
        self.lambda_a2 = float(os.environ.get('T4G_LAMBDA_A2',
                                              model_config.get('t4g_lambda_a2', 1.0)))
        self.a2_warmup = int(os.environ.get('T4G_A2_WARMUP',
                                            model_config.get('t4g_a2_warmup', 50)))
        self.a2_m_thr = float(os.environ.get('T4G_A2_M_THR',
                                             model_config.get('t4g_a2_m_thr', 0.7)))
        ich_win = int(os.environ.get('T4G_ICH_WIN', model_config.get('t4g_ich_win', 3)))

        # ---- v2: 冻结检测器 = 纯训练侧裁判 —— 是 trainer 属性而非模型子模块:
        #      不进 optimizer/EMA/checkpoint, 模型零结构改动, 推理即裸 transformer。
        tf = model['transformer']
        judge = ICHDModule(channels=tf.config.model_channels, weights_path=weights_path,
                           win=ich_win)
        judge.to(self.device)                             # fp32 + GPU (仅特征/打分)
        judge.eval()
        judge.requires_grad_(False)
        self.a2_judge = judge
        self._judge_hook_done = False
        self._final_stat_buffer = []
        self._final_logged_step = -1
        self._a2_rng = np.random.default_rng(20260823 + int(os.environ.get('RANK', '0')))
        self.print('[t4g-final] v2 judge <- %s (frozen16d, 训练侧裁判, 不进模型) | '
                   'a2: p=%.2f K=%d sigma=[%.1f->%.1f] lambda=%.2f warmup=%d '
                   'm_thr=%.2f | trainable=backbone only (零结构改动)'
                   % (weights_path, self.a2_p, self.a2_k, self.a2_sigma_hi,
                      self.a2_sigma_lo, self.lambda_a2, self.a2_warmup, self.a2_m_thr))
        return model

    # ------------------------------- hooks --------------------------------
    def _ensure_judge_hooks(self):
        """裁判 observe hooks (只读, 不写回主干; 注册在 corr hooks 之后)。"""
        if self._judge_hook_done:
            return
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        tf = model['transformer']
        judge = self.a2_judge
        for name in NOVELTY_BLOCKS + (CIC_BLOCK,):
            tf.blocks[name].register_forward_hook(
                lambda _m, _i, out, _n=name: judge.observe(_n, out))
        self._judge_hook_done = True
        self.print('[t4g-final] judge hooks -> observe %s (只读; 无 suppress, '
                   '模型前向零改动)' % list(NOVELTY_BLOCKS + (CIC_BLOCK,)))

    # ------------------------------ helpers -------------------------------
    def _lambda_a2_warmup(self):
        return self.lambda_a2 * min(1.0, self.cur_step / max(1, self.a2_warmup))

    def _sym_zero_corr_gather(self):
        """A2 分支的对称 collective: 与 corr._compute_corr 的 gather 同 shape 同顺序。"""
        stats = torch.zeros(self.CORR_STATS_DIM, device=self.device)
        gathered = self.accelerator.gather(stats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._stat_buffer.append(gathered)            # corr 哨兵缓冲 (全零行)

    # ------------------------------ forward -------------------------------
    def forward_step(self, batch_dict):
        self._ensure_hook()                               # corr hooks (幂等)
        self._ensure_judge_hooks()                        # 裁判只读 hooks
        judge = self.a2_judge
        judge.reset()

        is_a2 = bool(self._a2_rng.random() < self.a2_p)
        # final 哨兵 stats (跨进程可加):
        # [0]n_sft [1]n_a2 [2](空) [3]Σm_a2(全图,R1) [4]Σhit_cells [5]n_a2_hit
        # [6]Σl_a2_raw [7]cnt_l_a2 [8](空) [9](空) [10]n_calls
        fstats = torch.zeros(11, device=self.device)
        if is_a2:
            losses = self._a2_forward(batch_dict, fstats)
            self._sym_zero_corr_gather()                  # 对称: corr gather 补位
        else:
            losses = super().forward_step(batch_dict)     # L_diff + L_id (内含 gather)
            fstats[0] += 1
        judge.reset()                                     # 释放特征引用 (只读裁判)
        # ---- collective 对称铁律 (smoke 三次死锁 py-spy 实锤定因) ----
        # giga_train.parse_losses 对 losses dict **每个键**各做一次
        # accelerator.gather —— 两分支键集不同 => 每 micro-step collective
        # 次数不同 => 与 backward allreduce 错配 => NCCL 死锁。
        # 修复: 固定键集 (顺序 = 构造序, 各 rank 恒一致), 缺位补零标量。
        zero = torch.zeros((), device=self.device)
        canon = dict(l_diff=zero, l_id=zero, l_change=zero, l_a2=zero)
        canon.update(losses)
        losses = canon
        with torch.no_grad():
            fstats[10] += 1
        gathered = self.accelerator.gather(fstats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._final_stat_buffer.append(gathered)
        return losses

    # ------------------------------- A2 -----------------------------------
    def _a2_rollout_step(self, x0, ref_latents, ref_masks, prompt_embeds, fps,
                         sigma, grad):
        """一轮自我喂养 (输入构造与 GigaWorld0Trainer.forward_step 逐位同款):
        x0 -> renoise(σ, 显式) -> 前向 -> denoise -> 帧0 ref 恒定。

        关键: no_grad 轮**绕过 DeepSpeed engine 直调解包模块** —— DS 的梯度
        accumulation boundary 按 engine.forward 调用计数判定; A2 rank 若每
        micro-step 多调 engine, 各 rank 计数漂移 -> backward reduce 时机错位
        -> NCCL 死锁 (smoke 实锤)。末轮带梯度前向必须走 engine (reduce 语义)。"""
        if grad:
            transformer = functools.partial(self.model, 'transformer')
        else:
            _m = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            transformer = _m['transformer']               # AC wrapper 保留, hooks 照常
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            sig = torch.tensor([[sigma]], device=self.device, dtype=torch.float32)
            input_latents, timesteps = self.edm_loss.add_noise(x0.float(), sigma=sig)
            # dtype 纪律: fp32 数学 -> 网络输入前统一 self.dtype (混拼会崩 cat/linear)
            ref_m = ref_masks.float()
            augment_sigma = torch.tensor([0.0001], device=self.device,
                                         dtype=torch.float32)
            while len(augment_sigma.shape) < len(ref_latents.shape):
                augment_sigma = augment_sigma.unsqueeze(-1)
            input_latents = ref_m * ref_latents.float() + (1 - ref_m) * input_latents
            in_masks = ref_m.repeat(1, 1, 1, input_latents.shape[-2],
                                    input_latents.shape[-1])
            input_latents = torch.cat([input_latents, in_masks], dim=1)
            timesteps = timesteps.float().view(1, 1, 1, 1, 1).expand(
                x0.size(0), -1, x0.size(2), -1, -1)
            t_cond = augment_sigma / (augment_sigma + 1)
            timesteps = ref_m * t_cond + (1 - ref_m) * timesteps
            B = x0.shape[0]
            padding_mask = torch.zeros(
                (B, 1, x0.shape[-2] * 8, x0.shape[-1] * 8),
                dtype=self.dtype, device=self.device)
            pred = transformer(x=input_latents.to(self.dtype),
                               timesteps=timesteps.to(self.dtype),
                               crossattn_emb=prompt_embeds.to(self.dtype),
                               padding_mask=padding_mask, fps=fps)
            x0_new = self.edm_loss.denoise(pred.float())
            x0_new = ref_m.float() * ref_latents.float() + \
                (1 - ref_m.float()) * x0_new
        return x0_new

    def _a2_forward(self, batch_dict, fstats):
        """A2 样本: GT latent 起步滚 K 轮 (前 K-1 no_grad), 裁判指认末轮 x̂0
        涌现区 (M>0.7, 末轮 σ=0.3 在裁判标定带内), 指认格回填出现前帧作目标
        (detach) -> 修复损失全部教给主干。"""
        judge = self.a2_judge
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']
        fps = batch_dict['fps'][0]
        latents_gt = self.forward_vae(images).float()
        ref_latents = self.forward_vae(batch_dict['ref_images'])
        ref_masks = batch_dict['ref_masks']

        sigmas = a2_sigma_schedule(self.a2_k, self.a2_sigma_hi, self.a2_sigma_lo)
        x0 = latents_gt
        for k, s in enumerate(sigmas):
            last = (k == len(sigmas) - 1)
            judge.reset()
            x0 = self._a2_rollout_step(x0, ref_latents, ref_masks, prompt_embeds,
                                       fps, s, grad=last)
            if not last:
                x0 = x0.detach()

        with torch.no_grad():
            m = judge.detect()                            # 末轮特征 -> (B,T,30,48)
        hit = (m > self.a2_m_thr)
        n_hit = int(hit[:, 1:].sum())                     # 帧0=ref 不计
        hit = torch.cat([torch.zeros_like(hit[:, :1]), hit[:, 1:]], dim=1)
        lam = self._lambda_a2_warmup()
        if n_hit > 0:
            # ---- 像素精确目标 (align 对账 FAIL 修复指令): 末轮 x̂0 解码到像素 ->
            #      病灶区出现前帧回填+羽化 (case builder 同款) -> 整段重编码。
            #      = A1 双编码机制的在线版; 仅 A2 样本, no_grad, 目标天然 detach。
            with torch.no_grad():
                raw = x0.detach().to(self.dtype) / self.latents_std + self.latents_mean
                pix = self.vae.decode(raw).sample.float().clamp(-1, 1)   # (B,3,NF,H,W)
                pix_fill, _ = pixel_prefill(pix, hit)
                z_tgt = self.forward_vae(
                    pix_fill.clamp(-1, 1).permute(0, 2, 1, 3, 4).to(self.dtype)).float()
            hit_up = hit_upsample(hit)
            err = (x0 - z_tgt) ** 2 * hit_up
            l_a2_raw = err.sum() / (hit_up.sum() * x0.shape[1] + 1e-8)
            with torch.no_grad():
                fstats[5] += 1
                fstats[6] += float(l_a2_raw.detach())
                fstats[7] += 1
        else:
            l_a2_raw = (x0 * 0).mean()                    # 连图零 (无命中样本)
        with torch.no_grad():
            fstats[1] += 1
            fstats[3] += float(m.mean())
            fstats[4] += n_hit
        return dict(l_a2=lam * l_a2_raw)

    # ------------------------------ sentinel ------------------------------
    def print_step(self):
        super().print_step()                              # corr [t4g-sentinel]
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        if self.cur_step == self._final_logged_step:
            return
        self._final_logged_step = self.cur_step
        agg = (torch.stack(self._final_stat_buffer).sum(0)
               if self._final_stat_buffer else torch.zeros(11))
        self._final_stat_buffer = []

        def _r(n, d):
            return (agg[n] / agg[d]).item() if agg[d] > 0 else float('nan')

        n_sft, n_a2, n_a2_hit = int(agg[0]), int(agg[1]), int(agg[5])
        msg = ('[a2-sentinel] step=%d sft=%d a2=%d a2_hit=%d | hit_cells/a2=%.1f '
               'l_a2=%.4e lambda_a2=%.3f K=%d | m_a2=%.4f (R1: 骤降=报警) '
               '| judge=frozen16d (v2 无擦除子)'
               % (self.cur_step, n_sft, n_a2, n_a2_hit, _r(4, 1), _r(6, 7),
                  self._lambda_a2_warmup(), self.a2_k, _r(3, 1)))
        self.logger.info(msg)
        print(msg, flush=True)


# ====================================================================================
#  selftest (CPU): σ 调度 / prefill 目标 / S5 静态恒等链
# ====================================================================================
def selftest():
    import functools as ft
    import operator as op
    torch.manual_seed(0)
    print('[final selftest] start')
    # 1) σ 调度
    s = a2_sigma_schedule(4)
    assert len(s) == 4 and s[0] == 3.0 and abs(s[-1] - 0.3) < 1e-9
    assert all(s[i] > s[i + 1] for i in range(3))

    # 2) prefill 目标: 命中格回填最近无指认帧, 兜底帧0; 未中格目标=自身 (零损失)
    B, z, T, H, W = 1, 4, 6, 4, 6
    x0 = torch.arange(T).float().view(1, 1, T, 1, 1).expand(B, z, T, 2 * H, 2 * W).clone()
    hit = torch.zeros(B, T, H, W, dtype=torch.bool)
    hit[0, 3, 1, 2] = True                                # t=3 命中 -> 回填 t=2
    hit[0, 4, 1, 2] = True                                # t=4 连续命中 -> 仍回填 t=2
    hit[0, 2, 0, 0] = True                                # 另一格 t=2 -> 回填 t=1
    tgt, hit_up = prefill_target(x0, hit)
    assert tgt.shape == x0.shape and hit_up.shape == (B, 1, T, 2 * H, 2 * W)
    assert float(tgt[0, 0, 3, 2, 4]) == 2.0 and float(tgt[0, 0, 4, 2, 4]) == 2.0
    assert float(tgt[0, 0, 2, 0, 0]) == 1.0
    assert float(tgt[0, 0, 5, 2, 4]) == 5.0               # 未命中 -> 自身
    assert float(hit_up.sum()) == 3 * 4                   # 3 命中格 x 2x2 上采样
    # 全命中列 -> 兜底帧0
    hit2 = torch.zeros_like(hit); hit2[0, 1:, 2, 3] = True
    tgt2, _ = prefill_target(x0, hit2)
    assert all(float(tgt2[0, 0, t, 4, 6]) == 0.0 for t in range(1, T))
    print('[final selftest] prefill target OK')

    # 2b) pixel_prefill: 病灶像素段被出现前帧同位块覆盖 + 羽化; 区外逐位不变
    NF, Hp, Wp = 24, 64, 96
    Tq, Hc, Wc = 6, 4, 6
    pixv = torch.arange(NF).float().view(1, 1, NF, 1, 1).expand(1, 3, NF, Hp, Wp).clone()
    hitp = torch.zeros(1, Tq, Hc, Wc, dtype=torch.bool)
    hitp[0, 3, 1, 2] = True                               # t=3 -> t_pre=2
    pf, alpha = pixel_prefill(pixv, hitp, feather=4)
    rep = np.linspace(0, NF - 1, Tq).astype(int)
    mids = (rep[:-1] + rep[1:]) // 2 + 1
    slo = np.concatenate([[0], mids])[3]
    assert float(alpha[0, 0, slo, 24, 40]) > 0.99, '回填区内芯 alpha 应≈1'
    assert float(alpha[0, 0, slo, 0, 0]) == 0.0, '区外 alpha 应=0'
    assert torch.equal(pf[0, :, :, 0:8, 0:8], pixv[0, :, :, 0:8, 0:8]), '区外逐位不变'
    got = float(pf[0, 0, slo, 24, 40])
    want = float(rep[2])                                  # 回填 = t_pre=2 的代表帧值
    assert abs(got - want) < 0.6, (got, want)
    print(f'[final selftest] pixel_prefill OK (fill={got:.2f}≈rep[t_pre]={want}, '
          f'feathered edge, outside untouched)')

    # 3) S5 v2: A2 关闭时与裸 SFT 恒等链 —— 裁判是只读 observe (无 suppress 写回,
    #    模型前向零改动) + canon 键集补零不改 total_loss
    import json as _json
    import tempfile
    rngn = np.random.RandomState(0)
    tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    np.savez(tmp.name, w=rngn.randn(16), b=np.array([0.0]), mu=rngn.randn(16) * 0.1,
             sd=rngn.rand(16) + 0.5, meta=np.array(_json.dumps({}), dtype=object))
    judge = ICHDModule(channels=32, weights_path=tmp.name)
    judge.requires_grad_(False)
    for n, p in list(judge.named_parameters()) + list(judge.named_buffers()):
        assert p.dim() >= 1 and ft.reduce(op.mul, p.shape) == p.numel(), n
    feat = torch.randn(1, T_LAT, H_LAT, W_LAT, 32)
    for n in NOVELTY_BLOCKS + (CIC_BLOCK,):
        judge.observe(n, feat)                            # v2: 只读, 不改前向
    with torch.no_grad():
        m = judge.detect()
    assert m.shape == (1, T_LAT, H_LAT, W_LAT) and not m.requires_grad
    # canon 键集: 补零不改 total (S5 恒等的数值部分)
    l_diff = torch.randn(2).abs()
    losses = dict(l_diff=l_diff, l_id=torch.zeros(()), l_change=torch.zeros(()))
    zero = torch.zeros(())
    canon = dict(l_diff=zero, l_id=zero, l_change=zero, l_a2=zero)
    canon.update(losses)
    assert list(canon) == ['l_diff', 'l_id', 'l_change', 'l_a2'], list(canon)
    total_sft = sum(v.mean() for v in losses.values())
    total_canon = sum(v.mean() for v in canon.values())
    assert torch.equal(total_sft, total_canon), 'canon 补零键不得改 total_loss'
    os.unlink(tmp.name)
    print('[final selftest] S5 v2 (只读裁判 + canon 键集恒等) + R7 no-0dim OK')
    print('[final selftest] SELFTEST_OK')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    assert args.selftest, '本文件仅支持 --selftest 直跑; 训练经 config runners 进入'
    selftest()
