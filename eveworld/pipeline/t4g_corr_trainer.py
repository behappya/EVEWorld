#!/usr/bin/env python3
"""EVE · Track4Gen 式对应监督 trainer (43 号 §6 集成规格; 替代 42 号四道 loss)。

据 43 号实证链: 相似度检测复制 (旧 L_cnt/L_neg) 被 E5/E6 判死; 换成两道实证支撑的
L_id (block22, 前后帧接力, E4 EPE0.55) + L_change (block25, 前后帧自相似, E7 gap+0.45)。
**双层 hook** —— L_id 吃 block22 特征, L_change 吃 block25 特征。

子类 GigaWorld0Trainer, 与父类差异集中在四处:
  1. get_models: 复用父类 (self.vae/self.edm_loss/self.train_mode), 再挂载对应 loss 所需的
     标注缓存 (anno) / data_index→vid 映射 / 时间档 / λ warmup(2 值) / 哨兵探针; **不改模型
     结构** (纯特征上算 loss, 无新参数)。
  2. forward_step: 逐行复刻父类 L_diff 通路 + register_forward_hook 抓 block22 & block25
     两层输出 (B,T,H,W,D) + 读该 batch 各样本 vid 取 anno + 算两道 loss。仅当采样
     sigma∈[0.2,0.5] 且该样本有 anno 时施加 (否则纯 L_diff)。λ 由 warmup(0→(0.5,0.4)) 决定。
  3. print_step: 四哨兵 (每 log_interval) —— L_id/L_change 裸值 + 生效帧数 nf / **B 格闸状态
     (active 帧 vs released 帧, 验框重叠闸)** / 参数位移 (固定采样 1000 参数 L2) /
     [第四哨兵=50 步探针纪律, 在 launch]。
  4. save_checkpoint_step: 存档起点门控 (checkpoint_start_step, 与 ISOA 同机制)。

数据对齐 (42 号 §3 关键点, 见报告)
----------------------------------
  packed_data 每样本 transform 前含 {data_index, prompt, video, prompt_embeds}, **不含 vid**,
  且 8 组 prompt 在 92 条里重复 (按 prompt 匹配 anno 对这 16 条有歧义)。故采用**权威
  data_index→vid 映射** (由 packed 视频 md5 == raw <vid>.mp4 精确匹配预生成, 见
  t4g_anno/_packidx2vid.json): 自定义 transform T4GCorrTransform 读 data_index 注入 vid 字段
  (DefaultCollator 原样透传 str -> batch['vid']=list[str])。prompt 匹配仅作缺失兜底。
"""
from __future__ import annotations

import functools
import glob
import json
import os

import torch
from giga_train import TRANSFORMS

from giga_world_0.giga_world_0_trainer import GigaWorld0Trainer
from giga_world_0.giga_world_0_transforms import GigaWorld0Transform

from .t4g_corr_loss import compute_corr_losses

DEFAULT_ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
DEFAULT_IDX2VID = os.path.join(DEFAULT_ANNO_DIR, '_packidx2vid.json')


# ====================================================================================
#  自定义 transform: 注入 vid (data_index -> vid 权威映射)
# ====================================================================================
@TRANSFORMS.register
class T4GCorrTransform(GigaWorld0Transform):
    """在父 transform 输出上追加 `vid` 字段 (str), 供 forward_step 取 anno。

    主键 = data_index -> vid (md5 权威映射, idx2vid_path)。缺映射时兜底按 prompt->vid
    (anno 里有 prompt; 重复 prompt 会歧义, 仅作最后兜底)。都失败则 vid='-1' (跳过对应 loss)。
    """

    def __init__(self, num_frames, height, width, image_cfg, fps=16,
                 idx2vid_path=DEFAULT_IDX2VID, anno_dir=DEFAULT_ANNO_DIR):
        super().__init__(num_frames=num_frames, height=height, width=width,
                         image_cfg=image_cfg, fps=fps)
        self.idx2vid = {}
        if idx2vid_path and os.path.exists(idx2vid_path):
            raw = json.load(open(idx2vid_path))
            self.idx2vid = {int(k): str(v) for k, v in raw.items()}
        # prompt->vid 兜底 (重复 prompt 只保留任一, 打标以便报告)
        self.prompt2vid = {}
        self._prompt_dup = set()
        if anno_dir and os.path.isdir(anno_dir):
            for fp in glob.glob(os.path.join(anno_dir, '*.json')):
                base = os.path.basename(fp)
                if base.startswith('_'):
                    continue
                try:
                    a = json.load(open(fp))
                except Exception:
                    continue
                pr, vid = a.get('prompt'), a.get('vid')
                if pr is None or vid is None:
                    continue
                if pr in self.prompt2vid:
                    self._prompt_dup.add(pr)
                else:
                    self.prompt2vid[pr] = str(vid)

    def _resolve_vid(self, data_dict):
        di = data_dict.get('data_index', None)
        if di is not None:
            try:
                vid = self.idx2vid.get(int(di))
                if vid is not None:
                    return vid
            except (TypeError, ValueError):
                pass
        pr = data_dict.get('prompt', None)
        if pr is not None and pr not in self._prompt_dup:
            vid = self.prompt2vid.get(pr)
            if vid is not None:
                return vid
        return '-1'

    def __call__(self, data_dict):
        out = super().__call__(data_dict)
        out['vid'] = self._resolve_vid(data_dict)
        return out


# ====================================================================================
#  哨兵: 参数位移探针 (固定采样 ~1000 标量, 相对 init 的 L2)
# ====================================================================================
class ParamDisplacementProbe:
    """采样 20 张参数张量 × 每张 50 个平坦索引 (~1000 标量), 报告相对 init 的 L2 位移。"""

    def __init__(self, module, num_tensors=20, per_tensor=50, seed=20260720):
        named = [(n, p) for n, p in module.named_parameters() if p.numel() > 0]
        assert len(named) > 0
        gen = torch.Generator().manual_seed(seed)
        stride = max(1, len(named) // num_tensors)
        picked = named[::stride][:num_tensors]
        self.specs = []
        init_vals = []
        for name, p in picked:
            k = min(per_tensor, p.numel())
            idx = torch.randperm(p.numel(), generator=gen)[:k]
            self.specs.append((name, idx))
            init_vals.append(p.detach().flatten()[idx].float().cpu())
        self.init_vals = torch.cat(init_vals)
        self.num_scalars = self.init_vals.numel()

    @staticmethod
    def _named(module):
        # activation checkpointing 会插 '_checkpoint_wrapped_module.' 前缀, 归一化以对齐 init
        return {n.replace('_checkpoint_wrapped_module.', ''): p for n, p in module.named_parameters()}

    def displacement(self, module):
        params = self._named(module)
        cur = []
        for name, idx in self.specs:
            p = params[name]
            cur.append(p.detach().flatten()[idx.to(p.device)].float().cpu())
        cur = torch.cat(cur)
        return torch.linalg.vector_norm(cur - self.init_vals).item()


# ====================================================================================
#  helpers
# ====================================================================================
def _load_annos(anno_dir):
    annos = {}
    for fp in glob.glob(os.path.join(anno_dir, '*.json')):
        base = os.path.basename(fp)
        if base.startswith('_'):                       # _manifest.json / _packidx2vid.json
            continue
        try:
            a = json.load(open(fp))
        except Exception:
            continue
        vid = str(a.get('vid', os.path.splitext(base)[0]))
        annos[vid] = a
    return annos


def _parse_lambdas(s):
    parts = [float(x) for x in str(s).split(',')]
    assert len(parts) == 2, 'T4G_LAMBDAS 需 2 个值 (id,change)'
    return tuple(parts)


# ====================================================================================
#  Trainer
# ====================================================================================
class T4GCorrTrainer(GigaWorld0Trainer):
    """Track4Gen 式对应监督 trainer (L_diff + λ·四道对应 loss, 时间档门控 + λ warmup)。"""

    # ------------------------------- models -------------------------------
    def get_models(self, model_config):
        model = super().get_models(model_config)       # 设置 self.vae / self.edm_loss / self.train_mode
        assert self.train_mode == 'full', 't4g 对应 loss 需全参训练 (hook 反传, train_mode=full)'
        assert self.mixed_precision != 'fp8', 't4g 未与 fp8/TE 转换联调'

        # 双层 hook / 时间档 / warmup / λ / tol (43 号 §6 起步值; 允许 env / config 覆盖)
        self.id_block = os.environ.get('T4G_ID_BLOCK', model_config.get('t4g_id_block', 'block22'))
        self.change_block = os.environ.get('T4G_CHANGE_BLOCK', model_config.get('t4g_change_block', 'block25'))
        self.sigma_lo = float(os.environ.get('T4G_SIGMA_LO', model_config.get('t4g_sigma_lo', 0.2)))
        self.sigma_hi = float(os.environ.get('T4G_SIGMA_HI', model_config.get('t4g_sigma_hi', 0.5)))
        self.warmup_steps = int(os.environ.get('T4G_WARMUP', model_config.get('t4g_warmup', 20)))
        self.lambda_base = _parse_lambdas(os.environ.get('T4G_LAMBDAS', model_config.get('t4g_lambdas', '0.5,0.4')))
        self.tau = float(os.environ.get('T4G_TAU', model_config.get('t4g_tau', 0.07)))
        self.tol = float(os.environ.get('T4G_TOL', model_config.get('t4g_tol', 0.2)))
        self.win_r_id = int(os.environ.get('T4G_WIN_R_ID', model_config.get('t4g_win_r_id', 3)))
        self.obj_excl_r = int(os.environ.get('T4G_OBJ_EXCL_R', model_config.get('t4g_obj_excl_r', 1)))

        anno_dir = os.environ.get('T4G_ANNO_DIR', model_config.get('t4g_anno_dir', DEFAULT_ANNO_DIR))
        idx2vid_path = os.environ.get('T4G_IDX2VID', model_config.get('t4g_idx2vid', os.path.join(anno_dir, '_packidx2vid.json')))
        self.annos = _load_annos(anno_dir)
        self.idx2vid = {}
        if os.path.exists(idx2vid_path):
            self.idx2vid = {int(k): str(v) for k, v in json.load(open(idx2vid_path)).items()}

        # 存档起点门控 (与 ISOA 同机制; config['train']['checkpoint_start_step'] 经 **kwargs)
        self.checkpoint_start_step = int(os.environ.get('T4G_CHECKPOINT_START_STEP', self.kwargs.get('checkpoint_start_step', 0)))

        # 哨兵探针 (在 init 权重上取快照)
        self.param_probe = ParamDisplacementProbe(model['transformer'])
        self._hook_done = False
        self._id_block_out = None                        # block22 输出 (L_id)
        self._change_block_out = None                    # block25 输出 (L_change)
        self._stat_buffer = []                           # 主进程聚合缓冲
        self._logged_step = -1

        self.print(
            '[t4g] id_block=%s change_block=%s sigma_band=[%.2f,%.2f] warmup=%d '
            'lambda=(id=%.3f,change=%.3f) tau=%.3f tol=%.2f win_r_id=%d obj_excl_r=%d '
            '| annos=%d idx2vid=%d ckpt_start=%d probe_scalars=%d'
            % (self.id_block, self.change_block, self.sigma_lo, self.sigma_hi, self.warmup_steps,
               self.lambda_base[0], self.lambda_base[1], self.tau, self.tol,
               self.win_r_id, self.obj_excl_r, len(self.annos), len(self.idx2vid),
               self.checkpoint_start_step, self.param_probe.num_scalars))
        return model

    # ------------------------------- hook ---------------------------------
    def _ensure_hook(self):
        """惰性注册**双层** forward hook: block22 (L_id) + block25 (L_change)。

        在 accelerate.prepare (含 activation checkpointing 包装) 之后从解包模型取模块引用,
        故 hook 挂在 (若启用 AC) 的 CheckpointWrapper 外层 —— 抓到的是**与自动求导连图**的
        输出 (reentrant checkpoint 下内层 block 前向在 no_grad 里, 抓外层才有梯度)。
        """
        if self._hook_done:
            return
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        tf = model['transformer']

        blk_id = tf.blocks[self.id_block]
        blk_ch = tf.blocks[self.change_block]

        def _hook_id(_m, _inp, out):
            self._id_block_out = out                    # (B,T,H,W,D), 连图, 不 detach

        def _hook_change(_m, _inp, out):
            self._change_block_out = out                # (B,T,H,W,D), 连图, 不 detach

        blk_id.register_forward_hook(_hook_id)
        blk_ch.register_forward_hook(_hook_change)
        self._hook_done = True
        self.print('[t4g] forward hook -> transformer.blocks[%s] (L_id, %s) + blocks[%s] (L_change, %s)'
                   % (self.id_block, type(blk_id).__name__, self.change_block, type(blk_ch).__name__))

    # ------------------------------ warmup --------------------------------
    def _lambda_warmup(self):
        """λ warmup: 20 步内线性 0→(0.5,0.4)。返回 (λ_id, λ_change)。"""
        w = min(1.0, self.cur_step / max(1, self.warmup_steps))
        return tuple(w * b for b in self.lambda_base)

    # ---------------------------- forward ---------------------------------
    def forward_step(self, batch_dict):
        self._ensure_hook()
        transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']
        batch_size = images.shape[0]

        padding_mask = torch.zeros((batch_size, 1, images.shape[-2], images.shape[-1]), dtype=self.dtype, device=self.device)
        fps = batch_dict['fps'][0]

        latents = self.forward_vae(images)
        input_latents, timesteps = self.edm_loss.add_noise(latents)
        sigma = self.edm_loss.sigma.detach().reshape(-1)   # (B,) 本 micro-step 采样噪声档

        ref_images = batch_dict['ref_images']
        ref_masks = batch_dict['ref_masks'].to(self.dtype)
        ref_latents = self.forward_vae(ref_images)

        augment_sigma = torch.tensor([0.0001], device=ref_latents.device, dtype=latents.dtype)
        while len(augment_sigma.shape) < len(ref_latents.shape):
            augment_sigma = augment_sigma.unsqueeze(-1)

        input_latents = ref_masks * ref_latents + (1 - ref_masks) * input_latents
        input_masks = ref_masks.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])
        input_latents = torch.cat([input_latents, input_masks], dim=1)
        timesteps = timesteps.view(1, 1, 1, 1, 1).expand(latents.size(0), -1, latents.size(2), -1, -1)
        t_conditioning = augment_sigma / (augment_sigma + 1)
        timesteps = ref_masks * t_conditioning + (1 - ref_masks) * timesteps

        input_latents = input_latents.to(self.dtype)
        timesteps = timesteps.to(self.dtype)
        prompt_embeds = prompt_embeds.to(self.dtype)

        self._id_block_out = None
        self._change_block_out = None
        model_pred = transformer(
            x=input_latents,
            timesteps=timesteps,
            crossattn_emb=prompt_embeds,
            padding_mask=padding_mask,
            fps=fps,
        )
        denoised_latents = self.edm_loss.denoise(model_pred.float())
        denoised_latents = ref_masks * ref_latents + (1 - ref_masks) * denoised_latents
        l_diff = self.edm_loss.compute_loss(denoised_latents)   # (B,)

        corr = self._compute_corr(batch_dict, sigma)
        losses = dict(l_diff=l_diff)
        losses.update(corr)                              # l_id/l_change (已加 λ 权重)
        return losses

    # -------------------------- corr losses -------------------------------
    def _compute_corr(self, batch_dict, sigma):
        """对 batch 内每个 (sigma 在档 & 有 anno) 样本算两道 loss, 加 λ 权重后按样本取均值。
        L_id 吃 block22 特征, L_change 吃 block25 特征。

        返回 dict(l_id, l_change) = 加权后 0 维 tensor。同时聚合裸值/生效帧数/B 闸状态到
        stats 向量, 跨进程 gather 后缓冲, 供 print_step 打印哨兵。
        """
        device = self.device
        zero = torch.zeros((), device=device)
        weighted = dict(l_id=zero, l_change=zero)
        lam = self._lambda_warmup()
        # stats 向量 (跨进程可加): [applied,
        #   sumLid,cLid,nf_id,  sumLch,cLch,nf_ch,  Bactive,Breleased,  win_skip,  sum_static]
        stats = torch.zeros(11, device=device)

        out_id = self._id_block_out
        out_ch = self._change_block_out
        vids = batch_dict.get('vid', None)
        accum = {k: [] for k in weighted}
        if out_id is not None and out_ch is not None and vids is not None:
            B = out_id.shape[0]
            for b in range(B):
                s = float(sigma[b].item())
                if not (self.sigma_lo <= s <= self.sigma_hi):
                    continue
                vid = str(vids[b])
                anno = self.annos.get(vid, None)
                if anno is None:
                    continue
                feat_id = out_id[b].float()             # (T,H,W,D) block22, 连图
                feat_ch = out_ch[b].float()             # (T,H,W,D) block25, 连图
                res = compute_corr_losses(feat_id, feat_ch, anno, tau=self.tau, tol=self.tol,
                                          win_r=self.win_r_id, obj_r=self.obj_excl_r)
                L, C, G, Dg = res['losses'], res['counts'], res['gate'], res['diag']
                accum['l_id'].append(lam[0] * L['L_id'])
                accum['l_change'].append(lam[1] * L['L_change'])
                stats[0] += 1
                if C['n_id'] > 0:
                    stats[1] += float(L['L_id'].item()); stats[2] += 1; stats[3] += C['n_id']
                if C['n_change'] > 0:
                    stats[4] += float(L['L_change'].item()); stats[5] += 1; stats[6] += C['n_change']
                    stats[10] += float(Dg['static_cells_mean'])
                stats[7] += G['b_active_frames']
                stats[8] += G['b_released_frames']
                stats[9] += Dg['id_win_skipped']

        for k in weighted:
            if accum[k]:
                weighted[k] = torch.stack(accum[k]).mean()

        # 跨进程聚合 (collective: 每个 micro-step 各进程对称调用一次)
        gathered = self.accelerator.gather(stats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._stat_buffer.append(gathered)
        return weighted

    # --------------------------- sentinels --------------------------------
    def print_step(self):
        super().print_step()
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        if self.cur_step == self._logged_step:
            return
        self._logged_step = self.cur_step

        if self._stat_buffer:
            agg = torch.stack(self._stat_buffer).sum(0)
            self._stat_buffer = []
        else:
            agg = torch.zeros(11)

        # stats 布局: [0]applied [1]sumLid [2]cLid [3]nf_id [4]sumLch [5]cLch [6]nf_ch
        #             [7]Bactive [8]Breleased [9]win_skip [10]sum_static
        def _ratio(num, den):
            d = agg[den].item()
            return (agg[num].item() / d) if d > 0 else float('nan')

        applied = int(agg[0])
        L_id = _ratio(1, 2); f_id = int(agg[3])
        L_change = _ratio(4, 5); f_change = int(agg[6])
        b_active = int(agg[7]); b_released = int(agg[8])
        win_skip = int(agg[9])
        static_self = _ratio(10, 5)   # 静止格平均自相似(应~高; 掉=有变化)

        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        disp = self.param_probe.displacement(model['transformer'])
        lam = self._lambda_warmup()

        msg = ('[t4g-sentinel] step=%d applied=%d | L_id=%.4e(nf=%d) L_change=%.4e(nf=%d) '
               '| gate: B_active=%d B_released=%d id_win_skip=%d static_self=%.4f '
               '| param_disp_l2=%.4e | lambda=(%.3f,%.3f) band=[%.2f,%.2f]'
               % (self.cur_step, applied, L_id, f_id, L_change, f_change,
                  b_active, b_released, win_skip, static_self, disp,
                  lam[0], lam[1], self.sigma_lo, self.sigma_hi))
        self.logger.info(msg)
        print(msg, flush=True)

    # --------------------------- checkpoint -------------------------------
    def save_checkpoint_step(self):
        """存档起点门控: global step < checkpoint_start_step 跳过, 其余沿用基类。"""
        if self.cur_step < self.checkpoint_start_step:
            return
        super().save_checkpoint_step()
