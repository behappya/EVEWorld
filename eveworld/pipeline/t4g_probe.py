#!/usr/bin/env python3
"""EVE · Track4Gen 特征可追踪性探针 (t4g_probe).

目的 (生死 gate)
----------------
在给 GigaWorld-0 DiT 加"跨帧对应 loss"(Track4Gen, arXiv 2412.06016) 之前, 先验证
一个前提: **某一层的内部特征本就能追踪物体**。若测不到好层则止损 (NO-GO)。

本脚本 = 纯探针 (导特征 + 算指标), 不做任何训练。

方法
----
1. 对一条真实视频前向 (复用 giga_world_0_trainer.forward_step 的构建机制: VAE 编码 ->
   EDM add_noise -> 首帧 ref 条件 -> transformer(x, timesteps, crossattn_emb, fps)),
   但用独立轻量前向 (非完整 Trainer)。
2. 用 register_forward_hook 抓 28 个 block 的输出, 每个 block 输出 shape = (B, T, H, W, D)
   (规整特征网格, 无需 reshape; T/H/W 从 hook 到的 shape 读, 不硬编)。
3. 用真实视频的光流做"标准答案":
   - DISOpticalFlow 相邻帧稠密光流 -> 聚合到 (H, W) 特征网格分辨率;
   - 累计位移分动/静组: 动组 = totalmag > p85, 静组 = totalmag < p40。
4. 每层每 sigma 出两个指标:
   - **动组可追踪性** = 特征 argmax(soft-argmax 亚格) 追踪轨迹 vs 光流 GT 轨迹的端点误差
     中位数 (单位 = 特征网格 cell);
   - **静组稳定性** = 静组 cell 特征跨帧余弦自相似度均值。
5. 扫 3-4 个 sigma (低/中/高), 每层每 sigma 都出指标 (Track4Gen 指出特征质量随噪声级变)。
6. 汇总所有视频 -> 每层 x 每 sigma 的表, 标出最优层。

判据 (写进输出)
----------------
存在某 (层, sigma): 端点误差 < ~2 cell 且 静组自相似 > 0.9  ->  GO (那层做对应 loss);
全不达标  ->  NO-GO 止损。

用法 (kjob payload 内部调用, 单卡):
  python t4g_probe.py \
    --transformer /data/.../probe_anmix_s200/transformer \
    --vae         /data/.../probe_anmix_s200/vae \
    --text-encoder /data/.../probe_anmix_s200/text_encoder \
    --video-root  /data/.../gr1_finetune_data/raw_data \
    --video-ids   13,32,76,... \
    --sigmas      0.25,0.7,2.0,5.0 \
    --num-frames 93 --height 480 --width 768 --fps 16 \
    --out-dir /data/.../track4gen_probe/anmix_s200

多卡: 用 --num-shards / --shard-index 把视频切片分卡跑, 每卡产出一个 partial JSON,
      再用 --merge 汇总 (见 t4g_probe_kjob.sh 的 dispatcher)。
"""
import argparse
import json
import os
import sys
import time
from glob import glob

import cv2
import numpy as np

# ------------------------------------------------------------------------------------
# 默认判据阈值 (可命令行覆盖)
# ------------------------------------------------------------------------------------
DEFAULT_EPE_GO = 2.0      # 端点误差 < 2 个特征网格 cell 视为"可追踪"
DEFAULT_STAB_GO = 0.90    # 静组跨帧余弦自相似 > 0.9 视为"稳定"
DEFAULT_P_HI = 85.0       # 动组: 累计位移 > p85
DEFAULT_P_LO = 40.0       # 静组: 累计位移 < p40


# ====================================================================================
#  分析核 (纯 numpy / torch, 不依赖模型) —— CPU 冒烟脚本直接 import 复用
# ====================================================================================
def sample_frames_like_training(path, num_frames, height, width):
    """按训练 transform 的方式抽帧: 均匀 linspace 抽 num_frames 帧 -> 保长宽比 resize ->
    中心裁剪到 (height, width)。返回 uint8 (num_frames, height, width, 3) RGB。

    与 GigaWorld0Transform 的差异: 用中心裁剪 (而非随机裁剪) 保证探针可复现。
    """
    try:
        import decord
        vr = decord.VideoReader(path)
        n = len(vr)
        idx = np.linspace(0, n - 1, num_frames, dtype=int)
        frames = vr.get_batch(list(idx)).asnumpy()  # (T, H, W, 3) RGB
    except Exception:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = set(np.linspace(0, n - 1, num_frames, dtype=int).tolist())
        buf = {}
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in idx:
                buf[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        order = sorted(buf)
        frames = np.stack([buf[k] for k in order], 0)
        # linspace 可能有重复 idx (short video); 补齐
        while frames.shape[0] < num_frames:
            frames = np.concatenate([frames, frames[-1:]], 0)

    out = []
    ih, iw = frames.shape[1], frames.shape[2]
    # 保长宽比 resize (与 transform 同逻辑)
    if float(height) / ih < float(width) / iw:
        new_h = int(round(float(width) / iw * ih))
        new_w = width
    else:
        new_h = height
        new_w = int(round(float(height) / ih * iw))
    y1 = max(0, (new_h - height) // 2)
    x1 = max(0, (new_w - width) // 2)
    for fr in frames:
        r = cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        r = r[y1:y1 + height, x1:x1 + width]
        out.append(r)
    return np.stack(out, 0).astype(np.uint8)


def dense_flow_sequence(frames):
    """相邻帧 DISOpticalFlow 稠密光流。frames: uint8 (T, H, W, 3) RGB。
    返回 list[ (H, W, 2) float32 ], 长度 T-1; flow[..., 0]=dx, flow[..., 1]=dy (像素)。
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flows = []
    for a, b in zip(grays[:-1], grays[1:]):
        flows.append(dis.calc(a, b, None).astype(np.float32))
    return flows


def _avgpool_grid(field, gh, gw):
    """把 (H, W, C) 或 (H, W) 平均池化到 (gh, gw, ...)。"""
    H, W = field.shape[:2]
    hs = np.linspace(0, H, gh + 1).astype(int)
    ws = np.linspace(0, W, gw + 1).astype(int)
    if field.ndim == 2:
        out = np.zeros((gh, gw), np.float32)
    else:
        out = np.zeros((gh, gw, field.shape[2]), np.float32)
    for i in range(gh):
        for j in range(gw):
            out[i, j] = field[hs[i]:hs[i + 1], ws[j]:ws[j + 1]].mean(axis=(0, 1))
    return out


def motion_grid(flows, gh, gw):
    """累计每像素光流幅值 -> 平均池化到 (gh, gw)。返回 totalmag_grid (gh, gw)。"""
    if len(flows) == 0:
        return np.zeros((gh, gw), np.float32)
    H, W = flows[0].shape[:2]
    total = np.zeros((H, W), np.float32)
    for f in flows:
        total += np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
    return _avgpool_grid(total, gh, gw)


def classify_dyn_static(totalmag_grid, p_hi=DEFAULT_P_HI, p_lo=DEFAULT_P_LO):
    """动组 = totalmag > p_hi 百分位; 静组 = totalmag < p_lo 百分位。
    返回 (dyn_cells, static_cells) 各为 list[(gy, gx)]。"""
    thi = np.percentile(totalmag_grid, p_hi)
    tlo = np.percentile(totalmag_grid, p_lo)
    dyn = list(map(tuple, np.argwhere(totalmag_grid > thi)))
    static = list(map(tuple, np.argwhere(totalmag_grid < tlo)))
    return dyn, static


def pixel_to_latent_map(n_pix, n_lat):
    """latent 帧 t -> 代表像素帧下标。VAE 时间 4x 压缩 + patch_temporal 后, n_lat 个
    latent 帧均匀对应 n_pix 个像素帧。t=0 -> 0, t=n_lat-1 -> n_pix-1。"""
    if n_lat <= 1:
        return np.zeros(max(n_lat, 1), dtype=int)
    return np.round(np.arange(n_lat) * (n_pix - 1) / (n_lat - 1)).astype(int)


def _bilinear_sample(field, pts):
    """双线性采样 field(H, W, 2) 在点 pts(N, 2)=(x, y)。返回 (N, 2)。"""
    H, W = field.shape[:2]
    x = np.clip(pts[:, 0], 0, W - 1.001)
    y = np.clip(pts[:, 1], 0, H - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    wx = x - x0; wy = y - y0
    f00 = field[y0, x0]; f01 = field[y0, x1]
    f10 = field[y1, x0]; f11 = field[y1, x1]
    top = f00 * (1 - wx)[:, None] + f01 * wx[:, None]
    bot = f10 * (1 - wx)[:, None] + f11 * wx[:, None]
    return top * (1 - wy)[:, None] + bot * wy[:, None]


def gt_trajectories(flows, cells, pmap, cell_h, cell_w):
    """光流 GT 轨迹: 从每个 query cell 中心出发, 沿稠密光流逐像素-帧累计, 记录每个 latent
    帧的位置。
    flows: list[(H, W, 2)] 长 n_pix-1; cells: list[(gy, gx)]; pmap: latent->pixel 映射;
    cell_h/cell_w: 每个特征 cell 的像素高/宽。
    返回 gt_grid (T_lat, N, 2) = (y, x) 特征网格坐标。
    """
    n_lat = len(pmap)
    N = len(cells)
    # 起点 = cell 中心像素坐标 (x, y)
    pos = np.array([[(gx + 0.5) * cell_w, (gy + 0.5) * cell_h] for (gy, gx) in cells], np.float32)
    gt = np.zeros((n_lat, N, 2), np.float32)
    gt[0] = np.stack([pos[:, 1] / cell_h, pos[:, 0] / cell_w], 1)  # (y, x) in grid units
    for t in range(1, n_lat):
        for p in range(int(pmap[t - 1]), int(pmap[t])):
            if p < len(flows):
                pos = pos + _bilinear_sample(flows[p], pos)
        gt[t] = np.stack([pos[:, 1] / cell_h, pos[:, 0] / cell_w], 1)
    return gt


def soft_argmax_track(feat_thwd, cells, device='cpu', tau=0.07, win=2):
    """特征追踪 (预测轨迹)。对 query cell 取首帧特征向量, 对每个后续帧全网格算余弦相似度,
    在 argmax 邻域做 soft-argmax 取亚格精度。
    feat_thwd: torch.Tensor (T, H, W, D); cells: list[(gy, gx)]。
    返回 pred (T, N, 2) = (y, x) 特征网格坐标 (numpy)。
    """
    import torch
    feat = feat_thwd.to(device).float()
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)   # 单位化 (T, H, W, D)
    idx = torch.tensor([gy * W + gx for (gy, gx) in cells], device=device, dtype=torch.long)
    N = idx.numel()
    q = fn[0].reshape(H * W, D)[idx]                       # (N, D) 首帧 query 向量
    yy = torch.arange(H, device=device).view(H, 1).expand(H, W).reshape(-1).float()
    xx = torch.arange(W, device=device).view(1, W).expand(H, W).reshape(-1).float()
    pred = np.zeros((T, N, 2), np.float32)
    # 首帧: query 就在自身位置
    pred[0, :, 0] = np.array([gy for (gy, gx) in cells], np.float32)
    pred[0, :, 1] = np.array([gx for (gy, gx) in cells], np.float32)
    for t in range(1, T):
        sim = q @ fn[t].reshape(H * W, D).t()             # (N, H*W) 余弦相似度
        amax = sim.argmax(dim=1)                          # (N,)
        ay = (amax // W).float(); ax = (amax % W).float()
        # 只在 argmax 的 (2*win+1) 邻域内做 soft-argmax, 排除远处干扰
        dy = (yy.view(1, -1) - ay.view(-1, 1)).abs()
        dx = (xx.view(1, -1) - ax.view(-1, 1)).abs()
        mask = (dy <= win) & (dx <= win)
        masked = sim.masked_fill(~mask, float('-inf'))
        w = torch.softmax(masked / tau, dim=1)            # (N, H*W)
        ey = (w * yy.view(1, -1)).sum(1)
        ex = (w * xx.view(1, -1)).sum(1)
        pred[t, :, 0] = ey.cpu().numpy()
        pred[t, :, 1] = ex.cpu().numpy()
    return pred


def chained_track(feat_thwd, cells, device='cpu', tau=0.07, win=3):
    """逐帧接力追踪 (修正版, 排除首帧污染): 每帧 query = 上一帧预测位置的特征(双线性采样),
    在当前帧 argmax 邻域做 soft-argmax。返回 pred (T, N, 2)。"""
    import torch
    feat = feat_thwd.to(device).float()
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)     # (T,H,W,D)
    yy = torch.arange(H, device=device).view(H, 1).expand(H, W).reshape(-1).float()
    xx = torch.arange(W, device=device).view(1, W).expand(H, W).reshape(-1).float()
    N = len(cells)
    pred = np.zeros((T, N, 2), np.float32)
    cur = torch.tensor([[gy, gx] for (gy, gx) in cells], device=device, dtype=torch.float32)  # (N,2)
    pred[0, :, 0] = cur[:, 0].cpu().numpy(); pred[0, :, 1] = cur[:, 1].cpu().numpy()

    def sample_feat(frame_fn, pos):
        # 双线性采样 frame_fn(H,W,D) 在 pos(N,2)=(y,x)
        y = pos[:, 0].clamp(0, H - 1); x = pos[:, 1].clamp(0, W - 1)
        y0 = y.floor().long(); x0 = x.floor().long()
        y1 = (y0 + 1).clamp(max=H - 1); x1 = (x0 + 1).clamp(max=W - 1)
        wy = (y - y0.float()).unsqueeze(1); wx = (x - x0.float()).unsqueeze(1)
        f = (frame_fn[y0, x0] * (1 - wy) * (1 - wx) + frame_fn[y1, x0] * wy * (1 - wx) +
             frame_fn[y0, x1] * (1 - wy) * wx + frame_fn[y1, x1] * wy * wx)
        return f / (f.norm(dim=-1, keepdim=True) + 1e-8)

    for t in range(1, T):
        q = sample_feat(fn[t - 1], cur)                      # (N,D) 上一帧位置的特征
        sim = q @ fn[t].reshape(H * W, D).t()                # (N,H*W)
        amax = sim.argmax(dim=1)
        ay = (amax // W).float(); ax = (amax % W).float()
        dy = (yy.view(1, -1) - ay.view(-1, 1)).abs()
        dx = (xx.view(1, -1) - ax.view(-1, 1)).abs()
        mask = (dy <= win) & (dx <= win)
        w = torch.softmax(sim.masked_fill(~mask, float('-inf')) / tau, dim=1)
        ey = (w * yy.view(1, -1)).sum(1); ex = (w * xx.view(1, -1)).sum(1)
        cur = torch.stack([ey, ex], dim=1)
        pred[t, :, 0] = ey.cpu().numpy(); pred[t, :, 1] = ex.cpu().numpy()
    return pred


def static_stability(feat_thwd, cells, device='cpu'):
    """静组稳定性: 每个静组 cell 的特征跨帧余弦自相似度均值 (相对首帧)。
    返回 (mean_stab, per_cell array)。"""
    import torch
    if len(cells) == 0:
        return float('nan'), np.array([])
    feat = feat_thwd.to(device).float()
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    idx = torch.tensor([gy * W + gx for (gy, gx) in cells], device=device, dtype=torch.long)
    seq = fn.reshape(T, H * W, D)[:, idx, :]              # (T, N, D)
    ref = seq[0:1]                                        # (1, N, D)
    cos = (seq * ref).sum(-1)                             # (T, N)
    per_cell = cos[1:].mean(0).cpu().numpy() if T > 1 else cos[0].cpu().numpy()
    return float(np.mean(per_cell)), per_cell


def compute_layer_metrics(feat_thwd, flows, dyn_cells, static_cells, n_pix,
                          device='cpu', tau=0.07):
    """对单层特征 (T, H, W, D) 算两个指标。返回 dict。"""
    import torch
    T, H, W, D = feat_thwd.shape
    cell_h = flows[0].shape[0] / H if flows else 1.0
    cell_w = flows[0].shape[1] / W if flows else 1.0
    pmap = pixel_to_latent_map(n_pix, T)

    # 动组可追踪性 (端点误差, 单位 = 特征网格 cell)
    # 修正版(TRACK_CHAIN=1): 逐帧接力追踪(排除首帧污染)
    import os
    use_chain = os.environ.get('TRACK_CHAIN', '0') == '1'
    tracker = chained_track if use_chain else soft_argmax_track
    med_epe = float('nan')
    if len(dyn_cells) > 0 and len(flows) > 0 and T > 1:
        pred = tracker(feat_thwd, dyn_cells, device=device, tau=tau)             # (T, N, 2)
        gt = gt_trajectories(flows, dyn_cells, pmap, cell_h, cell_w)             # (T, N, 2)
        epe = np.sqrt(((pred[1:] - gt[1:]) ** 2).sum(-1))                        # (T-1, N)
        med_epe = float(np.median(epe))

    # 静组稳定性
    stab, _ = static_stability(feat_thwd, static_cells, device=device)
    return dict(median_epe=med_epe, static_stability=stab,
                n_dyn=len(dyn_cells), n_static=len(static_cells), T=T, H=H, W=W)


# ====================================================================================
#  模型前向 + hook (torch, GPU)
# ====================================================================================
def load_models(transformer_path, vae_path, text_encoder_path, device, dtype):
    """独立轻量加载 transformer + VAE + T5 (参考 generate_eag.py / trainer.get_models)。"""
    import torch
    from diffusers.models import AutoencoderKLWan
    from giga_models import GigaWorld0Transformer3DModel
    from giga_models.models.diffusion.giga_world_0.t5_text_encoder import T5TextEncoder

    print(f'[t4g] loading transformer <- {transformer_path}', flush=True)
    transformer = GigaWorld0Transformer3DModel.from_pretrained(transformer_path)
    transformer.to(device, dtype=dtype).eval()
    transformer.requires_grad_(False)

    print(f'[t4g] loading vae <- {vae_path}', flush=True)
    vae = AutoencoderKLWan.from_pretrained(vae_path)
    vae.to(device, dtype=dtype).eval()
    vae.requires_grad_(False)
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    lat_std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    print(f'[t4g] loading text_encoder <- {text_encoder_path}', flush=True)
    text_encoder = T5TextEncoder(text_encoder_path)
    text_encoder.text_encoder.to(device)

    return dict(transformer=transformer, vae=vae, text_encoder=text_encoder,
                lat_mean=lat_mean, lat_std=lat_std)


def _forward_vae(vae, lat_mean, lat_std, images):
    """images: (B, T, C, H, W) [-1,1] -> 归一化 latents (B, z, T_lat, H_lat, W_lat)。"""
    import torch
    from einops import rearrange
    images = images.to(vae.dtype)
    with torch.no_grad():
        images = rearrange(images, 'b t c h w -> b c t h w')
        latents = vae.encode(images).latent_dist.sample()
    return (latents - lat_mean) * lat_std


def forward_with_hooks(models, frames_norm, prompt_embeds, sigma, fps, device, dtype,
                       edm):
    """对一条视频做一次前向, 抓 28 层 block 输出。
    frames_norm: torch (1, T_pix, 3, H, W) 归一化到 [-1,1];
    返回 dict{ block_name: cpu float32 (T, H, W, D) }。
    复刻 giga_world_0_trainer.forward_step 的输入构建 (首帧 ref 条件 + 固定 sigma)。
    """
    import torch

    transformer = models['transformer']
    vae = models['vae']
    lat_mean, lat_std = models['lat_mean'], models['lat_std']

    B = frames_norm.shape[0]
    Hpix, Wpix = frames_norm.shape[-2], frames_norm.shape[-1]
    padding_mask = torch.zeros((B, 1, Hpix, Wpix), dtype=dtype, device=device)

    latents = _forward_vae(vae, lat_mean, lat_std, frames_norm)          # (B, z, Tlat, h, w)
    Tlat = latents.shape[2]

    # 固定 sigma 加噪 (扫描用): add_noise 接受显式 sigma
    sigma_t = torch.tensor([sigma], device=device, dtype=torch.float32).view(B, 1)
    input_latents, c_noise = edm.add_noise(latents.float(), sigma=sigma_t)
    input_latents = input_latents.to(dtype)

    # 首帧 ref 条件: 只有第 0 个 latent 帧作参考 (max_ref_frames=1 纯首帧 I2V)
    ref_mask = torch.zeros((B, 1, Tlat, 1, 1), dtype=dtype, device=device)
    ref_mask[:, :, 0] = 1.0
    ref_frames = frames_norm.clone()
    ref_frames[:, 1:] = 0.0                                              # 只保留首像素帧
    ref_latents = _forward_vae(vae, lat_mean, lat_std, ref_frames).to(dtype)

    augment_sigma = torch.tensor([1e-4], device=device, dtype=input_latents.dtype).view(1, 1, 1, 1, 1)
    input_latents = ref_mask * ref_latents + (1 - ref_mask) * input_latents
    input_masks = ref_mask.repeat(1, 1, 1, input_latents.shape[-2], input_latents.shape[-1])
    input_latents = torch.cat([input_latents, input_masks], dim=1)      # (B, z+1, Tlat, h, w)

    timesteps = c_noise.to(dtype).view(1, 1, 1, 1, 1).expand(B, -1, Tlat, -1, -1)
    t_cond = augment_sigma / (augment_sigma + 1)
    timesteps = ref_mask * t_cond + (1 - ref_mask) * timesteps
    timesteps = timesteps.to(dtype)

    prompt_embeds = prompt_embeds.to(dtype)

    # ---- 注册 hook: 抓每个 block 输出 (B, T, H, W, D) -> CPU float32 ----
    store = {}
    handles = []

    def mk_hook(name):
        def hook(_m, _inp, out):
            store[name] = out.detach().to('cpu', torch.float32)[0]  # 去 batch -> (T, H, W, D)
        return hook

    for name, block in transformer.blocks.items():
        handles.append(block.register_forward_hook(mk_hook(name)))

    try:
        with torch.no_grad():
            _ = transformer(x=input_latents, timesteps=timesteps,
                            crossattn_emb=prompt_embeds, padding_mask=padding_mask, fps=fps)
    finally:
        for h in handles:
            h.remove()
    return store


# ====================================================================================
#  主流程: 视频 x sigma 扫描 + 汇总表 + GO/NO-GO
# ====================================================================================
def read_prompt(video_root, vid):
    txt = os.path.join(video_root, f'{vid}.txt')
    if os.path.exists(txt):
        return open(txt).read().strip()
    return 'a robot manipulation video.'


def run_probe(args):
    import torch

    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
    device = args.device
    from giga_models.nn import EDMLoss
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)

    models = load_models(args.transformer, args.vae, args.text_encoder, device, dtype)

    sigmas = [float(s) for s in args.sigmas.split(',')]
    vids = [v.strip() for v in args.video_ids.split(',') if v.strip()]
    # 视频分卡切片
    if args.num_shards > 1:
        vids = vids[args.shard_index::args.num_shards]
    print(f'[t4g] shard {args.shard_index}/{args.num_shards}: {len(vids)} videos, sigmas={sigmas}', flush=True)

    # 累加器: records[(layer, sigma)] = { 'epe': [...per video-cell medians], 'stab': [...] }
    records = {}
    per_video = []

    for vi, vid in enumerate(vids):
        vpath = os.path.join(args.video_root, f'{vid}.mp4')
        if not os.path.exists(vpath):
            print(f'[t4g] skip missing {vpath}', flush=True)
            continue
        t0 = time.time()
        frames = sample_frames_like_training(vpath, args.num_frames, args.height, args.width)
        flows = dense_flow_sequence(frames)

        # 归一化到 [-1,1] tensor (1, T, 3, H, W)
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        fr = (fr - 0.5) / 0.5
        frames_norm = fr.unsqueeze(0).to(device)

        prompt = read_prompt(args.video_root, vid)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(prompt, max_length=512)
        emb = emb.to(device)

        vid_layer_best = {}
        for sigma in sigmas:
            store = forward_with_hooks(models, frames_norm, emb, sigma, args.fps, device, dtype, edm)
            # 用第一层的 (H, W) 定动/静组 (所有层同网格)
            any_feat = next(iter(store.values()))
            T, H, W, D = any_feat.shape
            tm = motion_grid(flows, H, W)
            dyn, static = classify_dyn_static(tm, args.p_hi, args.p_lo)

            for name, feat in store.items():
                m = compute_layer_metrics(feat, flows, dyn, static, args.num_frames,
                                          device=device, tau=args.tau)
                key = (name, sigma)
                rec = records.setdefault(key, {'epe': [], 'stab': []})
                if not np.isnan(m['median_epe']):
                    rec['epe'].append(m['median_epe'])
                if not np.isnan(m['static_stability']):
                    rec['stab'].append(m['static_stability'])
                vid_layer_best[(name, sigma)] = m
            del store
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

        per_video.append(dict(vid=vid, T=T, H=H, W=W, n_dyn=len(dyn), n_static=len(static),
                              sec=round(time.time() - t0, 1)))
        print(f'[t4g] {vi+1}/{len(vids)} vid={vid} T={T} HxW={H}x{W} '
              f'dyn={len(dyn)} static={len(static)} {time.time()-t0:.1f}s', flush=True)

    # ---- 汇总 ----
    layer_names = list(models['transformer'].blocks.keys())
    summary = aggregate(records, layer_names, sigmas, args)
    summary['per_video'] = per_video
    summary['config'] = dict(transformer=args.transformer, sigmas=sigmas,
                             num_frames=args.num_frames, height=args.height, width=args.width,
                             p_hi=args.p_hi, p_lo=args.p_lo, epe_go=args.epe_go, stab_go=args.stab_go,
                             shard=f'{args.shard_index}/{args.num_shards}')

    os.makedirs(args.out_dir, exist_ok=True)
    if args.num_shards > 1:
        out = os.path.join(args.out_dir, f'partial_shard{args.shard_index}.json')
        json.dump(dict(records={f'{k[0]}|{k[1]}': v for k, v in records.items()},
                       per_video=per_video, sigmas=sigmas, layer_names=layer_names,
                       config=summary['config']), open(out, 'w'), indent=2)
        print(f'[t4g] partial -> {out}', flush=True)
    else:
        out = os.path.join(args.out_dir, 't4g_probe_summary.json')
        json.dump(summary, open(out, 'w'), indent=2)
        print_report(summary, args)
        print(f'[t4g] summary -> {out}', flush=True)
    return summary


def aggregate(records, layer_names, sigmas, args):
    """records[(layer, sigma)] -> 每层每 sigma 的中位 EPE / 均值 stab + GO/NO-GO。"""
    table = {}
    go_cells = []
    for name in layer_names:
        for sigma in sigmas:
            rec = records.get((name, sigma), {'epe': [], 'stab': []})
            med_epe = float(np.median(rec['epe'])) if rec['epe'] else float('nan')
            mean_stab = float(np.mean(rec['stab'])) if rec['stab'] else float('nan')
            is_go = (not np.isnan(med_epe) and not np.isnan(mean_stab)
                     and med_epe < args.epe_go and mean_stab > args.stab_go)
            table[f'{name}|{sigma}'] = dict(median_epe=med_epe, mean_static_stability=mean_stab,
                                            n_videos=len(rec['epe']), go=bool(is_go))
            if is_go:
                go_cells.append(dict(layer=name, sigma=sigma, median_epe=med_epe,
                                     mean_static_stability=mean_stab))
    # 最优层: 在达标 cell 中选 EPE 最小; 若无达标, 报告全局 EPE 最小且 stab 最高的层作参考
    best = None
    if go_cells:
        best = min(go_cells, key=lambda c: c['median_epe'])
        verdict = 'GO'
    else:
        cand = [(k, v) for k, v in table.items() if not np.isnan(v['median_epe'])]
        if cand:
            k, v = min(cand, key=lambda kv: kv[1]['median_epe'])
            ln, sg = k.split('|')
            best = dict(layer=ln, sigma=float(sg), median_epe=v['median_epe'],
                        mean_static_stability=v['mean_static_stability'])
        verdict = 'NO-GO'
    return dict(table=table, go_cells=go_cells, best=best, verdict=verdict)


def run_dispatch(args):
    """单节点多卡编排 (纯 Python; kjobctl 的 slurm 解释器不支持 bash 后台作业)。
    对每张 GPU 起一个 worker 子进程, 用 CUDA_VISIBLE_DEVICES 钉卡, 各跑一个视频分片,
    产出 partial_shard{i}.json; 全部完成后合并出最终表。"""
    import subprocess
    n = args.dispatch_gpus
    os.makedirs(args.out_dir, exist_ok=True)
    procs = []
    base = [sys.executable, os.path.abspath(__file__),
            '--transformer', args.transformer, '--vae', args.vae,
            '--text-encoder', args.text_encoder,
            '--video-root', args.video_root, '--video-ids', args.video_ids,
            '--sigmas', args.sigmas, '--num-frames', str(args.num_frames),
            '--height', str(args.height), '--width', str(args.width),
            '--fps', str(args.fps), '--tau', str(args.tau),
            '--p-hi', str(args.p_hi), '--p-lo', str(args.p_lo),
            '--epe-go', str(args.epe_go), '--stab-go', str(args.stab_go),
            '--dtype', args.dtype, '--device', 'cuda', '--out-dir', args.out_dir]
    for i in range(n):
        cmd = base + ['--num-shards', str(n), '--shard-index', str(i)]
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        env['PYTHONUNBUFFERED'] = '1'
        lf = open(os.path.join(args.out_dir, f'worker_gpu{i}.log'), 'w')
        print(f'[t4g-dispatch] GPU {i} <- shard {i}/{n} (log worker_gpu{i}.log)', flush=True)
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
    fail = 0
    for i, p, lf in procs:
        rc = p.wait()
        lf.close()
        status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
        print(f'[t4g-dispatch] GPU {i} shard done: {status}', flush=True)
        if rc != 0:
            fail = 1
    if fail:
        print('[t4g-dispatch] 有 worker 失败, 仍尝试合并已产出的 partial。', file=sys.stderr, flush=True)
    merge_partials(args)
    sys.exit(fail)


def merge_partials(args):
    """合并多卡 partial JSON -> 最终汇总表 + GO/NO-GO。"""
    parts = sorted(glob(os.path.join(args.out_dir, 'partial_shard*.json')))
    if not parts:
        print(f'[t4g] no partial_shard*.json under {args.out_dir}', file=sys.stderr)
        sys.exit(1)
    records = {}
    per_video = []
    sigmas = None
    layer_names = None
    config = None
    for p in parts:
        d = json.load(open(p))
        sigmas = d['sigmas']; layer_names = d['layer_names']; config = d['config']
        per_video += d['per_video']
        for k, v in d['records'].items():
            name, sg = k.rsplit('|', 1)
            key = (name, float(sg))
            rec = records.setdefault(key, {'epe': [], 'stab': []})
            rec['epe'] += v['epe']; rec['stab'] += v['stab']
    summary = aggregate(records, layer_names, sigmas, args)
    summary['per_video'] = per_video
    summary['config'] = config
    out = os.path.join(args.out_dir, 't4g_probe_summary.json')
    json.dump(summary, open(out, 'w'), indent=2)
    print_report(summary, args)
    print(f'[t4g] merged {len(parts)} shards -> {out}', flush=True)


def print_report(summary, args):
    """打印每层 x 每 sigma 的表 + 判据结论。"""
    table = summary['table']
    sigmas = summary['config']['sigmas']
    layer_names = sorted({k.split('|')[0] for k in table},
                         key=lambda n: int(n.replace('block', '')))
    print('\n' + '=' * 96)
    print(' Track4Gen 特征可追踪性探针 —— 每层 x 每 sigma')
    print(f'   指标: EPE=动组追踪端点误差中位(cell, 越小越好) | STAB=静组跨帧余弦自相似(越大越好)')
    print(f'   判据: EPE < {args.epe_go} 且 STAB > {args.stab_go}  ->  该 cell 达标 (标 *)')
    print('=' * 96)
    header = 'layer   ' + ''.join([f'| sig={s:<6}      ' for s in sigmas])
    print(header)
    print('-' * len(header))
    for name in layer_names:
        row = f'{name:<7} '
        for s in sigmas:
            v = table.get(f'{name}|{s}', {})
            epe = v.get('median_epe', float('nan'))
            stab = v.get('mean_static_stability', float('nan'))
            mark = '*' if v.get('go') else ' '
            row += f'| {epe:5.2f}/{stab:4.2f}{mark}   '
        print(row)
    print('=' * 96)
    print(f' 判据结论: {summary["verdict"]}')
    if summary['best'] is not None:
        b = summary['best']
        tag = '最优达标层' if summary['verdict'] == 'GO' else '(无达标层) 最接近层, 供参考'
        print(f' {tag}: layer={b["layer"]} sigma={b["sigma"]} '
              f'EPE={b["median_epe"]:.2f} STAB={b["mean_static_stability"]:.3f}')
    if summary['verdict'] == 'GO':
        print(' => 存在可追踪层, 可在该层加 Track4Gen 对应 loss。')
    else:
        print(' => 无层同时满足端点误差与静组稳定性阈值, 建议对应 loss 方案止损 (NO-GO)。')
    print('=' * 96 + '\n')


def build_argparser():
    ap = argparse.ArgumentParser(description='Track4Gen 特征可追踪性探针 (纯探针, 不训练)')
    ap.add_argument('--transformer', required=False)
    ap.add_argument('--vae', required=False)
    ap.add_argument('--text-encoder', required=False, dest='text_encoder')
    ap.add_argument('--video-root', dest='video_root',
                    default='/data/datasets/gagi/gr1_finetune_data/raw_data')
    ap.add_argument('--video-ids', dest='video_ids',
                    default='13,32,76,14,15,16,17,18,19,20,21,23,24,25,26,27')
    ap.add_argument('--sigmas', default='0.25,0.7,2.0,5.0', help='低/中/高噪声级, 逗号分隔')
    ap.add_argument('--num-frames', dest='num_frames', type=int, default=93)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--tau', type=float, default=0.07, help='soft-argmax 温度')
    ap.add_argument('--p-hi', dest='p_hi', type=float, default=DEFAULT_P_HI)
    ap.add_argument('--p-lo', dest='p_lo', type=float, default=DEFAULT_P_LO)
    ap.add_argument('--epe-go', dest='epe_go', type=float, default=DEFAULT_EPE_GO)
    ap.add_argument('--stab-go', dest='stab_go', type=float, default=DEFAULT_STAB_GO)
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out-dir', dest='out_dir', required=False,
                    default='/data/datasets/gagi/eve_v2_outputs/track4gen_probe/run')
    ap.add_argument('--num-shards', dest='num_shards', type=int, default=1)
    ap.add_argument('--shard-index', dest='shard_index', type=int, default=0)
    ap.add_argument('--dispatch-gpus', dest='dispatch_gpus', type=int, default=0,
                    help='>0: 单节点该数量 GPU 并行 (视频分片) 后自动合并')
    ap.add_argument('--merge', action='store_true', help='合并多卡 partial_shard*.json 出最终表')
    return ap


def main():
    args = build_argparser().parse_args()
    if args.merge:
        merge_partials(args)
        return
    if args.dispatch_gpus and args.dispatch_gpus > 1:
        for req in ['transformer', 'vae', 'text_encoder']:
            if getattr(args, req) is None:
                raise SystemExit(f'--{req} required')
        run_dispatch(args)
        return
    for req in ['transformer', 'vae', 'text_encoder']:
        if getattr(args, req) is None:
            raise SystemExit(f'--{req} required (除非 --merge)')
    # 允许无 CUDA 时回退 CPU (仅用于调试; 真实探针走 GPU kjob)
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[t4g] WARNING: CUDA 不可用, 回退 CPU (仅调试)。', flush=True)
        args.device = 'cpu'
        if args.dtype == 'bf16':
            args.dtype = 'fp32'
    run_probe(args)


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    main()
