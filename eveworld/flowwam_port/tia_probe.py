#!/usr/bin/env python3
"""TIA 层探针（论文 eq:epe）for FlowWAM/Wan2.2-TI2V-5B stage1。

对采样 episode：VAE 编码干净视频 -> flow-matching 加噪(若干 t) ->
双流 forward（monkey-patch _dual_stream_block_fn 捕获每块 RGB token）->
接力追踪(上一帧目标格特征在当前帧 argmax) vs GDINO GT 的 EPE ->
逐 (block, t) 聚合中位 EPE，选 ℓ*。协议对齐 giga t4g_measure。

DiT patch (1,2,2): token 网格 31x15x20；anno 30x40 格 //2 映射。
flow 流喂白图 latent 加噪（不含信息，仅维持双流布局与联合注意力）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

FLOWWAM_ROOT = os.environ.get("FLOWWAM_ROOT", "/home/jovyan/FlowWAM")
for _p in (FLOWWAM_ROOT, os.path.join(FLOWWAM_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline_loader import build_pipeline  # noqa: E402
import diffsynth.pipelines.wan_video_dual_stream as ds  # noqa: E402

ANNO_DIR = "/data/datasets/gagi/flowwam/igr/anno_640"
N_LAT, GH, GW = 31, 15, 20  # DiT token 网格 (patch 1,2,2 于 latent 31x30x40)
NF, WPIX, HPIX = 121, 640, 480


def read_video_121(path: str) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < NF:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < NF:
        frames += [frames[-1]] * (NF - len(frames))
    return np.stack(frames[:NF])


def gt_cells_token_grid(anno) -> list:
    """anno 30x40 格 -> token 15x20 格（//2），缺测为 None。"""
    out = []
    for e in anno["per_lat_frame"]:
        c = e.get("target_cell")
        out.append(None if c is None else (min(c[0] // 2, GH - 1), min(c[1] // 2, GW - 1)))
    return out


def chained_epe(feat: torch.Tensor, cells: list) -> dict:
    """feat (T,GH,GW,D) fp32; 接力追踪 EPE（token 格单位）。

    RoboTwin 域物体多数帧静止 -> 全体对的中位数饱和为 0；有效信号在
    "目标格发生移动"的帧对上。返回 moving 对的 mean EPE / top-1 命中率
    与全体对 median（向后兼容）。"""
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    epes_all, epes_mov, hit_mov = [], [], []
    for t in range(1, T):
        c0, c1 = cells[t - 1], cells[t]
        if c0 is None or c1 is None:
            continue
        q = fn[t - 1, c0[0], c0[1]]
        sim = fn[t].reshape(H * W, D) @ q
        amax = int(sim.argmax())
        pr = (amax // W, amax % W)
        e = ((pr[0] - c1[0]) ** 2 + (pr[1] - c1[1]) ** 2) ** 0.5
        epes_all.append(e)
        if c0 != c1:
            epes_mov.append(e)
            hit_mov.append(1.0 if pr == tuple(c1) else 0.0)
    return {
        "median_all": float(np.median(epes_all)) if epes_all else float("nan"),
        "mean_moving": float(np.mean(epes_mov)) if epes_mov else float("nan"),
        "hit_moving": float(np.mean(hit_mov)) if hit_mov else float("nan"),
        "n_moving": len(epes_mov),
    }


@torch.no_grad()
def probe_episode(pipe, flow_stream, anno, video, instruction, t_fracs, device):
    dtype = pipe.torch_dtype
    frames = read_video_121(video)
    fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    frames_pil = [Image.fromarray(f) for f in frames]

    pipe.load_models_to_device(["vae"])
    z = pipe.vae.encode(pipe.preprocess_video(frames_pil), device=device).to(dtype=dtype, device=device)
    white = Image.new("RGB", (WPIX, HPIX), (255, 255, 255))
    zf1 = pipe.vae.encode(pipe.preprocess_video([white]), device=device).to(dtype=dtype, device=device)
    zf = zf1.repeat(1, 1, z.shape[2], 1, 1)

    pipe.load_models_to_device(["text_encoder"])
    ctx = pipe.prompter.encode_prompt(instruction, positive=True, device=device)
    pipe.load_models_to_device(pipe.in_iteration_models)

    cells = gt_cells_token_grid(anno)
    result = {}
    for tf in t_fracs:
        g = torch.Generator("cpu").manual_seed(42)
        eps_r = torch.randn(z.shape, generator=g).to(dtype=dtype, device=device)
        eps_f = torch.randn(zf.shape, generator=g).to(dtype=dtype, device=device)
        zt = (1 - tf) * z + tf * eps_r
        zft = (1 - tf) * zf + tf * eps_f
        tt = torch.tensor([tf * 1000.0], dtype=dtype, device=device)

        captured = []
        orig = ds._dual_stream_block_fn

        def wrapped(block, rtok, ftok, *a, **k):
            r, f = orig(block, rtok, ftok, *a, **k)
            captured.append(r.detach()[0].to("cpu", torch.float32))
            return r, f

        ds._dual_stream_block_fn = wrapped
        try:
            ds.model_fn_wan_video_dual_stream(
                dit=pipe.dit, flow_stream=flow_stream,
                latents=zt, flow_latents=zft, timestep=tt, context=ctx,
                fuse_vae_embedding_in_latents=False,
                use_gradient_checkpointing=False,
            )
        finally:
            ds._dual_stream_block_fn = orig

        for bi, tok in enumerate(captured):
            feat = tok.reshape(N_LAT, GH, GW, -1)
            result[(bi, tf)] = chained_epe(feat, cells)  # dict 指标
        del captured
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data/datasets/gagi/flowwam/igr/manifest_640.json")
    ap.add_argument("--ckpt", default="/data/datasets/gagi/flowwam/checkpoints/flowwam_worldarena_stage1.safetensors")
    ap.add_argument("--models-root", default="/data/datasets/gagi/flowwam/models")
    ap.add_argument("--out-dir", default="/data/datasets/gagi/flowwam/igr/tia_probe")
    ap.add_argument("--per-task", type=int, default=3, help="每任务采样条数(取检出率最高者)")
    ap.add_argument("--t-fracs", default="0.1,0.3,0.5")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    t_fracs = [float(x) for x in args.t_fracs.split(",")]
    device = torch.device("cuda:0")

    rows = json.load(open(args.manifest))
    by_key = {f"{r['task']}__{r['episode']}": r for r in rows}
    # 每任务取检出率最高的 per-task 条
    per_task = {}
    for f in sorted(os.listdir(ANNO_DIR)):
        if not f.endswith(".json"):
            continue
        a = json.load(open(os.path.join(ANNO_DIR, f)))
        task = a["vid"].split("/")[0]
        per_task.setdefault(task, []).append((a["n_detected_frames"], f))
    picked = []
    for task, lst in sorted(per_task.items()):
        for _, f in sorted(lst, reverse=True)[: args.per_task]:
            picked.append(f)
    picked = picked[i::n]
    print(f"shard {args.shard}: {len(picked)} episodes", flush=True)

    pipe, flow_stream = build_pipeline(args.models_root, device, full_path=args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)

    agg = {}
    for k, f in enumerate(picked):
        anno = json.load(open(os.path.join(ANNO_DIR, f)))
        row = by_key[f[:-5]]
        try:
            res = probe_episode(pipe, flow_stream, anno, row["video"], row["instruction"], t_fracs, device)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {f}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for key, v in res.items():
            agg.setdefault(f"{key[0]}|{key[1]}", []).append(v)
        print(f"{k + 1}/{len(picked)} {f} ok", flush=True)

    out = os.path.join(args.out_dir, f"probe_shard_{i}_{n}.json")
    json.dump(agg, open(out, "w"))
    print(f"SHARD_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
