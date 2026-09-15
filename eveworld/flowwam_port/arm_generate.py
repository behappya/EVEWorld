#!/usr/bin/env python3
"""Generate episodes from trained FlowWAM arm checkpoints (single process / sharded).

Used for the held-out RoboTwin evaluation and other manifest-driven
generation campaigns.

加载顺序: Wan2.2 基座 -> flowwam stage1(full) -> 臂增量 ckpt
(LoRA 合入 + flow_stream/modulation 覆盖); arm=eve 时再载 TIA adapter
并在 block ℓ*=12 注入(与训练一致)。生成协议与 smoke 相同:
官方首帧+manifest prompt, 双流 Stage-1 去噪, 只解码 RGB。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
FLOWWAM_ROOT = os.environ.get("FLOWWAM_ROOT", "/home/jovyan/FlowWAM")
for _p in (FLOWWAM_ROOT, os.path.join(FLOWWAM_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline_loader import build_pipeline  # noqa: E402
from diffsynth.models.utils import load_state_dict  # noqa: E402
from diffsynth.pipelines.wan_video_dual_stream import (  # noqa: E402
    model_fn_wan_video_dual_stream,
)
from diffsynth.data.video import save_video  # noqa: E402

from tia_adapter import TIAAdapter, TIAInjection  # noqa: E402
from dual_stream_cond import model_fn_dual_stream_flowcond  # noqa: E402

L_STAR_DEFAULT = 12
TOKEN_GH, TOKEN_GW = 15, 20


def load_arm(models_root: str, stage1: str, arm_ckpt: str | None, device) -> tuple:
    pipe, flow_stream = build_pipeline(models_root, device, full_path=stage1)
    if not arm_ckpt:
        flow_stream = flow_stream.to(device=device, dtype=torch.bfloat16).eval()
        print("[ArmLoad] Stage-1 only; no fine-tuning checkpoint loaded", flush=True)
        return pipe, flow_stream, None
    state = load_state_dict(arm_ckpt)
    lora_keys, dit_keys, flow_keys, tia_keys = {}, {}, {}, {}
    for k, v in state.items():
        if "lora_A" in k or "lora_B" in k:
            lora_keys[k] = v
        elif k.startswith("flow_stream."):
            flow_keys[k[len("flow_stream."):]] = v
        elif k.startswith("tia_adapter."):
            tia_keys[k[len("tia_adapter."):]] = v
        else:
            dit_keys[k] = v
    if dit_keys:
        missing, unexpected = pipe.dit.load_state_dict(dit_keys, strict=False)
        print(f"[ArmLoad] dit deltas: {len(dit_keys) - len(unexpected)} keys", flush=True)
    if flow_keys:
        flow_stream.load_state_dict(flow_keys, strict=False)
        print(f"[ArmLoad] flow_stream: {len(flow_keys)} keys", flush=True)
    if lora_keys:
        pipe.load_lora(pipe.dit, state_dict=lora_keys, alpha=1.0)
        print(f"[ArmLoad] LoRA: {len(lora_keys)} keys merged", flush=True)
    adapter = None
    if tia_keys:
        adapter = TIAAdapter(channels=int(pipe.dit.dim), rank=tia_keys["input_proj.weight"].shape[0])
        adapter.load_state_dict(tia_keys)
        adapter = adapter.to(device=device).eval()
        print(f"[ArmLoad] TIA adapter: rank={adapter.rank} "
              f"out_norm={adapter.output_proj.weight.detach().float().norm():.4f}", flush=True)
    flow_stream = flow_stream.to(device=device, dtype=torch.bfloat16).eval()
    return pipe, flow_stream, adapter


def read_frames(path: str, n: int, start: int = 0):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    out = []
    idx = 0
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        if idx >= start:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return out


_FLOW_TOOLS = {}


def flow_cond_latents(pipe, robot_video: str, n_frames: int, w: int, h: int, device, dtype,
                      start: int = 0):
    """robot_only 渲染 -> 官方 process_camera_flow -> VAE 编码为干净 flow latent。"""
    if "raft" not in _FLOW_TOOLS:
        from raft_flow_extractor import RAFTFlowExtractor
        from reversible_flow_codec import FlowCodec
        _FLOW_TOOLS["raft"] = RAFTFlowExtractor(device=str(device))
        _FLOW_TOOLS["codec"] = FlowCodec()
    from flow_prefix_utils import process_camera_flow
    frames = read_frames(robot_video, n_frames, start=start)
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    flow_pil, _mags = process_camera_flow(
        frames, (w, h), _FLOW_TOOLS["codec"],
        flow_method="raft", raft_extractor=_FLOW_TOOLS["raft"])
    flow_pil = flow_pil[:n_frames]
    z = pipe.vae.encode(pipe.preprocess_video(flow_pil), device=device)
    return z.to(dtype=dtype, device=device)


@torch.no_grad()
def generate_one(pipe, flow_stream, adapter, l_star, image, prompt,
                 num_frames, width, height, steps, sigma_shift, seed, device,
                 robot_video: str | None = None, cfg_scale: float = 1.0,
                 flow_start: int = 0):
    dtype = pipe.torch_dtype
    vae_z = getattr(pipe.vae, "z_dim", 16)
    h, w, vf = pipe.check_resize_height_width(height, width, num_frames)
    img = (Image.open(image) if isinstance(image, str) else image)
    img = img.convert("RGB").resize((w, h), Image.BICUBIC)

    pipe.load_models_to_device(["text_encoder"])
    ctx = pipe.prompter.encode_prompt(prompt, positive=True, device=device)
    ctx_neg = (pipe.prompter.encode_prompt("", positive=False, device=device)
               if cfg_scale > 1.0 else None)
    pipe.load_models_to_device(["vae"])
    up = pipe.vae.upsampling_factor
    t_lat = (vf - 1) // 4 + 1
    rgb_prefix = pipe.vae.encode(pipe.preprocess_video([img]), device=device).to(dtype=dtype, device=device)

    flow_cond = None
    if robot_video is not None:
        flow_cond = flow_cond_latents(pipe, robot_video, vf, w, h, device, dtype,
                                      start=flow_start)
        flow_prefix = flow_cond[:, :, :1]
    else:
        zero_flow = Image.new("RGB", (w, h), (255, 255, 255))
        flow_prefix = pipe.vae.encode(pipe.preprocess_video([zero_flow]), device=device).to(dtype=dtype, device=device)

    shape = (1, vae_z, t_lat, h // up, w // up)
    rgb = pipe.generate_noise(shape, seed=seed, rand_device="cpu").to(dtype=dtype, device=device)
    rgb[:, :, :1] = rgb_prefix
    if flow_cond is not None:
        flow = flow_cond.clone()          # 全帧干净光流条件(teacher-forcing)
        model_fn = model_fn_dual_stream_flowcond
    else:
        flow = pipe.generate_noise(shape, seed=seed + 1, rand_device="cpu").to(dtype=dtype, device=device)
        flow[:, :, :1] = flow_prefix
        model_fn = model_fn_wan_video_dual_stream

    pipe.scheduler.set_timesteps(steps, shift=sigma_shift)
    pipe.load_models_to_device(pipe.in_iteration_models)

    def denoise_loop():
        nonlocal rgb, flow
        for i, ts in enumerate(pipe.scheduler.timesteps):
            tt = ts.unsqueeze(0).to(dtype=dtype, device=device)
            rp, fp = model_fn(
                dit=pipe.dit, flow_stream=flow_stream,
                latents=rgb, flow_latents=flow, timestep=tt, context=ctx,
                fuse_vae_embedding_in_latents=True,
                use_gradient_checkpointing=False)
            if ctx_neg is not None:
                rp_u, fp_u = model_fn(
                    dit=pipe.dit, flow_stream=flow_stream,
                    latents=rgb, flow_latents=flow, timestep=tt, context=ctx_neg,
                    fuse_vae_embedding_in_latents=True,
                    use_gradient_checkpointing=False)
                rp = rp_u + cfg_scale * (rp - rp_u)
                fp = fp_u + cfg_scale * (fp - fp_u)
            rgb = pipe.scheduler.step(rp, pipe.scheduler.timesteps[i], rgb)
            rgb[:, :, :1] = rgb_prefix
            if flow_cond is None:
                flow = pipe.scheduler.step(fp, pipe.scheduler.timesteps[i], flow)
                flow[:, :, :1] = flow_prefix
            # flow_cond 模式: flow 恒为干净条件, 不去噪

    if adapter is not None:
        with TIAInjection(adapter, l_star, (t_lat, TOKEN_GH, TOKEN_GW)):
            denoise_loop()
    else:
        denoise_loop()

    pipe.load_models_to_device(["vae"])
    return pipe.vae_output_to_video(pipe.vae.decode(rgb, device=device))


def chunk_starts(total: int, win: int = 121, step: int = 110) -> list[int]:
    """完整覆盖 [0,total) 的窗口起点; 相邻窗口重叠 win-step 帧, 末窗贴尾对齐。"""
    if total <= win:
        return [0]
    starts = [0]
    while starts[-1] + win < total:
        nxt = starts[-1] + step
        if nxt + win > total:
            nxt = total - win
        starts.append(nxt)
    return starts


CROSSFADE = 10  # 接缝交叉淡化帧数(须 < win-step 重叠)


def generate_full_traj(pipe, flow_stream, adapter, l_star, image_path, prompt,
                       total, width, height, steps, sigma_shift, seed, device,
                       robot_video, cfg_scale, resume_video: str | None = None):
    """分块滚动生成完整轨迹。

    块k首帧=已生成序列第s_k帧(避开块尾劣化区: 重叠11帧), 光流条件取渲染
    [s_k,s_k+121); 重叠尾部 CROSSFADE 帧旧->新线性淡化抹平接缝。
    resume_video: 已有的 121 帧生成(同seed同条件) -> 直接作为 chunk0。
    """
    import numpy as np
    win = 121
    starts = chunk_starts(total, win)
    acc: list = []
    covered = 0
    for ci, s in enumerate(starts):
        if ci == 0 and resume_video and os.path.exists(resume_video):
            acc = [Image.fromarray(f) for f in read_frames(resume_video, win)]
            if len(acc) == win:
                covered = win
                print(f"  chunk 1/{len(starts)} 复用已有生成 {win}帧", flush=True)
                continue
            acc = []
        img = image_path if ci == 0 else acc[s]
        frames = generate_one(pipe, flow_stream, adapter, l_star, img, prompt,
                              win, width, height, steps, sigma_shift,
                              seed + 1000 * ci, device,
                              robot_video=robot_video, cfg_scale=cfg_scale,
                              flow_start=s)
        if ci == 0:
            acc = list(frames)
        else:
            for j in range(CROSSFADE):
                abs_i = covered - CROSSFADE + j
                alpha = (j + 1) / (CROSSFADE + 1)
                acc[abs_i] = Image.blend(acc[abs_i].convert("RGB"),
                                         frames[abs_i - s].convert("RGB"), alpha)
            acc.extend(frames[covered - s:])
        covered = s + win
        print(f"  chunk {ci + 1}/{len(starts)} start={s} 累计={len(acc)}帧", flush=True)
    return acc[:total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm-ckpt", default="",
        help="Fine-tuning checkpoint. Leave empty to evaluate FlowWAM Stage-1.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True, help="Episode manifest JSON.")
    ap.add_argument("--models-root", default="/data/datasets/gagi/flowwam/models")
    ap.add_argument("--stage1", required=True, help="Released FlowWAM Stage-1 checkpoint (.safetensors).")
    ap.add_argument("--l-star", type=int, default=L_STAR_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help=">0: 只生成前 N 条 episode")
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--sigma-shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--flow-cond", choices=["none", "robot_only"], default="none",
                    help="robot_only: 用 manifest 的 robot_only_video 做光流条件(HDF5 动作驱动)")
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--tia-inject", choices=["on","off"], default="on")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--full-traj", choices=["on", "off", "direct"], default="off",
                    help="on: 121帧窗口滚动生成; direct: 按实际长度一次生成(实验模式)")
    ap.add_argument("--episodes", default="", help="逗号分隔 request_id 过滤")
    ap.add_argument("--resume-from-video", default="",
                    help="已有 121 帧生成的目录(同seed同条件), chunk0 直接复用")
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    rows = json.load(open(args.manifest))
    if args.episodes:
        want = set(args.episodes.split(","))
        rows = [r for r in rows if r["request_id"] in want]
    if args.offset > 0:
        rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[: args.limit]
    rows = rows[i::n]
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0")
    pipe, flow_stream, adapter = load_arm(args.models_root, args.stage1, args.arm_ckpt, device)
    if args.tia_inject == "off":
        adapter = None  # score-first: 推理期注入实测微降分, 关闭

    done = skip = 0
    for k, row in enumerate(rows):
        out_path = os.path.join(args.out, f"{row['request_id']}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skip += 1
            continue
        t0 = time.time()
        robot_video = None
        n_frames = args.num_frames
        total = 0
        if args.flow_cond == "robot_only":
            robot_video = row["robot_only_video"]
            if not os.path.exists(robot_video):
                print(f"SKIP_NO_RENDER {row['request_id']}", flush=True)
                continue
            import cv2
            cap = cv2.VideoCapture(robot_video)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            n = min(args.num_frames, max(total, 5))
            n_frames = 4 * ((n - 1) // 4) + 1
        if args.full_traj == "direct" and robot_video is not None:
            # Experimental path: exercise the model on the whole trajectory in
            # one diffusion call. The pipeline rounds temporal length to its
            # VAE divisibility rule, so trim the decoded output back to the
            # render's exact frame count before writing the mp4.
            video = generate_one(pipe, flow_stream, adapter, args.l_star,
                                 row["image"], row["prompt"], total,
                                 args.width, args.height, args.steps,
                                 args.sigma_shift, args.seed, device,
                                 robot_video=robot_video, cfg_scale=args.cfg_scale)
            video = video[:total]
        elif args.full_traj == "on" and robot_video is not None and total > args.num_frames:
            resume = (os.path.join(args.resume_from_video, f"{row['request_id']}.mp4")
                      if args.resume_from_video else None)
            video = generate_full_traj(pipe, flow_stream, adapter, args.l_star,
                                       row["image"], row["prompt"], total,
                                       args.width, args.height, args.steps,
                                       args.sigma_shift, args.seed, device,
                                       robot_video, args.cfg_scale,
                                       resume_video=resume)
        else:
            video = generate_one(pipe, flow_stream, adapter, args.l_star,
                                 row["image"], row["prompt"], n_frames,
                                 args.width, args.height, args.steps,
                                 args.sigma_shift, args.seed, device,
                                 robot_video=robot_video, cfg_scale=args.cfg_scale)
        save_video(video, out_path, fps=args.fps)
        done += 1
        if done % 5 == 0 or k == len(rows) - 1:
            print(f"shard {args.shard}: {k + 1}/{len(rows)} done={done} skip={skip} "
                  f"last={time.time() - t0:.1f}s", flush=True)
    print(f"SHARD_DONE {args.shard} done={done} skip={skip}", flush=True)


if __name__ == "__main__":
    main()
