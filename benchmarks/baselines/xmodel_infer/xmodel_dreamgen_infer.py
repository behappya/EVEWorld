#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨模型 DreamGen 批量 I2V 推理（EVE Model-Laziness 跨模型对比）。

一个入口，用 --model-family 切换不同 diffusers 视频生成模型，读 DreamGen 92 条
(image + prompt) 清单，逐条生成视频，输出 generated-only mp4，目录布局对齐 GW-0
（文件名 = <request_id>.mp4），便于统一做偷懒度量。

支持的 family（均为 diffusers 原生 I2V）：
  - wan      : WanImageToVideoPipeline        (Wan2.1-I2V-14B / Wan2.2-I2V-A14B)
  - wan_ti2v : WanPipeline (Wan2.2-TI2V-5B, 文+图→视频)
  - cogvideox: CogVideoXImageToVideoPipeline  (CogVideoX1.5-5B-I2V)
  - cosmos   : Cosmos2VideoToWorldPipeline    (Cosmos-Predict2.5-2B, 单图→视频)

注意：本脚本设计为在 GPU kjob pod 内运行，不在无 GPU 的 workspace 主机跑。
本机只做 --dry-run（打印计划、检查输入清单，不加载模型、不推理）。

用法示例（在 kjob 内，由 kjob_xmodel_dreamgen_infer.sh 调用）：
  python xmodel_dreamgen_infer.py \
    --model-family wan_ti2v \
    --model-path /data/.../xmodels/wan22_ti2v_5b \
    --data-path /data/.../gr1_dreamgen_it2v.json \
    --save-dir  /data/.../xmodel_eval/wan22_ti2v_5b_5p8s \
    --num-frames 93 --fps 16 --height 480 --width 768 \
    --num-inference-steps 30 --seed 6666
"""
import argparse
import json
import os
import sys
import time
import traceback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-family", required=True,
                   choices=["wan", "wan_ti2v", "cogvideox", "cosmos"])
    p.add_argument("--model-path", required=True, help="本地 diffusers 权重目录")
    p.add_argument("--data-path", required=True, help="DreamGen it2v.json（92 条）")
    p.add_argument("--save-dir", required=True, help="输出目录（generated-only mp4）")
    p.add_argument("--num-frames", type=int, default=93)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=6666)
    p.add_argument("--data-limit", type=int, default=0, help="仅跑前 N 条，0=全部（smoke 用）")
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dry-run", action="store_true",
                   help="只检查输入/打印计划，不加载模型（可在无 GPU 主机跑）")
    p.add_argument("--summary-path", type=str, default="")
    p.add_argument("--gpu-ids", type=str, default="0",
                   help="用哪些卡做数据并行，如 '0 1 2 3 4 5 6 7'（92条按卡数切分）")
    return p.parse_args()


def load_manifest(path, limit):
    with open(path, "r") as f:
        data = json.load(f)
    if limit and limit > 0:
        data = data[:limit]
    # 统一取 request_id 作为输出文件名（与 GW-0 的 generated-only 对齐）
    items = []
    for i, d in enumerate(data):
        rid = d.get("request_id") or d.get("id") or f"sample_{i}"
        img = d["image"]
        prompt = d.get("prompt", "")
        items.append({"request_id": rid, "image": img, "prompt": prompt})
    return items


def get_dtype(name):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def build_pipeline(family, model_path, dtype, device="cuda"):
    """加载对应 family 的 diffusers pipeline。返回 (pipe, gen_fn)。

    device: 该进程绑定的卡(如 'cuda:3')，多卡数据并行时每进程不同。
    gen_fn(pipe, image, prompt, args, generator) -> list[np.ndarray frames]
    """
    import torch
    from diffusers.utils import load_image

    if family in ("wan", "wan_ti2v"):
        # Wan2.2 的 I2V：A14B(有image_encoder,MoE双transformer) 和 TI2V-5B(无image_encoder,
        # 靠VAE首帧latent条件) 都用 WanImageToVideoPipeline;image_encoder 在 from_pretrained
        # 时按 repo 是否含该子目录自动加载(5B 缺则为 None,官方支持)。WanPipeline 是纯T2V不吃image。
        from diffusers import WanImageToVideoPipeline as PipeCls
        pipe = PipeCls.from_pretrained(model_path, torch_dtype=dtype)
        pipe.to(device)

        def gen_fn(pipe, image, prompt, a, generator):
            kwargs = dict(
                image=image,
                prompt=prompt,
                negative_prompt=a.negative_prompt or None,
                height=a.height, width=a.width,
                num_frames=a.num_frames,
                num_inference_steps=a.num_inference_steps,
                guidance_scale=a.guidance_scale,
                generator=generator,
            )
            out = pipe(**kwargs)
            return out.frames[0]

        return pipe, gen_fn

    if family == "cogvideox":
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.to(device)
        pipe.vae.enable_tiling()

        def gen_fn(pipe, image, prompt, a, generator):
            out = pipe(
                image=image, prompt=prompt,
                num_frames=a.num_frames,
                num_inference_steps=a.num_inference_steps,
                guidance_scale=a.guidance_scale,
                generator=generator,
            )
            return out.frames[0]

        return pipe, gen_fn

    if family == "cosmos":
        from diffusers import Cosmos2VideoToWorldPipeline

        # 本地 benchmark 评测用直通 safety checker:官方 cosmos_guardrail 需在线下载
        # 模型, 离线环境不可用;输入为固定 benchmark prompt(机器人操作), 无过滤需求。
        class _PassthroughSafetyChecker:
            def to(self, *args, **kwargs):
                return self

            def check_text_safety(self, *args, **kwargs):
                return True

            def check_video_safety(self, video, *args, **kwargs):
                return video

        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
            model_path, torch_dtype=dtype, safety_checker=_PassthroughSafetyChecker())
        pipe.to(device)

        def gen_fn(pipe, image, prompt, a, generator):
            out = pipe(
                image=image, prompt=prompt,
                height=a.height, width=a.width,
                num_frames=a.num_frames,
                num_inference_steps=a.num_inference_steps,
                guidance_scale=a.guidance_scale,
                generator=generator,
            )
            return out.frames[0]

        return pipe, gen_fn

    raise ValueError(f"unknown family: {family}")


def run_worker(rank, gpu_ids, all_items, args):
    """单个 GPU 进程：绑一张卡, 跑 all_items[rank::N] 分片, 写各自视频 + 分片 summary。"""
    import torch
    from diffusers.utils import load_image, export_to_video

    n = len(gpu_ids)
    gid = gpu_ids[rank]
    torch.cuda.set_device(gid)
    dev = f"cuda:{gid}"
    my_items = all_items[rank::n]        # 跨步切分, 均匀
    tag = f"[rank{rank}/{n} gpu{gid}]"
    print(f"{tag} 分到 {len(my_items)}/{len(all_items)} 条", flush=True)

    # 空分片(如 4 条数据 8 卡, rank>=4)：不加载模型, 直接写空 shard 退出
    if len(my_items) == 0:
        with open(os.path.join(args.save_dir, f".shard_{rank}.json"), "w") as f:
            json.dump({"rank": rank, "gpu": gid, "n_ok": 0, "n_total": 0, "results": []}, f)
        print(f"{tag} 无分片, 跳过", flush=True)
        return

    dtype = get_dtype(args.dtype)
    t0 = time.time()
    print(f"{tag} 加载 {args.model_family} pipeline ...", flush=True)
    pipe, gen_fn = build_pipeline(args.model_family, args.model_path, dtype, device=dev)
    print(f"{tag} 加载完成 {time.time()-t0:.1f}s", flush=True)

    results = []
    n_ok = 0
    for j, it in enumerate(my_items):
        rid = it["request_id"]
        out_path = os.path.join(args.save_dir, f"{rid}.mp4")
        if os.path.exists(out_path):
            print(f"{tag} [skip] {j+1}/{len(my_items)} {rid}", flush=True)
            n_ok += 1
            results.append({"request_id": rid, "status": "exists", "path": out_path})
            continue
        try:
            image = load_image(it["image"]).resize((args.width, args.height))
            generator = torch.Generator(device=dev).manual_seed(args.seed)
            ts = time.time()
            frames = gen_fn(pipe, image, it["prompt"], args, generator)
            export_to_video(frames, out_path, fps=args.fps)
            dt = time.time() - ts
            n_ok += 1
            print(f"{tag} [ok] {j+1}/{len(my_items)} {rid} ({dt:.1f}s)", flush=True)
            results.append({"request_id": rid, "status": "ok", "path": out_path, "sec": dt})
        except Exception as e:
            print(f"{tag} [FAIL] {j+1}/{len(my_items)} {rid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results.append({"request_id": rid, "status": "fail", "error": str(e)})

    # 写分片 summary（主进程再合并）
    shard_path = os.path.join(args.save_dir, f".shard_{rank}.json")
    with open(shard_path, "w") as f:
        json.dump({"rank": rank, "gpu": gid, "n_ok": n_ok,
                   "n_total": len(my_items), "results": results}, f, ensure_ascii=False)
    print(f"{tag} 完成 ok={n_ok}/{len(my_items)}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    items = load_manifest(args.data_path, args.data_limit)
    gpu_ids = [int(x) for x in str(args.gpu_ids).replace(",", " ").split()]

    print("=" * 60)
    print(f"[xmodel-infer] family={args.model_family}")
    print(f"  model_path = {args.model_path}")
    print(f"  data_path  = {args.data_path}  ({len(items)} items)")
    print(f"  save_dir   = {args.save_dir}")
    print(f"  size       = {args.width}x{args.height}  frames={args.num_frames} fps={args.fps}")
    print(f"  steps={args.num_inference_steps} cfg={args.guidance_scale} seed={args.seed} dtype={args.dtype}")
    print(f"  gpu_ids    = {gpu_ids}  (数据并行 {len(gpu_ids)} 卡)")
    print("=" * 60)

    missing = [it for it in items if not os.path.exists(it["image"])]
    if missing:
        print(f"[WARN] {len(missing)} 条 image 路径不存在，示例: {missing[0]['image']}")
    if not os.path.isdir(args.model_path):
        print(f"[WARN] model_path 不存在: {args.model_path}")

    if args.dry_run:
        print(f"[dry-run] 通过：{len(items)} 条待生成, {len(gpu_ids)} 卡切分, 前 3 条：")
        for it in items[:3]:
            print(f"    {it['request_id']}  <- {os.path.basename(it['image'])}")
        print("[dry-run] 未加载模型、未推理。")
        return 0

    import torch.multiprocessing as mp
    t0 = time.time()
    # 清旧分片
    for r in range(len(gpu_ids)):
        p = os.path.join(args.save_dir, f".shard_{r}.json")
        if os.path.exists(p):
            os.remove(p)

    if len(gpu_ids) == 1:
        run_worker(0, gpu_ids, items, args)
    else:
        mp.start_processes(run_worker, nprocs=len(gpu_ids),
                           args=(gpu_ids, items, args), start_method="spawn")

    # 合并分片 summary
    results, n_ok = [], 0
    for r in range(len(gpu_ids)):
        p = os.path.join(args.save_dir, f".shard_{r}.json")
        if os.path.exists(p):
            sd = json.load(open(p))
            results.extend(sd["results"])
            n_ok += sd["n_ok"]
    summary = {
        "model_family": args.model_family, "model_path": args.model_path,
        "total": len(items), "ok": n_ok, "gpu_ids": gpu_ids,
        "num_frames": args.num_frames, "fps": args.fps,
        "height": args.height, "width": args.width,
        "steps": args.num_inference_steps, "seed": args.seed,
        "wall_sec": time.time() - t0, "results": results,
    }
    sp = args.summary_path or os.path.join(args.save_dir, "generation_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] {n_ok}/{len(items)} 生成成功 (用时 {time.time()-t0:.0f}s)，summary: {sp}")
    return 0 if n_ok == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
