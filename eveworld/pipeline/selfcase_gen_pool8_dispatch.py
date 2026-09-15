#!/usr/bin/env python3
"""SELF-CASE · 重考池生成 8 卡满载分发器。

bestofn_dispatch 是"每 seed 钉 1 卡"(8 seed 场景), 重考池只有 4 个协议 seed
(42/314/777/999) 会闲置 4 卡。本分发器把题目 JSON 预先切成 N 半(不相交),
worker = seed x half, 逐个钉卡: 4 seed x 2 half = 8 worker 占满 8 卡。
同 seed 两半写同一 save_dir, 题目不相交无文件碰撞; --skip-existing 补齐语义不变。

用法: python selfcase_gen_pool8_dispatch.py --seeds 42 314 777 999 \
        --data-paths half0.json half1.json --out-root <pool_dir> ... (其余同 bestofn)
"""
import argparse
import os
import subprocess
import sys
import time
from glob import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--data-paths", nargs="+", required=True, help="不相交的题目 JSON 分片")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--lam", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gen-script", required=True)
    ap.add_argument("--eag-weight", default="0")
    ap.add_argument("--num-frames", default="93")
    ap.add_argument("--steps", default="30")
    ap.add_argument("--height", default="480")
    ap.add_argument("--width", default="768")
    ap.add_argument("--fps", default="16")
    ap.add_argument("--poll-sec", type=int, default=60)
    a = ap.parse_args()

    procs = []
    gpu = 0
    for sd in a.seeds:
        save_dir = os.path.join(a.out_root, f"seed{sd}_f{a.num_frames}")
        os.makedirs(save_dir, exist_ok=True)
        for hi, dp in enumerate(a.data_paths):
            log_path = os.path.join(save_dir, f"gen_gpu{gpu}_half{hi}.log")
            cmd = [a.python, a.gen_script,
                   "--data-path", dp, "--save-dir", save_dir,
                   "--transformer", a.transformer, "--text-encoder", a.text_encoder,
                   "--vae", a.vae, "--lam", a.lam,
                   "--eag-weight", a.eag_weight,
                   "--num-inference-steps", a.steps, "--num-frames", a.num_frames,
                   "--height", a.height, "--width", a.width, "--fps", a.fps,
                   "--seed", str(sd), "--limit", "0", "--skip-existing"]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            lf = open(log_path, "w")
            print(f"[dispatch8] GPU {gpu} <- seed {sd} half{hi} -> {save_dir}", flush=True)
            procs.append((gpu, sd, hi, subprocess.Popen(
                cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), save_dir, lf))
            gpu += 1

    print(f"[pool8] {len(procs)} 路已分发到 GPU 0..{gpu - 1}, 等待完成...", flush=True)
    while True:
        alive = [pr for pr in procs if pr[3].poll() is None]
        done = {}
        for _, sd, _, _, save_dir, _ in procs:
            done[sd] = len(glob(os.path.join(save_dir, "generated_only", "*.mp4")))
        print("[pool8] 进度 " + " ".join(f"seed{s}:{n}" for s, n in sorted(done.items()))
              + f" | 存活 worker {len(alive)}/{len(procs)}", flush=True)
        if not alive:
            break
        time.sleep(a.poll_sec)

    bad = [(pr[1], pr[2]) for pr in procs if pr[3].returncode != 0]
    for pr in procs:
        pr[5].close()
    if bad:
        raise SystemExit(f"[pool8] 失败 worker (seed, half): {bad}")
    print("[pool8] 全部完成", flush=True)


if __name__ == "__main__":
    main()
