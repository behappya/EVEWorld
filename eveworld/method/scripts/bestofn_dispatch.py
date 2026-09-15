#!/usr/bin/env python3
"""EVE · best-of-N 单节点8卡分发器(纯 Python, 不用 bash wait —— kjobctl 不支持)。

对每个 seed 起一个 generate_eag.py 子进程, 用 CUDA_VISIBLE_DEVICES 钉到一张卡,
8 seed 并行占满 8 卡, 各卡串行跑 92 条。周期性打印各 seed 完成度, 末尾汇总退出码。
默认补齐模式(--skip-existing): 已有 mp4 跳过, 只生成缺的。
"""
import argparse, os, subprocess, sys, time
from glob import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--lam", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gen-script", required=True, help="generate_eag.py 绝对路径")
    ap.add_argument("--eag-weight", default="0")
    ap.add_argument("--num-frames", default="93")
    ap.add_argument("--steps", default="30")
    ap.add_argument("--height", default="480")
    ap.add_argument("--width", default="768")
    ap.add_argument("--fps", default="16")
    ap.add_argument("--limit", default="0")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--poll-sec", type=int, default=60)
    a = ap.parse_args()

    procs = []   # (gpu, seed, Popen, save_dir, log_path)
    for gpu, sd in enumerate(a.seeds):
        save_dir = os.path.join(a.out_root, f"seed{sd}_f{a.num_frames}")
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, f"gen_gpu{gpu}.log")
        cmd = [a.python, a.gen_script,
               "--data-path", a.data_path, "--save-dir", save_dir,
               "--transformer", a.transformer, "--text-encoder", a.text_encoder,
               "--vae", a.vae, "--lam", a.lam,
               "--eag-weight", a.eag_weight,
               "--num-inference-steps", a.steps, "--num-frames", a.num_frames,
               "--height", a.height, "--width", a.width, "--fps", a.fps,
               "--seed", str(sd), "--limit", a.limit]
        if a.skip_existing:
            cmd.append("--skip-existing")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        lf = open(log_path, "w")
        print(f"[dispatch] GPU {gpu} <- seed {sd} -> {save_dir} (log {log_path})", flush=True)
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((gpu, sd, p, save_dir, log_path, lf))

    print(f"[bon-8gpu] {len(procs)} 路已分发到 GPU 0..{len(procs)-1}, 等待完成...", flush=True)
    # 周期性进度: 直到全部退出
    while True:
        alive = [pr for pr in procs if pr[2].poll() is None]
        counts = []
        for gpu, sd, p, save_dir, _lp, _lf in procs:
            n = len(glob(os.path.join(save_dir, "generated_only", "*.mp4")))
            st = "run" if p.poll() is None else (f"done({p.returncode})" if p.returncode == 0 else f"FAIL({p.returncode})")
            counts.append(f"s{sd}:{n}[{st}]")
        print(f"[bon-8gpu] {time.strftime('%H:%M:%S')} " + " ".join(counts), flush=True)
        if not alive:
            break
        time.sleep(a.poll_sec)

    fail = 0
    for gpu, sd, p, save_dir, log_path, lf in procs:
        lf.close()
        rc = p.returncode
        n = len(glob(os.path.join(save_dir, "generated_only", "*.mp4")))
        if rc != 0:
            fail = 1
            print(f"[bon-8gpu] GPU {gpu} seed {sd} FAILED rc={rc} n={n} -> {log_path}", flush=True)
        else:
            print(f"[bon-8gpu] GPU {gpu} seed {sd} DONE n={n}", flush=True)
    print(f"[bon-8gpu] 全部结束 fail={fail}", flush=True)
    sys.exit(fail)


if __name__ == "__main__":
    main()
