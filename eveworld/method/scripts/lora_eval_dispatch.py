#!/usr/bin/env python3
"""Run paired LoRA generation tasks on separate GPUs of one node."""

import argparse
import os
import subprocess
import sys
import time
from glob import glob


def parse_task(spec: str) -> tuple[str, str, str]:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(f"bad task {spec!r}; expected NAME::LORA_PATH::SEED")
    name, lora_path, seed = parts
    if not name or "/" in name:
        raise ValueError(f"invalid task name: {name!r}")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(lora_path)
    int(seed)
    return name, lora_path, seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--lam", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gen-script", required=True)
    ap.add_argument("--num-frames", default="93")
    ap.add_argument("--steps", default="30")
    ap.add_argument("--height", default="480")
    ap.add_argument("--width", default="768")
    ap.add_argument("--fps", default="16")
    ap.add_argument("--limit", default="16")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--poll-sec", type=int, default=60)
    args = ap.parse_args()

    tasks = [parse_task(spec) for spec in args.tasks]
    if len(tasks) > 8:
        raise ValueError(f"one node has 8 GPUs, got {len(tasks)} tasks")
    names = [task[0] for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("task names must be unique")

    procs = []
    for gpu, (name, lora_path, seed) in enumerate(tasks):
        save_dir = os.path.join(args.out_root, name)
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, f"gen_gpu{gpu}.log")
        cmd = [
            args.python,
            args.gen_script,
            "--data-path", args.data_path,
            "--save-dir", save_dir,
            "--transformer", args.transformer,
            "--text-encoder", args.text_encoder,
            "--vae", args.vae,
            "--lam", args.lam,
            "--lora", lora_path,
            "--eag-weight", "0",
            "--num-inference-steps", args.steps,
            "--num-frames", args.num_frames,
            "--height", args.height,
            "--width", args.width,
            "--fps", args.fps,
            "--seed", seed,
            "--limit", args.limit,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"[lora-eval] GPU {gpu} <- {name} seed={seed}", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((gpu, name, proc, save_dir, log_path, log_file))

    while True:
        alive = [item for item in procs if item[2].poll() is None]
        status = []
        for gpu, name, proc, save_dir, _log_path, _log_file in procs:
            count = len(glob(os.path.join(save_dir, "generated_only", "*.mp4")))
            state = "run" if proc.poll() is None else f"done({proc.returncode})"
            status.append(f"{name}:{count}[{state}]")
        print(f"[lora-eval] {time.strftime('%H:%M:%S')} " + " ".join(status), flush=True)
        if not alive:
            break
        time.sleep(args.poll_sec)

    failed = False
    for gpu, name, proc, save_dir, log_path, log_file in procs:
        log_file.close()
        count = len(glob(os.path.join(save_dir, "generated_only", "*.mp4")))
        if proc.returncode != 0:
            failed = True
            print(f"[lora-eval] GPU {gpu} {name} FAILED rc={proc.returncode} n={count}: {log_path}", flush=True)
        else:
            print(f"[lora-eval] GPU {gpu} {name} DONE n={count}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
