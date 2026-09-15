#!/usr/bin/env python3
"""Pin frontier generation shards to the GPUs of one cluster node."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_task(value: str) -> tuple[str, Path, int, int, int, float]:
    parts = value.split("::")
    if len(parts) not in {5, 6}:
        raise ValueError("task must be NAME::LORA::SEED::SHARD_INDEX::NUM_SHARDS[::GUARD_STRENGTH]")
    name, lora, seed, shard_index, num_shards = parts[:5]
    guard_strength = float(parts[5]) if len(parts) == 6 else 0.0
    path = Path(lora)
    if not name or "/" in name or not path.exists():
        raise ValueError(f"invalid task: {value}")
    return name, path, int(seed), int(shard_index), int(num_shards), guard_strength


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--expected-split", default="train")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--gen-script", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", default="8")
    parser.add_argument("--steps", default="30")
    parser.add_argument("--num-frames", default="93")
    parser.add_argument("--height", default="480")
    parser.add_argument("--width", default="768")
    parser.add_argument("--fps", default="16")
    parser.add_argument("--block-size", default="4")
    parser.add_argument("--boundary-guard-velocity-scale", default="0.5")
    parser.add_argument("--boundary-guard-decay", default="1.0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    tasks = [parse_task(value) for value in args.tasks]
    if len(tasks) > 8:
        raise ValueError("one node exposes at most 8 GPUs")
    if len({task[0] for task in tasks}) != len(tasks):
        raise ValueError("task names must be unique")

    processes = []
    for gpu, (name, lora, seed, shard_index, num_shards, guard_strength) in enumerate(tasks):
        save_dir = Path(args.out_root) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / f"gen_gpu{gpu}.log"
        command = [
            args.python,
            args.gen_script,
            "--data-path", args.data_path,
            "--split-manifest", args.split_manifest,
            "--expected-split", args.expected_split,
            "--save-dir", str(save_dir),
            "--transformer", args.transformer,
            "--text-encoder", args.text_encoder,
            "--vae", args.vae,
            "--lora", str(lora),
            "--seed", str(seed),
            "--limit", args.limit,
            "--shard-index", str(shard_index),
            "--num-shards", str(num_shards),
            "--num-inference-steps", args.steps,
            "--num-frames", args.num_frames,
            "--height", args.height,
            "--width", args.width,
            "--fps", args.fps,
            "--block-size", args.block_size,
            "--boundary-guard-strength", str(guard_strength),
            "--boundary-guard-velocity-scale", args.boundary_guard_velocity_scale,
            "--boundary-guard-decay", args.boundary_guard_decay,
        ]
        if args.skip_existing:
            command.append("--skip-existing")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((gpu, name, process, log_path, log_handle))
        print(f"[dispatch] gpu={gpu} task={name} pid={process.pid}", flush=True)

    while any(process.poll() is None for _, _, process, _, _ in processes):
        status = [
            f"{name}:{'run' if process.poll() is None else process.returncode}"
            for _, name, process, _, _ in processes
        ]
        print(f"[dispatch] {time.strftime('%H:%M:%S')} {' '.join(status)}", flush=True)
        time.sleep(args.poll_sec)

    failed = []
    for gpu, name, process, log_path, log_handle in processes:
        log_handle.close()
        if process.returncode:
            failed.append(name)
        print(f"[dispatch] gpu={gpu} task={name} rc={process.returncode} log={log_path}", flush=True)
    if failed:
        raise SystemExit(f"failed tasks: {','.join(failed)}")


if __name__ == "__main__":
    main()
