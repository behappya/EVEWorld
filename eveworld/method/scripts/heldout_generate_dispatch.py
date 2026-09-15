#!/usr/bin/env python3
"""Use all GPUs of one node for one held-out method/checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def selected_count(manifest: Path, expected_split: str, limit: int) -> int:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [row for row in rows if str(row.get("split", "")) != expected_split]
    if bad:
        raise ValueError(f"manifest contains {len(bad)} rows outside split={expected_split!r}")
    return min(limit, len(rows)) if limit > 0 else len(rows)


def merge_shards(seed_root: Path, shard_count: int, expected: int) -> None:
    for kind in ("generated_only", "side_by_side"):
        merged = seed_root / kind
        merged.mkdir(parents=True, exist_ok=True)
        for shard_index in range(shard_count):
            source_dir = seed_root / f"shard_{shard_index}" / kind
            for source in sorted(source_dir.glob("*.mp4")):
                target = merged / source.name
                if target.is_symlink() and target.resolve() == source.resolve():
                    continue
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"merge collision: {target}")
                target.symlink_to(source.resolve())
        count = len(list(merged.glob("*.mp4")))
        if count != expected:
            raise RuntimeError(f"{seed_root}: merged {count} {kind} videos, expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("joint", "frontier"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--training-seed", required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--lora", default="NONE")
    parser.add_argument("--joint-script", type=Path, required=True)
    parser.add_argument("--frontier-script", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    if args.gpu_count != 8:
        raise ValueError("held-out generation must request exactly 8 GPUs")
    if not args.generation_seeds or args.gpu_count % len(args.generation_seeds):
        raise ValueError("generation seed count must divide 8 so every GPU receives one shard")
    if "/" in args.method or "/" in args.training_seed:
        raise ValueError("method and training-seed must be path-safe")
    lora = None if args.lora.upper() == "NONE" else Path(args.lora)
    if lora is not None and not lora.is_dir():
        raise FileNotFoundError(lora)
    if args.pipeline == "frontier" and lora is None:
        raise ValueError("frontier generation requires a LoRA checkpoint")

    expected = selected_count(args.split_manifest, args.expected_split, args.limit)
    shards_per_seed = args.gpu_count // len(args.generation_seeds)
    script = args.frontier_script if args.pipeline == "frontier" else args.joint_script
    processes = []
    gpu = 0
    for seed in args.generation_seeds:
        seed_root = args.out_root / args.method / f"train_seed_{args.training_seed}" / f"gen_seed_{seed}"
        for shard_index in range(shards_per_seed):
            save_dir = seed_root / f"shard_{shard_index}"
            save_dir.mkdir(parents=True, exist_ok=True)
            log_path = save_dir / f"gen_gpu{gpu}.log"
            command = [
                args.python,
                str(script),
                "--data-path", str(args.data_path),
                "--split-manifest", str(args.split_manifest),
                "--expected-split", args.expected_split,
                "--save-dir", str(save_dir),
                "--transformer", str(args.transformer),
                "--text-encoder", str(args.text_encoder),
                "--vae", str(args.vae),
                "--seed", str(seed),
                "--limit", str(args.limit),
                "--shard-index", str(shard_index),
                "--num-shards", str(shards_per_seed),
                "--num-inference-steps", str(args.steps),
                "--num-frames", str(args.num_frames),
                "--height", str(args.height),
                "--width", str(args.width),
                "--fps", str(args.fps),
            ]
            if lora is not None:
                command.extend(("--lora", str(lora)))
            if args.pipeline == "frontier":
                command.extend(("--block-size", str(args.block_size)))
            if args.skip_existing:
                command.append("--skip-existing")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONUNBUFFERED"] = "1"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT
            )
            processes.append((gpu, seed, shard_index, process, log_path, log_handle))
            print(
                f"[heldout-dispatch] gpu={gpu} seed={seed} shard={shard_index}/{shards_per_seed}",
                flush=True,
            )
            gpu += 1

    while any(item[3].poll() is None for item in processes):
        status = [
            f"g{gpu}:s{seed}q{shard}=" + ("run" if process.poll() is None else str(process.returncode))
            for gpu, seed, shard, process, _, _ in processes
        ]
        print(f"[heldout-dispatch] {time.strftime('%H:%M:%S')} {' '.join(status)}", flush=True)
        time.sleep(args.poll_sec)

    failed = []
    for gpu, seed, shard, process, log_path, log_handle in processes:
        log_handle.close()
        if process.returncode:
            failed.append(f"gpu={gpu},seed={seed},shard={shard},log={log_path}")
    if failed:
        raise SystemExit("failed generation shards: " + "; ".join(failed))

    for seed in args.generation_seeds:
        seed_root = args.out_root / args.method / f"train_seed_{args.training_seed}" / f"gen_seed_{seed}"
        merge_shards(seed_root, shards_per_seed, expected)

    summary = {
        "pipeline": args.pipeline,
        "method": args.method,
        "training_seed": args.training_seed,
        "generation_seeds": args.generation_seeds,
        "gpu_count": args.gpu_count,
        "shards_per_seed": shards_per_seed,
        "expected_videos_per_seed": expected,
        "split_manifest": str(args.split_manifest),
        "expected_split": args.expected_split,
        "lora": str(lora) if lora else None,
    }
    summary_path = args.out_root / args.method / f"train_seed_{args.training_seed}" / "dispatch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
