#!/usr/bin/env python3
"""Dispatch full EVAL-175 inference seeds across one eight-GPU node."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from eveworld.evaluation.x9_eval175_dispatch import MODELS


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_model(model: str) -> Path:
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    model_dir = Path(MODELS[model])
    required = (
        model_dir / "transformer" / "config.json",
        model_dir / "text_encoder" / "config.json",
        model_dir / "vae" / "config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model {model} is incomplete: {missing}")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t4g_wmapA_pre_seed42_s250")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chain-name", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=Path(__file__).with_name("eval175_multiseed_worker.py"),
    )
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.gpu_count != 8:
        raise ValueError("this dispatcher requires exactly eight GPUs")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be non-negative")
    if not args.worker_script.is_file():
        raise FileNotFoundError(args.worker_script)
    model_dir = validate_model(args.model)
    for split, expected in (("gr1_env", 29), ("gr1_object", 50), ("gr1_behavior", 47)):
        data_path = args.data_root / f"eval175_{split}.json"
        rows = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError(f"{data_path}: expected {expected} rows")

    seed_shards = [args.seeds[gpu :: args.gpu_count] for gpu in range(args.gpu_count)]
    model_root = args.output_root / args.model
    log_root = model_root / "logs" / args.chain_name
    plan = {
        "status": "planned" if args.dry_run else "running",
        "chain_name": args.chain_name,
        "host": socket.gethostname(),
        "model": args.model,
        "model_dir": str(model_dir),
        "resolved_model": {
            "transformer": str((model_dir / "transformer").resolve()),
            "text_encoder": str((model_dir / "text_encoder").resolve()),
            "vae": str((model_dir / "vae").resolve()),
        },
        "data_root": str(args.data_root),
        "output": str(model_root),
        "seeds": args.seeds,
        "seed_shards": {str(gpu): seeds for gpu, seeds in enumerate(seed_shards)},
        "gpu_count": args.gpu_count,
        "parameters": {
            "num_inference_steps": args.num_inference_steps,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
            "eag_weight": 0,
        },
        "started_at_unix": None if args.dry_run else time.time(),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    log_root.mkdir(parents=True, exist_ok=True)
    plan_path = log_root / "execution_plan.json"
    atomic_json_dump(plan_path, plan)
    processes = []
    for gpu, seeds in enumerate(seed_shards):
        if not seeds:
            continue
        log_path = log_root / f"gpu{gpu}.log"
        summary_path = log_root / f"gpu{gpu}_summary.json"
        command = [
            args.python,
            str(args.worker_script),
            "--data-root",
            str(args.data_root),
            "--save-root",
            str(model_root),
            "--transformer",
            str(model_dir / "transformer"),
            "--text-encoder",
            str(model_dir / "text_encoder"),
            "--vae",
            str(model_dir / "vae"),
            "--seeds",
            *[str(seed) for seed in seeds],
            "--worker-id",
            f"{args.chain_name}-gpu{gpu}",
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--num-frames",
            str(args.num_frames),
            "--fps",
            str(args.fps),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--summary-path",
            str(summary_path),
        ]
        if args.skip_existing:
            command.append("--skip-existing")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT
        )
        processes.append((gpu, seeds, process, log_path, log_handle))
        print(
            f"[seed-dispatch] chain={args.chain_name} gpu={gpu} seeds={seeds} "
            f"pid={process.pid} log={log_path}",
            flush=True,
        )

    while any(process.poll() is None for _, _, process, _, _ in processes):
        complete = sum(
            (model_root / f"seed{seed:03d}" / "_COMPLETE.json").is_file()
            for seed in args.seeds
        )
        status = " ".join(
            f"g{gpu}={'run' if process.poll() is None else process.returncode}"
            for gpu, _, process, _, _ in processes
        )
        print(
            f"[seed-dispatch] {time.strftime('%H:%M:%S')} chain={args.chain_name} "
            f"complete={complete}/{len(args.seeds)} {status}",
            flush=True,
        )
        time.sleep(args.poll_sec)

    failures = []
    worker_summaries = {}
    for gpu, seeds, process, log_path, log_handle in processes:
        log_handle.close()
        if process.returncode:
            failures.append(
                {"gpu": gpu, "seeds": seeds, "returncode": process.returncode, "log": str(log_path)}
            )
        summary_path = log_root / f"gpu{gpu}_summary.json"
        if summary_path.is_file():
            worker_summaries[str(gpu)] = str(summary_path)

    completed = [
        seed
        for seed in args.seeds
        if (model_root / f"seed{seed:03d}" / "_COMPLETE.json").is_file()
    ]
    plan.update(
        {
            "status": "failed" if failures or len(completed) != len(args.seeds) else "complete",
            "completed_seeds": completed,
            "failures": failures,
            "worker_summaries": worker_summaries,
            "completed_at_unix": time.time(),
        }
    )
    plan["wall_sec"] = round(plan["completed_at_unix"] - plan["started_at_unix"], 3)
    atomic_json_dump(plan_path, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if plan["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
