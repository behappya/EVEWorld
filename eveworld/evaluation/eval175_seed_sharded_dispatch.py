#!/usr/bin/env python3
"""Generate each inference seed serially while sharding its 126 tasks over 8 GPUs."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eveworld.evaluation.eval175_multiseed_worker import (
    atomic_json_dump,
    load_items,
    seed_complete,
    seed_output_counts,
)
from eveworld.evaluation.x9_eval175_dispatch import MODELS


def validate_model(model: str, model_dir_override: Path | None = None) -> Path:
    if model_dir_override is not None:
        model_dir = model_dir_override
    else:
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


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="round0")
    parser.add_argument("--model-dir", type=Path)
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
    parser.add_argument("--poll-sec", type=int, default=30)
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

    model_dir = validate_model(args.model, args.model_dir)
    items = load_items(args.data_root)
    model_root = args.output_root / args.model
    log_root = model_root / "logs" / args.chain_name
    plan_path = log_root / "execution_plan.json"
    plan: dict[str, Any] = {
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
        "data_root": str(args.data_root.resolve()),
        "output": str(model_root.resolve()),
        "seeds": args.seeds,
        "execution": "seeds serial; 126 tasks per seed sharded over 8 GPUs",
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
        "seed_summaries": {},
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    log_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(plan_path, plan)
    failures: list[dict[str, Any]] = []

    for seed in args.seeds:
        seed_name = f"seed{seed:03d}"
        seed_root = model_root / seed_name
        if args.skip_existing and seed_complete(model_root, seed, items):
            print(f"[shard-dispatch] {seed_name} already complete; skipping", flush=True)
            continue

        seed_started = time.time()
        processes = []
        print(
            f"[shard-dispatch] {seed_name} START: 126 tasks across "
            f"{args.gpu_count} GPUs",
            flush=True,
        )
        for gpu in range(args.gpu_count):
            log_path = log_root / f"{seed_name}_gpu{gpu}.log"
            summary_path = log_root / f"{seed_name}_gpu{gpu}_summary.json"
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
                str(seed),
                "--worker-id",
                f"{args.chain_name}-{seed_name}-gpu{gpu}",
                "--task-shard-index",
                str(gpu),
                "--task-shard-count",
                str(args.gpu_count),
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
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONUNBUFFERED="1",
            )
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu, process, log_path, summary_path, log_handle))

        while any(process.poll() is None for _, process, _, _, _ in processes):
            counts = seed_output_counts(model_root, seed)
            generated = sum(item["generated_only"] for item in counts.values())
            state = " ".join(
                f"g{gpu}={'run' if process.poll() is None else process.returncode}"
                for gpu, process, _, _, _ in processes
            )
            print(
                f"[shard-dispatch] {time.strftime('%H:%M:%S')} {seed_name} "
                f"generated={generated}/126 {state}",
                flush=True,
            )
            time.sleep(args.poll_sec)

        worker_summaries: dict[str, Any] = {}
        seed_failures = []
        for gpu, process, log_path, summary_path, log_handle in processes:
            log_handle.close()
            if process.returncode:
                seed_failures.append(
                    {"gpu": gpu, "returncode": process.returncode, "log": str(log_path)}
                )
            if summary_path.is_file():
                worker_summaries[str(gpu)] = load_summary(summary_path)

        complete = seed_complete(model_root, seed, items)
        if seed_failures or not complete:
            failure = {
                "seed": seed,
                "worker_failures": seed_failures,
                "counts": seed_output_counts(model_root, seed),
            }
            failures.append(failure)
            plan["status"] = "failed"
            plan["failures"] = failures
            atomic_json_dump(plan_path, plan)
            raise RuntimeError(f"{seed_name} failed: {failure}")

        completed_at = time.time()
        seed_summary = {
            "status": "complete",
            "seed": seed,
            "host": socket.gethostname(),
            "execution": "126 tasks sharded over 8 GPUs",
            "counts": seed_output_counts(model_root, seed),
            "wall_sec": round(completed_at - seed_started, 3),
            "parameters": plan["parameters"],
            "model": {
                "transformer": str((model_dir / "transformer").resolve()),
                "text_encoder": str((model_dir / "text_encoder").resolve()),
                "vae": str((model_dir / "vae").resolve()),
            },
            "worker_summaries": {
                gpu: str(log_root / f"{seed_name}_gpu{gpu}_summary.json")
                for gpu in worker_summaries
            },
            "completed_at_unix": completed_at,
        }
        atomic_json_dump(seed_root / "generation_summary.json", seed_summary)
        atomic_json_dump(seed_root / "_COMPLETE.json", seed_summary)
        plan["seed_summaries"][seed_name] = str(seed_root / "generation_summary.json")
        atomic_json_dump(plan_path, plan)
        print(
            f"[shard-dispatch] {seed_name} COMPLETE wall={seed_summary['wall_sec']:.1f}s",
            flush=True,
        )

    completed_at = time.time()
    plan.update(
        {
            "status": "complete",
            "failures": failures,
            "completed_at_unix": completed_at,
            "wall_sec": round(completed_at - plan["started_at_unix"], 3),
        }
    )
    atomic_json_dump(plan_path, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
