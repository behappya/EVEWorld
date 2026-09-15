#!/usr/bin/env python3
"""Run six checkpoint CFG blocks serially; each block uses all eight GPUs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def cell_complete(root: Path, step: int, cfg: float, expected: int = 126) -> bool:
    tag = f"cfg_{cfg:g}".replace(".", "p")
    d = root / f"step_{step:03d}" / tag / "generated_only"
    return len([p for p in d.glob("*.mp4") if p.is_file() and p.stat().st_size > 10_000]) == expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--steps-list", nargs="+", type=int, default=[50, 100, 150, 200, 250, 300])
    ap.add_argument("--cfg-values", nargs="+", type=float, default=[1.0, 2.5, 5.0, 7.5])
    ap.add_argument("--gpu-count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--inference-steps", type=int, default=30)
    ap.add_argument("--data-pythonpath", default="")
    args = ap.parse_args()
    if args.gpu_count != 8:
        raise ValueError("the runner requires exactly eight GPUs")
    if len(set(args.steps_list)) != len(args.steps_list):
        raise ValueError("duplicate training steps")
    if len(args.cfg_values) != 4:
        raise ValueError("expected exactly four CFG values")
    if args.data_root.is_dir():
        rows = []
        for split in ("env", "object", "behavior"):
            path = args.data_root / f"eval175_gr1_{split}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    elif args.data_root.is_file():
        rows = json.loads(args.data_root.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(args.data_root)
    if len(rows) != 126:
        raise ValueError(f"expected 126 rows, got {len(rows)}")
    ids = [str(row.get("request_id") or row.get("id") or "") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != 126:
        raise ValueError("DreamGen manifest request ids must be present and unique")
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "steps": args.steps_list, "cfg_values": args.cfg_values,
        "seed": args.seed, "inference_steps": args.inference_steps,
        "rows": len(rows), "status": "running", "started_at": time.time(),
    }
    (args.out_root / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    worker = Path(__file__).with_name("cfg_grid_worker.py")
    for step in args.steps_list:
        model_dir = args.model_root / f"step_{step:03d}"
        required = [model_dir / "transformer" / "config.json",
                    model_dir / "text_encoder" / "config.json",
                    model_dir / "vae" / "config.json"]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"step {step} missing: {missing}")
        if all(cell_complete(args.out_root, step, c) for c in args.cfg_values):
            print(f"STEP {step} COMPLETE; skip", flush=True)
            continue
        procs = []
        print(f"STEP {step} START; loading one model per GPU", flush=True)
        for gpu in range(args.gpu_count):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            cmd = [sys.executable, str(worker), "--model-dir", str(model_dir),
                   "--data-root", str(args.data_root), "--out-root", str(args.out_root),
                   "--training-step", str(step), "--cfg-values", *map(str, args.cfg_values),
                   "--worker-id", str(gpu), "--worker-count", str(args.gpu_count),
                   "--seed", str(args.seed), "--steps", str(args.inference_steps),
                   "--pythonpath", args.data_pythonpath]
            log = (args.out_root / f"step_{step:03d}_gpu{gpu}.log").open("a", encoding="utf-8")
            procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        rc = 0
        for p, log in procs:
            value = p.wait()
            rc = rc or value
            log.close()
        if rc:
            for p, _ in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM)
            raise SystemExit(f"step {step} worker failure rc={rc}")
        bad = [c for c in args.cfg_values if not cell_complete(args.out_root, step, c)]
        if bad:
            raise RuntimeError(f"step {step} incomplete cells: {bad}")
        print(f"STEP {step} COMPLETE", flush=True)
    manifest["status"] = "complete"
    manifest["finished_at"] = time.time()
    (args.out_root / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
