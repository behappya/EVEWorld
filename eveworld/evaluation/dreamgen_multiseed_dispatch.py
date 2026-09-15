#!/usr/bin/env python3
"""Run one DreamGenBench item across seeds and models on one eight-GPU node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from eveworld.evaluation.x9_eval175_dispatch import MODELS


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_source_item(path: Path, request_id: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"source JSON must be a list: {path}")
    matches = [item for item in data if item.get("request_id") == request_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {request_id!r} item in {path}, found {len(matches)}")
    item = matches[0]
    if not item.get("prompt") or not item.get("image"):
        raise ValueError(f"source item {request_id!r} is missing prompt or image")
    image_path = Path(item["image"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return item


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_video(path: Path, frames: int, width: int, height: int, fps: int) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    decoded = 0
    decoded_hash = hashlib.sha256()
    actual_width = None
    actual_height = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        actual_height, actual_width = frame.shape[:2]
        decoded_hash.update(frame.tobytes())
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if decoded != frames:
        raise RuntimeError(f"{path}: decoded {decoded} frames, expected {frames}")
    if (actual_width, actual_height) != (width, height):
        raise RuntimeError(
            f"{path}: resolution {(actual_width, actual_height)}, expected {(width, height)}"
        )
    if abs(actual_fps - fps) > 0.05:
        raise RuntimeError(f"{path}: fps {actual_fps}, expected {fps}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "frames": decoded,
        "width": actual_width,
        "height": actual_height,
        "fps": actual_fps,
        "sha256": sha256_file(path),
        "decoded_frame_sha256": decoded_hash.hexdigest(),
    }


def audit_model_outputs(
    model_root: Path, seeds: list[int], frames: int, width: int, height: int, fps: int
) -> dict:
    expected_names = {f"seed{seed}.mp4" for seed in seeds}
    generated_dir = model_root / "generated_only"
    side_by_side_dir = model_root / "side_by_side"
    generated_names = {path.name for path in generated_dir.glob("*.mp4")}
    side_by_side_names = {path.name for path in side_by_side_dir.glob("*.mp4")}
    if generated_names != expected_names:
        raise RuntimeError(
            f"{generated_dir}: expected {sorted(expected_names)}, found {sorted(generated_names)}"
        )
    if side_by_side_names != expected_names:
        raise RuntimeError(
            f"{side_by_side_dir}: expected {sorted(expected_names)}, found {sorted(side_by_side_names)}"
        )

    generated = {
        str(seed): audit_video(
            generated_dir / f"seed{seed}.mp4", frames=frames, width=width, height=height, fps=fps
        )
        for seed in seeds
    }
    side_by_side = {
        str(seed): audit_video(
            side_by_side_dir / f"seed{seed}.mp4",
            frames=frames,
            width=((width * 2 + 2 + 15) // 16) * 16,
            height=height,
            fps=fps,
        )
        for seed in seeds
    }
    return {
        "complete": True,
        "seed_count": len(seeds),
        "generated_only": generated,
        "side_by_side": side_by_side,
    }


def run_model(args, model: str, model_dir: Path, input_json: Path, task_root: Path) -> dict:
    model_root = task_root / model
    log_root = model_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    seed_shards = [args.seeds[gpu :: args.gpu_count] for gpu in range(args.gpu_count)]
    processes = []
    started_at = time.time()
    for gpu, seeds in enumerate(seed_shards):
        if not seeds:
            continue
        log_path = log_root / f"gpu{gpu}.log"
        summary_path = log_root / f"gpu{gpu}_summary.json"
        command = [
            args.python,
            str(args.worker_script),
            "--input-json",
            str(input_json),
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
            f"gpu{gpu}",
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
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT
        )
        processes.append((gpu, seeds, process, log_path, log_handle))
        print(f"[dreamgen-dispatch] {model} gpu={gpu} seeds={seeds}", flush=True)

    while any(item[2].poll() is None for item in processes):
        count = len(list((model_root / "generated_only").glob("*.mp4")))
        status = " ".join(
            f"g{gpu}={'run' if process.poll() is None else process.returncode}"
            for gpu, _, process, _, _ in processes
        )
        print(
            f"[dreamgen-dispatch] {time.strftime('%H:%M:%S')} model={model} "
            f"generated={count}/{len(args.seeds)} {status}",
            flush=True,
        )
        time.sleep(args.poll_sec)

    failures = []
    worker_summaries = {}
    for gpu, seeds, process, log_path, log_handle in processes:
        log_handle.close()
        if process.returncode:
            failures.append(f"gpu={gpu},seeds={seeds},rc={process.returncode},log={log_path}")
        summary_path = log_root / f"gpu{gpu}_summary.json"
        if summary_path.is_file():
            worker_summaries[str(gpu)] = json.loads(summary_path.read_text(encoding="utf-8"))
    if failures:
        raise RuntimeError(f"{model} worker failures: {'; '.join(failures)}")

    audit = audit_model_outputs(
        model_root,
        args.seeds,
        frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    summary = {
        "model": model,
        "model_dir": str(model_dir),
        "transformer": str((model_dir / "transformer").resolve()),
        "text_encoder": str((model_dir / "text_encoder").resolve()),
        "vae": str((model_dir / "vae").resolve()),
        "seeds": args.seeds,
        "seed_shards": {str(gpu): seeds for gpu, seeds, _, _, _ in processes},
        "wall_sec": round(time.time() - started_at, 3),
        "workers": worker_summaries,
        "audit": audit,
    }
    atomic_json_dump(model_root / "generation_summary.json", summary)
    print(
        f"[dreamgen-dispatch] model={model} COMPLETE seeds={len(args.seeds)} "
        f"wall={summary['wall_sec']:.1f}s",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=Path(__file__).with_name("dreamgen_multiseed_worker.py"),
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

    if "/" in args.task_id or args.task_id in (".", ".."):
        raise ValueError("task-id must be a single path-safe component")
    if args.gpu_count != 8:
        raise ValueError("this dispatcher requires exactly one eight-GPU node")
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must be unique")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not args.worker_script.is_file():
        raise FileNotFoundError(args.worker_script)
    source_item = load_source_item(args.source_json, args.request_id)
    model_dirs = {model: validate_model(model) for model in args.models}
    if args.source_video is not None and not args.source_video.is_file():
        raise FileNotFoundError(args.source_video)

    execution_plan = {
        "task_id": args.task_id,
        "request_id": args.request_id,
        "prompt": source_item["prompt"],
        "condition_image": source_item["image"],
        "source_json": str(args.source_json),
        "source_video": str(args.source_video) if args.source_video else None,
        "models_in_order": args.models,
        "model_dirs": {model: str(path) for model, path in model_dirs.items()},
        "seeds": args.seeds,
        "seed_shards": {
            str(gpu): args.seeds[gpu :: args.gpu_count] for gpu in range(args.gpu_count)
        },
        "gpu_count": args.gpu_count,
        "parameters": {
            "num_inference_steps": args.num_inference_steps,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
            "eag_weight": 0,
        },
        "output": str(args.output_root / args.task_id),
    }
    if args.dry_run:
        print(json.dumps(execution_plan, ensure_ascii=False, indent=2))
        return

    task_root = args.output_root / args.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    input_json = task_root / "input.json"
    atomic_json_dump(input_json, [source_item])
    metadata = {
        **execution_plan,
        "status": "running",
        "started_at_unix": time.time(),
        "completed_models": [],
        "model_summaries": {},
    }
    task_json = task_root / "task.json"
    atomic_json_dump(task_json, metadata)

    try:
        for model in args.models:
            print(f"[dreamgen-dispatch] START model={model}", flush=True)
            summary = run_model(args, model, model_dirs[model], input_json, task_root)
            metadata["completed_models"].append(model)
            metadata["model_summaries"][model] = str(
                task_root / model / "generation_summary.json"
            )
            atomic_json_dump(task_json, metadata)
        metadata["status"] = "complete"
        metadata["completed_at_unix"] = time.time()
        metadata["wall_sec"] = round(metadata["completed_at_unix"] - metadata["started_at_unix"], 3)
        atomic_json_dump(task_json, metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failed_at_unix"] = time.time()
        metadata["error"] = str(error)
        metadata["traceback"] = traceback.format_exc()
        atomic_json_dump(task_json, metadata)
        raise


if __name__ == "__main__":
    main()
