#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml


METRICS = (
    "image_quality",
    "aesthetic_quality",
    "dynamic_degree",
    "flow_score",
    "motion_smoothness",
    "subject_consistency",
    "background_consistency",
    "photometric_smoothness",
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def isolate_local_gpu() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    selected = devices[local_rank] if len(devices) > local_rank else str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    os.environ["LOCAL_RANK"] = "0"


def load_manifest(path: Path, expected_count: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise ValueError(
            f"Manifest coverage mismatch: {len(payload) if isinstance(payload, list) else 'invalid'}"
            f"/{expected_count}"
        )
    request_ids = [row.get("request_id") for row in payload]
    if any(not isinstance(value, str) or not value for value in request_ids):
        raise ValueError("Manifest contains an invalid request_id")
    if len(set(request_ids)) != expected_count:
        raise ValueError("Manifest request IDs are not unique")
    return payload


def build_full_info(
    manifest: list[dict[str, Any]], video_dir: Path, metric: str
) -> list[dict[str, Any]]:
    full_info = []
    for row in manifest:
        video_path = (video_dir / f"{row['request_id']}.mp4").resolve()
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError(video_path)
        full_info.append(
            {
                "dimension": [metric],
                "video_list": [str(video_path)],
                "prompt": row["prompt"],
                "prompt_en": row["prompt"],
            }
        )
    return full_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one public WorldArena metric on WorldArena 1.0 MP4 inputs"
    )
    parser.add_argument("--worldarena-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def result_ids(result: Any) -> set[str]:
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return set()
    details = result[1]
    if not isinstance(details, list):
        return set()
    return {Path(item.get("video_path", "")).stem for item in details}


def main() -> None:
    args = parse_args()
    isolate_local_gpu()
    sys.path.insert(0, str((args.worldarena_root / "video_quality").resolve()))
    from WorldArena import (  # noqa: PLC0415
        _add_normalized_scores,
        compute_aesthetic_quality,
        compute_background_consistency,
        compute_dynamic_degree,
        compute_flow_score,
        compute_imaging_quality,
        compute_motion_smoothness,
        compute_photometric_smoothness,
        compute_subject_consistency,
    )
    from WorldArena.distributed import barrier, dist_init, get_rank  # noqa: PLC0415

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt = config["ckpt"]
    manifest = load_manifest(args.manifest, args.expected_count)
    expected_ids = {row["request_id"] for row in manifest}
    full_info_path = args.output.parent / f".{args.metric}_full_info.json"

    dist_init()
    if get_rank() == 0:
        atomic_write_json(
            full_info_path, build_full_info(manifest, args.video_dir, args.metric)
        )
    barrier()

    evaluators: dict[str, tuple[Callable[..., Any] | None, dict[str, Any]]] = {
        "image_quality": (
            compute_imaging_quality,
            {"model_path": ckpt["image_quality"]["musiq"]},
        ),
        "aesthetic_quality": (
            compute_aesthetic_quality,
            {
                "clip_model": ckpt["aesthetic_quality"]["clip"],
                "aesthetic_head": ckpt["aesthetic_quality"]["aesthetic_head"],
            },
        ),
        "dynamic_degree": (
            compute_dynamic_degree,
            {"model": ckpt["dynamic_degree"]["raft"]},
        ),
        "flow_score": (
            compute_flow_score,
            {"model": ckpt["flow_score"]["raft"]},
        ),
        "motion_smoothness": (
            compute_motion_smoothness,
            {"model": ckpt["motion_smoothness"]["model"]},
        ),
        "subject_consistency": (
            compute_subject_consistency,
            {
                "repo_or_dir": ckpt["subject_consistency"]["repo"],
                "path": ckpt["subject_consistency"]["weight"],
                "model": ckpt["subject_consistency"].get("model", "dino_vitb16"),
                "source": "local",
                "read_frame": False,
                "raft_model": ckpt["subject_consistency"]["raft"],
            },
        ),
        "background_consistency": (
            compute_background_consistency,
            {
                "clip_model": ckpt["background_consistency"]["clip"],
                "raft_model": ckpt["background_consistency"]["raft"],
                "read_frame": False,
            },
        ),
        "photometric_smoothness": (
            compute_photometric_smoothness,
            {
                "config": ckpt["photometric_smoothness"]["cfg"],
                "model": ckpt["photometric_smoothness"]["model"],
            },
        ),
    }
    evaluator, submodules = evaluators[args.metric]
    if evaluator is None:
        raise RuntimeError(f"{args.metric} evaluator is unavailable")
    result = _add_normalized_scores(args.metric, evaluator(str(full_info_path), submodules))
    if get_rank() == 0:
        actual_ids = result_ids(result)
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"{args.metric} coverage mismatch: actual={len(actual_ids)} "
                f"expected={len(expected_ids)} missing={sorted(expected_ids-actual_ids)[:5]}"
            )
        atomic_write_json(args.output, {args.metric: result})
        print(
            f"metric={args.metric} coverage={len(actual_ids)} output={args.output}"
        )
    barrier()

    import torch.distributed as dist

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
