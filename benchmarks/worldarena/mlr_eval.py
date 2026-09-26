#!/usr/bin/env python3
"""Per-shard MLR worker for the WorldArena 1.0 protocol.

The counting stack (GroundingDINO target detections, robot detection and the
gripper-overlap filter) lives in `mlr_occlusion.py`, together with the
deviation/occlusion rules and the named protocol profiles in
`mlr_protocol_profiles.yaml`.  Running with the default profile reproduces the
frozen `worldarena1_mlr_gdino_v2` protocol; pass
`--profile appendix_alg1_sam2_occlusion` (or the individual flags) to apply the
appendix Algorithm 1 rule instead, which additionally needs SAM2.1 for the
occlusion evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PIPELINE_DIR = HERE.parents[1] / "eveworld" / "pipeline"
for directory in (PIPELINE_DIR, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import mlr_occlusion as mlo  # noqa: E402


RULE_FLAGS = (
    "--deviation-mode",
    "--occlusion-rule",
    "--sample-mode",
    "--persistence",
    "--tau-occ",
    "--reliable-logit",
    "--reliable-area-ratio",
    "--presence-logit",
    "--contact-margin-ratio",
    "--object-topk",
    "--object-box-threshold",
    "--robot-prompt",
    "--robot-topk",
    "--robot-box-threshold",
    "--gripper-overlap-threshold",
    "--min-center-distance",
    "--initial-count-source",
    "--sampled-count-source",
    "--device",
    "--gdino-path",
    "--sam2-checkpoint",
    "--sam2-model-config",
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inventory_count(record: dict[str, Any], source: str) -> int:
    """Reference count N_0 for one condition image."""
    if source == "raw":
        return int(record["raw_count"])
    return int(record["count"])


def build_samples(
    records: list[dict[str, Any]],
    source_indices: list[int],
    evidence: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample: dict[str, Any] = {
            "sample_index": index,
            "source_frame": source_indices[index],
            "raw_count": int(record["raw_count"]),
            "count": int(record["count"]),
        }
        if evidence is not None:
            sample["instances"] = evidence[index]["instances"]
            sample["contact"] = evidence[index]["contact"]
        samples.append(sample)
    return samples


def evaluate_video(
    locator: Any,
    video: Path,
    condition_image: Path,
    mover: str,
    settings: dict[str, Any],
    inventory_cache: dict[tuple[str, str], dict[str, Any]],
    predictor: Any = None,
    frames_root: Path | None = None,
    keep_frames: bool = False,
) -> dict[str, Any]:
    cache_key = (str(condition_image), mover)
    if cache_key not in inventory_cache:
        initial = np.asarray(Image.open(condition_image).convert("RGB"))
        inventory_cache[cache_key] = mlo.detect_sample_counts(
            locator, [initial], mover, settings
        )[0]
    initial_count = inventory_count(inventory_cache[cache_key], settings["initial_count_source"])
    if initial_count == 0:
        return {
            "eligible": False,
            "initial_count": 0,
            "counts": [],
            "raw_counts": [],
            "adjusted_counts": [],
            "deviation_flags": [],
            "onset_sample_index": None,
            "onset_source_frame": None,
            "occlusion_exempt_frames": [],
            "mlr_event": False,
            "max_count": 0,
        }

    frames, source_indices = mlo.read_sample_frames(
        Path(video), settings["frame_count"], settings["sample_mode"]
    )
    records = mlo.detect_sample_counts(locator, frames, mover, settings)

    evidence = None
    frames_dir = None
    if settings["occlusion_rule"] != "none":
        if predictor is None:
            raise ValueError("a SAM2 predictor is required when occlusion_rule != 'none'")
        frames_dir = Path(
            tempfile.mkdtemp(
                prefix="mlr_eval_", dir=None if frames_root is None else str(frames_root)
            )
        )
        evidence, _ = mlo.collect_sam2_evidence(
            predictor, frames, records, initial_count, frames_dir, settings
        )

    try:
        evaluation = mlo.evaluate_samples(
            build_samples(records, source_indices, evidence),
            initial_count,
            deviation_mode=settings["deviation_mode"],
            occlusion_rule=settings["occlusion_rule"],
            tau_occ=settings["tau_occ"],
            persistence=settings["persistence"],
            reliable_logit=settings["reliable_logit"],
            reliable_area_ratio=settings["reliable_area_ratio"],
            presence_logit=settings["presence_logit"],
            count_source=settings["sampled_count_source"],
        )
        counts = [int(record["count"]) for record in records]
        result: dict[str, Any] = {
            "eligible": evaluation["eligible"],
            "initial_count": initial_count,
            "counts": counts,
            "raw_counts": [int(record["raw_count"]) for record in records],
            "adjusted_counts": evaluation["adjusted_counts"],
            "deviation_flags": evaluation["deviation_flags"],
            "onset_sample_index": evaluation["onset_sample_index"],
            "onset_source_frame": evaluation["onset_source_frame"],
            "occlusion_exempt_frames": evaluation["under_count_exempt_frames"],
            "under_count_visible_frames": evaluation["under_count_visible_frames"],
            "mlr_event": evaluation["event"],
            "max_count": max(counts) if counts else 0,
        }
    finally:
        if frames_dir is not None and not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

    if settings.get("compare_all"):
        result["comparison"] = mlo.compare_rules(
            build_samples(records, source_indices, evidence),
            initial_count,
            tau_occ=settings["tau_occ"],
            persistence=settings["persistence"],
            reliable_logit=settings["reliable_logit"],
            reliable_area_ratio=settings["reliable_area_ratio"],
            presence_logit=settings["presence_logit"],
            count_source=settings["sampled_count_source"],
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WorldArena 1.0 MLR on one shard")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument(
        "--profile",
        default="worldarena1_mlr_gdino_v2",
        help="named protocol profile from mlr_protocol_profiles.yaml (default: the frozen "
        "WorldArena 1.0 protocol)",
    )
    parser.add_argument(
        "--profiles-file", type=Path, default=mlo.PROFILES_FILE, help="profile YAML path"
    )
    parser.add_argument("--deviation-mode", choices=mlo.DEVIATION_MODES)
    parser.add_argument("--occlusion-rule", choices=mlo.OCCLUSION_RULES)
    parser.add_argument("--sample-mode", choices=mlo.SAMPLE_MODES)
    parser.add_argument("--persistence", type=int)
    parser.add_argument("--tau-occ", type=float)
    parser.add_argument("--reliable-logit", type=float)
    parser.add_argument("--reliable-area-ratio", type=float)
    parser.add_argument("--presence-logit", type=float)
    parser.add_argument("--contact-margin-ratio", type=float)
    parser.add_argument("--object-topk", type=int)
    parser.add_argument("--object-box-threshold", type=float)
    parser.add_argument("--robot-prompt")
    parser.add_argument("--robot-topk", type=int)
    parser.add_argument("--robot-box-threshold", type=float)
    parser.add_argument("--gripper-overlap-threshold", type=float)
    parser.add_argument("--min-center-distance", type=float)
    parser.add_argument("--initial-count-source", choices=mlo.INITIAL_COUNT_SOURCES)
    parser.add_argument("--sampled-count-source", choices=mlo.SAMPLED_COUNT_SOURCES)
    parser.add_argument("--device")
    parser.add_argument("--gdino-path")
    parser.add_argument("--sam2-checkpoint")
    parser.add_argument("--sam2-model-config")
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="also record all deviation-mode x occlusion-rule combinations per clip",
    )
    parser.add_argument(
        "--sam2-frames-root",
        type=Path,
        help="write the sampled frame directories here instead of the system temp dir",
    )
    parser.add_argument(
        "--keep-sam2-frames", action="store_true", help="keep the sampled frame directories"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = mlo.resolve_settings(args)
    settings["profile_name"] = args.profile
    settings["compare_all"] = bool(args.compare_all)
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    jobs = jobs[args.shard_index :: args.num_shards]
    locator = mlo.make_locator(settings["device"], settings["gdino_path"])
    predictor = None
    if settings["occlusion_rule"] != "none":
        predictor = mlo.build_sam2_predictor(settings)
    if args.sam2_frames_root is not None:
        args.sam2_frames_root.mkdir(parents=True, exist_ok=True)

    inventory_cache: dict[tuple[str, str], dict[str, Any]] = {}
    records = []
    for index, job in enumerate(jobs, start=1):
        record = {
            "model": job["model"],
            "request_id": job["request_id"],
            "mover": job["mover"],
            "video": job["video"],
            "condition_image": job["condition_image"],
        }
        try:
            record.update(
                evaluate_video(
                    locator,
                    Path(job["video"]),
                    Path(job["condition_image"]),
                    job["mover"],
                    settings,
                    inventory_cache,
                    predictor=predictor,
                    frames_root=args.sam2_frames_root,
                    keep_frames=args.keep_sam2_frames,
                )
            )
            record["error"] = None
        except Exception as error:  # Preserve item-level failures for the coverage gate.
            record.update(
                {
                    "eligible": False,
                    "initial_count": None,
                    "counts": [],
                    "raw_counts": [],
                    "adjusted_counts": [],
                    "deviation_flags": [],
                    "onset_sample_index": None,
                    "occlusion_exempt_frames": [],
                    "mlr_event": False,
                    "max_count": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        records.append(record)
        if index % 20 == 0 or index == len(jobs):
            print(f"shard={args.shard_index} progress={index}/{len(jobs)}", flush=True)
    metadata = mlo.protocol_metadata(settings, None)
    metadata.pop("initial_count", None)
    metadata.update(
        {
            "protocol": settings["profile_name"] or "worldarena1_mlr_gdino_v2",
            "frame_count": settings["frame_count"],
            "object_threshold": settings["object_box_threshold"],
            "gripper_threshold": settings["robot_box_threshold"],
            "gripper_overlap_threshold": settings["gripper_overlap_threshold"],
            "minimum_center_distance": settings["min_center_distance"],
            "consecutive_frames": settings["persistence"],
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
    )
    atomic_write_json(args.output, {"metadata": metadata, "records": records})


if __name__ == "__main__":
    main()
