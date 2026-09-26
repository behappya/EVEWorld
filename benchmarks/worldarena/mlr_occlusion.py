#!/usr/bin/env python3
"""MLR evaluation with pluggable deviation and occlusion rules.

The appendix ("Model Laziness and MLR", Algorithm 1) defines MLR over sampled
timestamps: a clip is a deviation event when the adjusted instance count differs
from the initial inventory N_0 at two consecutive sampled timestamps; MLR is the
event rate over clips with an eligible D+ (N_0 >= 1) and coverage is the share of
clips that are eligible.  Two independent choices decide how the adjusted count
is built, and both are exposed here so a user can reproduce either the frozen
WorldArena 1.0 protocol or the appendix rule:

deviation-mode
  over_only   only N_t > N_0 is evidence (frozen WorldArena 1.0 protocol)
  symmetric   N_t != N_0 is evidence (appendix Algorithm 1)

occlusion-rule (decides whether an under-count N_t < N_0 is exempted)
  none              never exempt an under-count
  paper_overlap     exempt when at least `missing` targets are visibility degraded
                    and their reference region overlaps the robot mask by at
                    least tau_occ (appendix eq. occ_overlap / mlr_adjusted_count)
  sam_presence      exempt when at least `missing` targets have a SAM2 presence
                    logit below presence_logit
  contact_recovery  exempt a whole under-count run when the target touches the
                    robot within the run and the run ends in a recovery

sampled-count-source (the per-timestamp count N_t both rules read)
  filtered   gripper-filtered target count (after spatial merging)
  raw        raw target-detection count, before merging and filtering

Both the over-count and the under-count side read this same N_t; with
gripper-overlap-threshold 1.0 the filtered count is exactly the spatially
merged detection count.

Trace schema (input for --trace, output of the video path):

{
  "initial_count": 3,
  "mover": "toy car",
  "samples": [
    {"sample_index": 0, "source_frame": 0, "raw_count": 3, "count": 3,
     "contact": false,
     "instances": [
       {"instance_id": 1, "logit": 12.4, "area": 5120, "initial_area": 5120,
        "overlap": 0.0, "contact": false}
     ]}
  ]
}

A bare list of integers, or {"counts": [...]}, is also accepted; each count is
then used as both the raw and the filtered count.

Examples:

  # Rule choices on a stored trace; no model weights needed
  python benchmarks/worldarena/mlr_occlusion.py --trace trace.json \
      --deviation-mode symmetric --occlusion-rule paper_overlap
  python benchmarks/worldarena/mlr_occlusion.py --trace trace.json --compare-all

  # Named protocol profile (see mlr_protocol_profiles.yaml)
  python benchmarks/worldarena/mlr_occlusion.py --trace trace.json \
      --profile appendix_alg1_sam2_occlusion

  # End-to-end on one video (GroundingDINO, plus SAM2.1 when occlusion is used)
  python benchmarks/worldarena/mlr_occlusion.py --video clip.mp4 --mover "toy car" \
      --condition-image cond.png --occlusion-rule paper_overlap --output out.json

  # Logic checks without any model or video
  python benchmarks/worldarena/mlr_occlusion.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "eveworld" / "pipeline"
PROFILES_FILE = Path(__file__).with_name("mlr_protocol_profiles.yaml")

DEVIATION_MODES = ("over_only", "symmetric")
OCCLUSION_RULES = ("none", "paper_overlap", "sam_presence", "contact_recovery")
SAMPLE_MODES = ("round", "linspace")
INITIAL_COUNT_SOURCES = ("filtered", "raw")
SAMPLED_COUNT_SOURCES = ("filtered", "raw")

DEFAULTS: dict[str, Any] = {
    "deviation_mode": "symmetric",
    "occlusion_rule": "paper_overlap",
    "sample_mode": "round",
    "frame_count": 24,
    "persistence": 2,
    "tau_occ": 0.15,
    "reliable_logit": 0.0,
    "reliable_area_ratio": 0.50,
    "presence_logit": 0.0,
    "contact_margin_ratio": 0.035,
    "object_topk": 6,
    "object_box_threshold": 0.35,
    "robot_prompt": "robot gripper",
    "robot_topk": 3,
    "robot_box_threshold": 0.15,
    "gripper_overlap_threshold": 0.35,
    "min_center_distance": 60.0,
    "initial_count_source": "filtered",
    "sampled_count_source": "filtered",
    "device": "cuda",
    "gdino_path": None,
    "sam2_checkpoint": None,
    "sam2_model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
}
PROFILE_KEYS = tuple(DEFAULTS)


# --------------------------------------------------------------------------- #
# rules: sampling, persistence, adjusted counts
# --------------------------------------------------------------------------- #


def sample_indices(frame_count: int, sample_count: int = 24, mode: str = "round") -> list[int]:
    """Sampled source-frame indices.

    "round" rounds i*(F-1)/(n-1) to the nearest frame (appendix protocol);
    "linspace" keeps the legacy truncated-linspace behaviour.
    """
    if mode not in SAMPLE_MODES:
        raise ValueError(f"Unknown sample mode: {mode!r}; expected one of {SAMPLE_MODES}")
    if frame_count <= 0:
        return []
    if sample_count <= 1:
        return [0]
    last = frame_count - 1
    if mode == "linspace":
        try:
            import numpy as np
        except ImportError:
            return [int(i * last / (sample_count - 1)) for i in range(sample_count)]
        return [int(value) for value in np.linspace(0, last, sample_count).astype(int)]
    return [int(math.floor(i * last / (sample_count - 1) + 0.5)) for i in range(sample_count)]


def longest_run(flags: list[bool]) -> int:
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def has_persistent_run(flags: list[bool], length: int) -> bool:
    if length <= 1:
        return any(flags)
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        if run >= length:
            return True
    return False


def first_persistent_onset(flags: list[bool], length: int) -> int | None:
    """Sample index where the first qualifying run starts, or None."""
    if length <= 1:
        for index, flag in enumerate(flags):
            if flag:
                return index
        return None
    run = 0
    for index, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= length:
            return index - length + 1
    return None


def _normalize_instance(raw: Any, index: int) -> dict[str, Any]:
    source = dict(raw) if isinstance(raw, dict) else {"logit": raw}
    logit = source.get("logit", source.get("presence_logit"))
    area = source.get("area", source.get("mask_area"))
    initial_area = source.get("initial_area")
    overlap = source.get("overlap", source.get("overlap_ratio", 0.0))
    return {
        "instance_id": int(source.get("instance_id", index + 1)),
        "logit": math.inf if logit is None else float(logit),
        "area": None if area is None else int(area),
        "initial_area": None if initial_area in (None, 0) else int(initial_area),
        "overlap": 0.0 if overlap is None else float(overlap),
        "contact": bool(source.get("contact", False)),
    }


def normalize_sample(raw: Any, index: int) -> dict[str, Any]:
    if isinstance(raw, dict):
        sample = dict(raw)
    elif isinstance(raw, (int, float)):
        sample = {"count": int(raw)}
    else:
        raise TypeError(f"Unsupported sample entry: {raw!r}")
    count = sample.get("count", sample.get("raw_count"))
    raw_count = sample.get("raw_count", sample.get("count"))
    if count is None or raw_count is None:
        raise ValueError(f"Sample {index} needs a count or raw_count: {sample!r}")
    instances = [
        _normalize_instance(entry, position)
        for position, entry in enumerate(sample.get("instances") or [])
    ]
    contact = sample.get("contact")
    return {
        "sample_index": int(sample.get("sample_index", index)),
        "source_frame": int(sample.get("source_frame", sample.get("sample_index", index))),
        "raw_count": int(raw_count),
        "count": int(count),
        "instances": instances,
        "contact": bool(contact) if contact is not None else any(i["contact"] for i in instances),
    }


def instance_is_reliable(
    instance: dict[str, Any], reliable_logit: float, reliable_area_ratio: float
) -> bool:
    """A tracked instance still carries trustworthy visibility evidence."""
    if instance["logit"] < reliable_logit:
        return False
    area, initial_area = instance["area"], instance["initial_area"]
    if area is None or initial_area is None:
        return True
    return area >= reliable_area_ratio * initial_area


def _basis_count(sample: dict[str, Any], count_source: str) -> int:
    """The per-timestamp count N_t that the over/under rules read."""
    if count_source not in SAMPLED_COUNT_SOURCES:
        raise ValueError(f"Unknown sampled count source: {count_source!r}")
    return int(sample["raw_count"] if count_source == "raw" else sample["count"])


def _contact_recovery_flags(
    samples: list[dict[str, Any]],
    initial_count: int,
    flags: list[bool],
    reasons: list[str],
    *,
    count_source: str = "filtered",
) -> tuple[list[bool], list[str]]:
    """Extend a contact-supported exemption through a recovering under-count run."""
    index = 0
    while index < len(samples):
        if _basis_count(samples[index], count_source) >= initial_count:
            index += 1
            continue
        start = index
        while index < len(samples):
            if _basis_count(samples[index], count_source) >= initial_count:
                break
            index += 1
        if index >= len(samples):
            for position in range(start, index):
                reasons[position] = "under_count_run_without_recovery"
            break
        contact = any(sample["contact"] for sample in samples[start:index])
        for position in range(start, index):
            if contact:
                flags[position] = True
                reasons[position] = "contact_then_recovery"
            else:
                reasons[position] = "under_count_run_without_contact"
    return flags, reasons


def exemption_flags(
    samples: list[dict[str, Any]],
    initial_count: int,
    *,
    occlusion_rule: str = "paper_overlap",
    tau_occ: float = 0.15,
    reliable_logit: float = 0.0,
    reliable_area_ratio: float = 0.50,
    presence_logit: float = 0.0,
    count_source: str = "filtered",
) -> tuple[list[bool], list[str]]:
    """Per-sample under-count exemptions and the reason behind each decision."""
    if occlusion_rule not in OCCLUSION_RULES:
        raise ValueError(f"Unknown occlusion rule: {occlusion_rule!r}")
    if count_source not in SAMPLED_COUNT_SOURCES:
        raise ValueError(f"Unknown sampled count source: {count_source!r}")
    flags = [False] * len(samples)
    reasons = ["not_an_under_count"] * len(samples)
    for index, sample in enumerate(samples):
        missing = max(0, initial_count - _basis_count(sample, count_source))
        if missing <= 0:
            continue
        if occlusion_rule == "none":
            reasons[index] = "under_count_rule_none"
            continue
        if occlusion_rule == "paper_overlap":
            occluded = sum(
                1
                for instance in sample["instances"]
                if not instance_is_reliable(instance, reliable_logit, reliable_area_ratio)
                and instance["overlap"] >= tau_occ
            )
            if occluded >= missing:
                flags[index] = True
                reasons[index] = f"occluded_targets={occluded}>={missing}"
            else:
                reasons[index] = f"occluded_targets={occluded}<{missing}"
            continue
        if occlusion_rule == "sam_presence":
            low_presence = sum(
                1 for instance in sample["instances"] if instance["logit"] < presence_logit
            )
            if low_presence >= missing:
                flags[index] = True
                reasons[index] = f"low_presence={low_presence}>={missing}"
            else:
                reasons[index] = f"low_presence={low_presence}<{missing}"
            continue
        reasons[index] = "pending_contact_recovery"
    if occlusion_rule == "contact_recovery":
        flags, reasons = _contact_recovery_flags(
            samples, initial_count, flags, reasons, count_source=count_source
        )
    return flags, reasons


def evaluate_samples(
    samples: list[Any],
    initial_count: int,
    *,
    deviation_mode: str = "symmetric",
    occlusion_rule: str = "paper_overlap",
    tau_occ: float = 0.15,
    persistence: int = 2,
    reliable_logit: float = 0.0,
    reliable_area_ratio: float = 0.50,
    presence_logit: float = 0.0,
    count_source: str = "filtered",
) -> dict[str, Any]:
    """Algorithm 1 over an ordered list of samples for one clip."""
    if deviation_mode not in DEVIATION_MODES:
        raise ValueError(f"Unknown deviation mode: {deviation_mode!r}")
    if occlusion_rule not in OCCLUSION_RULES:
        raise ValueError(f"Unknown occlusion rule: {occlusion_rule!r}")
    if count_source not in SAMPLED_COUNT_SOURCES:
        raise ValueError(f"Unknown sampled count source: {count_source!r}")
    if persistence < 1:
        raise ValueError("persistence must be >= 1")

    summary = {
        "initial_count": initial_count,
        "deviation_mode": deviation_mode,
        "occlusion_rule": occlusion_rule,
        "persistence": persistence,
        "tau_occ": tau_occ,
        "sampled_count_source": count_source,
        "sample_count": len(samples),
    }
    if initial_count is None or initial_count <= 0:
        return {
            **summary,
            "eligible": False,
            "event": False,
            "onset_sample_index": None,
            "onset_source_frame": None,
            "deviation_flags": [],
            "adjusted_counts": [],
            "max_run": 0,
            "over_count_frames": [],
            "under_count_exempt_frames": [],
            "under_count_visible_frames": [],
            "rows": [],
        }

    normalized = [normalize_sample(entry, index) for index, entry in enumerate(samples)]
    exempt, reasons = exemption_flags(
        normalized,
        initial_count,
        occlusion_rule=occlusion_rule,
        tau_occ=tau_occ,
        reliable_logit=reliable_logit,
        reliable_area_ratio=reliable_area_ratio,
        presence_logit=presence_logit,
        count_source=count_source,
    )

    rows = []
    for index, sample in enumerate(normalized):
        basis = _basis_count(sample, count_source)
        over = basis > initial_count
        under = basis < initial_count
        missing = max(0, initial_count - basis)
        if over:
            adjusted = basis
            state, note = "over_count", "over_count_is_evidence"
        elif under and deviation_mode == "over_only":
            adjusted = initial_count
            state, note = "under_count_ignored", "over_only_mode_ignores_under_counts"
        elif under and exempt[index]:
            adjusted = initial_count
            state, note = "under_count_exempt", reasons[index]
        elif under:
            adjusted = basis
            state, note = "under_count", reasons[index]
        else:
            adjusted = initial_count
            state, note = "conserved", "count_matches_inventory"
        rows.append(
            {
                "sample_index": sample["sample_index"],
                "source_frame": sample["source_frame"],
                "raw_count": sample["raw_count"],
                "count": sample["count"],
                "instances": sample["instances"],
                "contact": sample["contact"],
                "missing_count": missing,
                "adjusted_count": adjusted,
                "state": state,
                "reason": note,
                "deviation": adjusted != initial_count,
            }
        )

    flags = [row["deviation"] for row in rows]
    onset = first_persistent_onset(flags, persistence)
    return {
        **summary,
        "eligible": True,
        "event": has_persistent_run(flags, persistence),
        "onset_sample_index": onset,
        "onset_source_frame": None if onset is None else rows[onset]["source_frame"],
        "deviation_flags": flags,
        "adjusted_counts": [row["adjusted_count"] for row in rows],
        "max_run": longest_run(flags),
        "over_count_frames": [row["sample_index"] for row in rows if row["state"] == "over_count"],
        "under_count_exempt_frames": [
            row["sample_index"] for row in rows if row["state"] == "under_count_exempt"
        ],
        "under_count_visible_frames": [
            row["sample_index"]
            for row in rows
            if row["state"] in {"under_count", "under_count_ignored"}
        ],
        "rows": rows,
    }


def compare_rules(samples: list[Any], initial_count: int, **options: Any) -> dict[str, Any]:
    """Evaluate one clip under every deviation mode x occlusion rule combination."""
    shared = {
        key: options[key]
        for key in (
            "tau_occ",
            "persistence",
            "reliable_logit",
            "reliable_area_ratio",
            "presence_logit",
            "count_source",
        )
        if key in options
    }
    comparison = {}
    for deviation_mode in DEVIATION_MODES:
        for occlusion_rule in OCCLUSION_RULES:
            result = evaluate_samples(
                samples,
                initial_count,
                deviation_mode=deviation_mode,
                occlusion_rule=occlusion_rule,
                **shared,
            )
            comparison[f"{deviation_mode}|{occlusion_rule}"] = {
                "event": result["event"],
                "mlr_percent": summarize([result])["mlr_percent"],
                "onset_sample_index": result["onset_sample_index"],
                "adjusted_counts": result["adjusted_counts"],
                "deviation_flags": result["deviation_flags"],
                "max_run": result["max_run"],
            }
    return comparison


def summarize(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """MLR and coverage over a set of clip evaluations."""
    total = len(evaluations)
    eligible = [entry for entry in evaluations if entry.get("eligible")]
    events = [entry for entry in eligible if entry.get("event")]
    count = len(eligible)
    return {
        "videos": total,
        "eligible_videos": count,
        "events": len(events),
        "mlr_percent": 100.0 * len(events) / count if count else None,
        "coverage_percent": 100.0 * count / total if total else None,
        "event_onsets": [entry.get("onset_sample_index") for entry in events],
    }


# --------------------------------------------------------------------------- #
# trace input
# --------------------------------------------------------------------------- #


def load_trace(path: Path) -> tuple[int | None, list[Any], dict[str, Any]]:
    """Read a trace file and return (initial_count, sample entries, metadata)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return None, payload, {}
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported trace payload: {type(payload).__name__}")
    if "samples" in payload:
        entries = payload["samples"]
    elif "counts" in payload:
        entries = payload["counts"]
    elif "records" in payload:
        entries = payload["records"]
    else:
        raise ValueError("Trace needs one of 'samples', 'counts' or 'records'")
    metadata = {key: value for key, value in payload.items() if key not in {"samples", "counts", "records"}}
    initial_count = payload.get("initial_count")
    if "counts" in payload and payload.get("counts"):
        first = payload["counts"][0]
        if isinstance(first, dict) and initial_count is None:
            initial_count = first.get("initial_count")
    return initial_count, entries, metadata


# --------------------------------------------------------------------------- #
# runtime: GroundingDINO counts and SAM2.1 occlusion evidence
# --------------------------------------------------------------------------- #


def load_profile(path: Path, name: str) -> dict[str, Any]:
    """Named protocol profile from the profiles YAML file."""
    try:
        import yaml
    except ImportError as error:
        raise SystemExit(
            f"Reading the protocol profiles ({path}) needs PyYAML: {error}. "
            "Install `pyyaml`, or pass the rule flags directly instead of --profile."
        )

    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Profiles file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles", payload)
    if name not in profiles:
        known = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else ""
        raise SystemExit(f"Unknown profile {name!r}. Available: {known}")
    profile = profiles[name] or {}
    if not isinstance(profile, dict):
        raise SystemExit(f"Profile {name!r} must map option names to values")
    unknown = sorted(key for key in profile if key not in PROFILE_KEYS and key != "description")
    if unknown:
        raise SystemExit(
            f"Profile {name!r} sets unknown option(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(PROFILE_KEYS))}"
        )
    return {key: value for key, value in profile.items() if key in PROFILE_KEYS}


def _available_profiles(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles", payload)
    return profiles if isinstance(profiles, dict) else {}


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Fill unset options from the selected profile, then from DEFAULTS."""
    profile = load_profile(args.profiles_file, args.profile) if args.profile else {}
    settings = {}
    for key, default in DEFAULTS.items():
        value = getattr(args, key, None)
        if value is None:
            value = profile.get(key, default)
        settings[key] = value
    if settings["deviation_mode"] not in DEVIATION_MODES:
        raise SystemExit(f"Unknown deviation mode: {settings['deviation_mode']!r}")
    if settings["occlusion_rule"] not in OCCLUSION_RULES:
        raise SystemExit(f"Unknown occlusion rule: {settings['occlusion_rule']!r}")
    if settings["sample_mode"] not in SAMPLE_MODES:
        raise SystemExit(f"Unknown sample mode: {settings['sample_mode']!r}")
    if settings["initial_count_source"] not in INITIAL_COUNT_SOURCES:
        raise SystemExit(f"Unknown initial count source: {settings['initial_count_source']!r}")
    if settings["sampled_count_source"] not in SAMPLED_COUNT_SOURCES:
        raise SystemExit(
            f"Unknown sampled count source: {settings['sampled_count_source']!r}"
        )
    if not 0.0 <= float(settings["tau_occ"]) <= 1.0:
        raise SystemExit("--tau-occ must be within [0, 1]")
    if int(settings["persistence"]) < 1:
        raise SystemExit("--persistence must be >= 1")
    if int(settings["frame_count"]) < 1:
        raise SystemExit("--frame-count must be >= 1")
    if settings["sam2_checkpoint"] is None:
        settings["sam2_checkpoint"] = os.environ.get("SAM2_CHECKPOINT")
    if settings["gdino_path"] is None:
        settings["gdino_path"] = os.environ.get("GDINO_PATH")
    if settings["sam2_checkpoint"] is not None:
        settings["sam2_checkpoint"] = str(settings["sam2_checkpoint"])
    settings["tau_occ"] = float(settings["tau_occ"])
    settings["reliable_logit"] = float(settings["reliable_logit"])
    settings["reliable_area_ratio"] = float(settings["reliable_area_ratio"])
    settings["presence_logit"] = float(settings["presence_logit"])
    settings["contact_margin_ratio"] = float(settings["contact_margin_ratio"])
    settings["min_center_distance"] = float(settings["min_center_distance"])
    settings["persistence"] = int(settings["persistence"])
    settings["frame_count"] = int(settings["frame_count"])
    args.__dict__.update(settings)
    return settings


def _load_pipeline_modules() -> tuple[Any, Any, Any]:
    if str(PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(PIPELINE_DIR))
    from t4g_detect import detect_all
    from t4g_exam_v2 import count_valid_instances
    from t4g_gdino import GDinoLocator

    return detect_all, count_valid_instances, GDinoLocator


def make_locator(device: str, gdino_path: str | None) -> Any:
    detect_all, count_valid_instances, GDinoLocator = _load_pipeline_modules()
    if gdino_path:
        import t4g_gdino

        t4g_gdino.GDINO_PATH = gdino_path
    return GDinoLocator(device=device)


def read_sample_frames(
    video: Path, frame_count: int, sample_mode: str
) -> tuple[list[Any], list[int]]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decodable frames: {video}")
    indices = sample_indices(len(frames), frame_count, sample_mode)
    return [frames[index] for index in indices], indices


def detect_sample_counts(
    locator: Any, frames: list[Any], mover: str, settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-sample target/robot detections plus raw and gripper-filtered counts."""
    detect_all, count_valid_instances, _ = _load_pipeline_modules()
    records = []
    for frame in frames:
        objects = detect_all(
            locator,
            frame,
            mover,
            topk=int(settings["object_topk"]),
            box_thr=float(settings["object_box_threshold"]),
        )
        grippers = detect_all(
            locator,
            frame,
            settings["robot_prompt"],
            topk=int(settings["robot_topk"]),
            box_thr=float(settings["robot_box_threshold"]),
        )
        filtered = count_valid_instances(
            objects,
            grippers,
            min_dist=float(settings["min_center_distance"]),
            grip_overlap_thr=float(settings["gripper_overlap_threshold"]),
        )
        records.append(
            {
                "objects": [list(detection[2]) for detection in objects],
                "grippers": [list(detection[2]) for detection in grippers],
                "raw_count": len(objects),
                "count": int(filtered),
            }
        )
    return records


def build_sam2_predictor(settings: dict[str, Any]) -> Any:
    if not settings.get("sam2_checkpoint"):
        raise SystemExit(
            "The occlusion rules need a SAM2 checkpoint: pass --sam2-checkpoint, or set "
            "the SAM2_CHECKPOINT environment variable. Checkpoints are at "
            "https://github.com/facebookresearch/sam2 (e.g. sam2.1_hiera_tiny.pt)."
        )
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise SystemExit(
            "SAM2 is required for --occlusion-rule paper_overlap/sam_presence/contact_recovery. "
            "Install it from https://github.com/facebookresearch/sam2, pass --sam2-checkpoint and "
            "--sam2-model-config, or fall back to --occlusion-rule none."
        ) from error
    return build_sam2_video_predictor(
        settings["sam2_model_config"],
        str(settings["sam2_checkpoint"]),
        device=settings["device"],
        apply_postprocessing=False,
    )


def _bbox_from_mask(mask: Any) -> tuple[float, float, float, float] | None:
    import numpy as np

    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _bbox_distance(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return math.inf
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    dx = max(ax0 - bx1, bx0 - ax1, 0.0)
    dy = max(ay0 - by1, by0 - ay1, 0.0)
    return math.hypot(dx, dy)


def _mask_distance(first: Any, second: Any) -> float:
    import cv2
    import numpy as np

    if first is None or second is None or not first.any() or not second.any():
        return math.inf
    distance_to_second = cv2.distanceTransform(
        np.logical_not(second).astype(np.uint8), cv2.DIST_L2, 3
    )
    return float(distance_to_second[first].min())


def _object_logit(state: Any, frame_index: int, object_id: int) -> float | None:
    try:
        object_index = state["obj_id_to_idx"][object_id]
        outputs = state["output_dict_per_obj"][object_index]
    except (KeyError, TypeError):
        return None
    entry = outputs["cond_frame_outputs"].get(frame_index) or outputs[
        "non_cond_frame_outputs"
    ].get(frame_index)
    if entry is None:
        return None
    return float(entry["object_score_logits"].float().item())


def _write_frame_directory(frames: list[Any], directory: Path) -> None:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.jpg"):
        stale.unlink()
    for index, frame in enumerate(frames):
        path = directory / f"{index:05d}.jpg"
        if not cv2.imwrite(
            str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]
        ):
            raise RuntimeError(f"Failed to write {path}")


def collect_sam2_evidence(
    predictor: Any,
    frames: list[Any],
    detection_records: list[dict[str, Any]],
    initial_count: int,
    frames_dir: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Track target instances and the robot with SAM2.1; return evidence + visuals."""
    import numpy as np
    import torch

    _write_frame_directory(frames, frames_dir)
    first_objects = detection_records[0]["objects"] if detection_records else []
    target_boxes = sorted(first_objects, key=lambda box: (box[0] + box[2]) * 0.5)[
        : max(1, int(initial_count))
    ]
    state = predictor.init_state(
        str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    target_ids = []
    for object_id, box in enumerate(target_boxes, start=1):
        target_ids.append(object_id)
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=object_id, box=np.asarray(box, dtype=np.float32)
        )
    # The robot prompt is refreshed on every sampled frame: GroundingDINO's box
    # usually spans the active gripper and attached arm, and SAM2 turns that
    # current-frame box into the mask used by the overlap test.
    gripper_id = 100
    for frame_index, record in enumerate(detection_records):
        if record["grippers"]:
            predictor.add_new_points_or_box(
                state,
                frame_idx=frame_index,
                obj_id=gripper_id,
                box=np.asarray(record["grippers"][0], dtype=np.float32),
            )

    diagonal = math.hypot(frames[0].shape[1], frames[0].shape[0]) if frames else 0.0
    evidence: dict[int, dict[str, Any]] = {}
    visuals: list[dict[str, Any]] = [{"target_masks": [], "gripper_mask": None} for _ in frames]
    initial_areas: list[int] | None = None
    last_reliable_masks: list[Any] = [None] * len(target_ids)
    last_target_boxes: list[Any] = [None] * len(target_ids)
    with torch.inference_mode():
        for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_index >= len(frames):
                continue
            masks = {
                int(object_id): mask_logits[position, 0].detach().cpu().numpy() > 0.0
                for position, object_id in enumerate(object_ids)
            }
            target_masks = [masks.get(object_id) for object_id in target_ids]
            gripper_mask = masks.get(gripper_id)
            if any(mask is None for mask in target_masks):
                continue
            target_areas = [int(mask.sum()) for mask in target_masks]
            if initial_areas is None:
                initial_areas = [max(1, area) for area in target_areas]
            gripper_box = _bbox_from_mask(gripper_mask)
            instances = []
            for position, (mask, area, logit) in enumerate(
                zip(target_masks, target_areas, [_object_logit(state, frame_index, oid) for oid in target_ids])
            ):
                initial_area = initial_areas[position]
                reliable = (logit is None or logit >= float(settings["reliable_logit"])) and (
                    area >= float(settings["reliable_area_ratio"]) * initial_area
                )
                box = _bbox_from_mask(mask)
                if reliable and box is not None:
                    last_target_boxes[position] = box
                if reliable and area > 0:
                    last_reliable_masks[position] = mask.copy()
                # A non-empty current mask is the best current position; when SAM2
                # emits nothing, the last reliable mask stands in for it.
                reference = mask if area > 0 else last_reliable_masks[position]
                source = "current" if area > 0 else "last_reliable"
                overlap = 0.0
                if reference is not None and reference.any() and gripper_mask is not None:
                    intersection = np.logical_and(reference, gripper_mask).sum()
                    overlap = float(intersection / reference.sum())
                anchor = box if reliable and box is not None else last_target_boxes[position]
                pixel_distance = (
                    _mask_distance(mask, gripper_mask) if reliable and area > 0 else math.inf
                )
                contact_distance = (
                    pixel_distance
                    if math.isfinite(pixel_distance)
                    else _bbox_distance(anchor, gripper_box)
                )
                instances.append(
                    {
                        "instance_id": target_ids[position],
                        "logit": logit,
                        "area": area,
                        "initial_area": initial_area,
                        "overlap": overlap,
                        "overlap_source": source,
                        "contact": bool(
                            contact_distance <= float(settings["contact_margin_ratio"]) * diagonal
                        ),
                    }
                )
            evidence[frame_index] = {
                "instances": instances,
                "contact": any(instance["contact"] for instance in instances),
            }
            visuals[frame_index] = {"target_masks": target_masks, "gripper_mask": gripper_mask}

    ordered: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for frame_index in range(len(frames)):
        entry = evidence.get(frame_index)
        if entry is None:
            entry = dict(previous) if previous is not None else {"instances": [], "contact": False}
            entry["propagation_missing"] = True
        ordered.append(entry)
        previous = entry
    predictor.reset_state(state)
    return ordered, visuals


def collect_trace(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[Any], Any]:
    """Run the detection stack on one video and return trace, metadata, frames, visuals."""
    need_evidence = (
        args.occlusion_rule != "none" or args.compare_all or args.render_dir is not None
    )
    frames, source_indices = read_sample_frames(
        Path(args.video), args.frame_count, args.sample_mode
    )
    locator = make_locator(args.device, args.gdino_path)
    records = detect_sample_counts(locator, frames, args.mover, args.__dict__)

    initial_count = args.initial_count
    detect_all, count_valid_instances, _ = _load_pipeline_modules()
    if initial_count is None:
        if args.condition_image is None:
            raise SystemExit("Pass --initial-count or --condition-image so N_0 is known.")
        import numpy as np
        from PIL import Image

        image = np.asarray(Image.open(args.condition_image).convert("RGB"))
        objects = detect_all(
            locator,
            image,
            args.mover,
            topk=int(args.object_topk),
            box_thr=float(args.object_box_threshold),
        )
        grippers = detect_all(
            locator,
            image,
            args.robot_prompt,
            topk=int(args.robot_topk),
            box_thr=float(args.robot_box_threshold),
        )
        if args.initial_count_source == "raw":
            initial_count = len(objects)
        else:
            initial_count = int(
                count_valid_instances(
                    objects,
                    grippers,
                    min_dist=float(args.min_center_distance),
                    grip_overlap_thr=float(args.gripper_overlap_threshold),
                )
            )
    initial_count = int(initial_count)

    evidence: list[dict[str, Any]] | None = None
    visuals: Any = None
    if need_evidence:
        predictor = build_sam2_predictor(args.__dict__)
        frames_dir = args.frames_dir or (
            args.output.parent / f"{args.output.stem}_frames"
            if args.output
            else Path(tempfile.mkdtemp(prefix="mlr_frames_"))
        )
        evidence, visuals = collect_sam2_evidence(
            predictor, frames, records, initial_count, Path(frames_dir), args.__dict__
        )

    samples = []
    for index, record in enumerate(records):
        sample: dict[str, Any] = {
            "sample_index": index,
            "source_frame": source_indices[index],
            "raw_count": record["raw_count"],
            "count": record["count"],
        }
        if evidence is not None:
            sample["instances"] = evidence[index]["instances"]
            sample["contact"] = evidence[index]["contact"]
        samples.append(sample)
    trace = {
        "initial_count": initial_count,
        "mover": args.mover,
        "video": str(args.video),
        "condition_image": None if args.condition_image is None else str(args.condition_image),
        "initial_count_source": args.initial_count_source,
        "sample_mode": args.sample_mode,
        "frame_count": args.frame_count,
        "samples": samples,
    }
    metadata = {
        "video": str(args.video),
        "detection_records": [
            {
                "sample_index": index,
                "source_frame": source_indices[index],
                "raw_count": record["raw_count"],
                "filtered_count": record["count"],
                "object_count": len(record["objects"]),
                "gripper_count": len(record["grippers"]),
            }
            for index, record in enumerate(records)
        ],
    }
    return trace, metadata, frames, visuals


def render_samples(
    frames: list[Any],
    samples: list[dict[str, Any]],
    visuals: Any,
    labels: list[str],
    output_dir: Path,
) -> Path | None:
    import cv2
    import numpy as np

    if visuals is None:
        return None
    colors = [(50, 220, 70), (230, 90, 40), (210, 80, 210), (40, 200, 230)]
    tiles = []
    for index, frame in enumerate(frames):
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        visual = visuals[index] if index < len(visuals) else None
        if visual and (visual["target_masks"] or visual["gripper_mask"] is not None):
            overlay = canvas.copy()
            for position, mask in enumerate(visual["target_masks"]):
                if mask is not None:
                    overlay[mask] = colors[position % len(colors)]
            if visual["gripper_mask"] is not None:
                overlay[visual["gripper_mask"]] = (0, 180, 255)
            canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0.0)
        sample = samples[index]
        text = (
            f"#{index:02d} frame={sample['source_frame']:03d} "
            f"raw={sample['raw_count']} N={sample['count']} {labels[index]}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (25, 25, 25), -1)
        cv2.putText(
            canvas, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
        )
        tiles.append(canvas)
    height, width = tiles[0].shape[:2]
    thumb_width = 384
    thumb_height = max(1, round(height * thumb_width / width))
    small = [
        cv2.resize(tile, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        for tile in tiles
    ]
    columns = 4
    rows_count = math.ceil(len(small) / columns)
    blank = np.full_like(small[0], 245)
    small.extend([blank.copy() for _ in range(rows_count * columns - len(small))])
    montage = np.vstack(
        [np.hstack(small[row * columns : (row + 1) * columns]) for row in range(rows_count)]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "montage.jpg"
    cv2.imwrite(str(path), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return path


# --------------------------------------------------------------------------- #
# result assembly and CLI
# --------------------------------------------------------------------------- #


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
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


def protocol_metadata(settings: dict[str, Any], initial_count: int | None) -> dict[str, Any]:
    return {
        "protocol": "mlr_occlusion_v1",
        "deviation_mode": settings["deviation_mode"],
        "occlusion_rule": settings["occlusion_rule"],
        "persistence_samples": settings["persistence"],
        "tau_occ": settings["tau_occ"],
        "reliable_logit": settings["reliable_logit"],
        "reliable_area_ratio": settings["reliable_area_ratio"],
        "presence_logit": settings["presence_logit"],
        "contact_margin_ratio": settings["contact_margin_ratio"],
        "sample_mode": settings["sample_mode"],
        "frame_count": settings["frame_count"],
        "initial_count": initial_count,
        "initial_count_source": settings["initial_count_source"],
        "sampled_count_source": settings["sampled_count_source"],
        "object_prompt_topk": settings["object_topk"],
        "object_box_threshold": settings["object_box_threshold"],
        "robot_prompt": settings["robot_prompt"],
        "robot_topk": settings["robot_topk"],
        "robot_box_threshold": settings["robot_box_threshold"],
        "gripper_overlap_threshold": settings["gripper_overlap_threshold"],
        "min_center_distance": settings["min_center_distance"],
        "sam2_checkpoint": settings["sam2_checkpoint"],
        "sam2_model_config": settings["sam2_model_config"],
        "device": settings["device"],
        "profile": settings.get("profile_name"),
    }


def build_result(
    args: argparse.Namespace,
    samples: list[Any],
    initial_count: int | None,
    *,
    source: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = evaluate_samples(
        samples,
        initial_count if initial_count is not None else 0,
        deviation_mode=args.deviation_mode,
        occlusion_rule=args.occlusion_rule,
        tau_occ=args.tau_occ,
        persistence=args.persistence,
        reliable_logit=args.reliable_logit,
        reliable_area_ratio=args.reliable_area_ratio,
        presence_logit=args.presence_logit,
        count_source=args.sampled_count_source,
    )
    payload: dict[str, Any] = {
        "protocol": protocol_metadata(args.__dict__, initial_count),
        "summary": summarize([result]),
        "evaluation": result,
    }
    if source:
        payload["source"] = source
    if trace is not None:
        payload["trace"] = trace
    if args.compare_all:
        payload["comparison"] = compare_rules(
            samples,
            initial_count if initial_count is not None else 0,
            tau_occ=args.tau_occ,
            persistence=args.persistence,
            reliable_logit=args.reliable_logit,
            reliable_area_ratio=args.reliable_area_ratio,
            presence_logit=args.presence_logit,
            count_source=args.sampled_count_source,
        )
    return payload


def print_comparison(comparison: dict[str, Any]) -> None:
    key_width = max(len(key) for key in comparison)
    print(f"{'rule'.ljust(key_width)}  event  mlr%   onset  max_run  adjusted_counts")
    for key, entry in comparison.items():
        mlr = entry["mlr_percent"]
        print(
            f"{key.ljust(key_width)}  {str(entry['event']):5s}  "
            f"{'n/a' if mlr is None else format(mlr, '.1f'):>5s}  "
            f"{str(entry['onset_sample_index']):>5s}  {entry['max_run']:>7d}  "
            f"{entry['adjusted_counts']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLR with selectable deviation and occlusion rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Deviation modes: "
            + ", ".join(DEVIATION_MODES)
            + "\nOcclusion rules: "
            + ", ".join(OCCLUSION_RULES)
            + "\nSampled count sources: "
            + ", ".join(SAMPLED_COUNT_SOURCES)
            + "\nProfiles live in "
            + str(PROFILES_FILE)
        ),
    )
    parser.add_argument("--trace", type=Path, help="stored trace JSON to evaluate")
    parser.add_argument("--video", type=Path, help="video to evaluate end to end")
    parser.add_argument("--mover", type=str, help="object name used as the detection prompt")
    parser.add_argument("--condition-image", type=Path, help="image that defines N_0")
    parser.add_argument("--initial-count", type=int, default=None, help="override N_0")
    parser.add_argument("--output", type=Path, help="write the result JSON here")
    parser.add_argument("--render-dir", type=Path, help="write a mask montage here")
    parser.add_argument("--frames-dir", type=Path, help="frame dump for the SAM2 pass")
    parser.add_argument("--quiet", action="store_true", help="print only a one-line summary")
    parser.add_argument("--compare-all", action="store_true", help="evaluate every rule combination")
    parser.add_argument("--self-test", action="store_true", help="run offline logic checks")
    parser.add_argument("--profile", type=str, help="named profile from the profiles YAML")
    parser.add_argument("--profiles-file", type=Path, default=PROFILES_FILE)
    parser.add_argument(
        "--list-profiles", action="store_true", help="print the available profiles and exit"
    )
    for name, choices in (
        ("deviation-mode", DEVIATION_MODES),
        ("occlusion-rule", OCCLUSION_RULES),
        ("sample-mode", SAMPLE_MODES),
        ("sampled-count-source", SAMPLED_COUNT_SOURCES),
    ):
        parser.add_argument(f"--{name}", choices=choices, default=None)
    for name, kind in (
        ("frame-count", int),
        ("persistence", int),
        ("tau-occ", float),
        ("reliable-logit", float),
        ("reliable-area-ratio", float),
        ("presence-logit", float),
        ("contact-margin-ratio", float),
        ("object-topk", int),
        ("object-box-threshold", float),
        ("robot-topk", int),
        ("robot-box-threshold", float),
        ("gripper-overlap-threshold", float),
        ("min-center-distance", float),
    ):
        parser.add_argument(f"--{name}", type=kind, default=None)
    parser.add_argument("--robot-prompt", type=str, default=None)
    parser.add_argument("--initial-count-source", choices=INITIAL_COUNT_SOURCES, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gdino-path", type=str, default=None)
    parser.add_argument("--sam2-checkpoint", type=str, default=None)
    parser.add_argument("--sam2-model-config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if args.list_profiles:
        profiles = _available_profiles(args.profiles_file)
        if not profiles:
            if not Path(args.profiles_file).is_file():
                print(f"Profiles file not found: {args.profiles_file}")
            else:
                print(
                    f"No profiles found in {args.profiles_file} "
                    "(PyYAML missing? install `pyyaml` to read the profiles file)"
                )
            return
        for name, entry in profiles.items():
            description = (entry or {}).get("description", "") if isinstance(entry, dict) else ""
            print(f"{name}\n    {description}")
        return
    if args.trace is None and args.video is None:
        raise SystemExit("Pass --trace, --video, --list-profiles or --self-test.")

    resolve_settings(args)
    args.profile_name = args.profile

    frames: list[Any] = []
    visuals: Any = None
    if args.trace is not None:
        initial_count, entries, metadata = load_trace(args.trace)
        if initial_count is None:
            initial_count = args.initial_count
        if initial_count is None:
            raise SystemExit("--initial-count is required when the trace has no initial_count.")
        samples = [normalize_sample(entry, index) for index, entry in enumerate(entries)]
        payload = build_result(
            args,
            samples,
            int(initial_count),
            source={"kind": "trace", "path": str(args.trace), **metadata},
        )
    else:
        if not args.mover:
            raise SystemExit("--mover is required with --video.")
        trace, metadata, frames, visuals = collect_trace(args)
        samples = [normalize_sample(entry, index) for index, entry in enumerate(trace["samples"])]
        payload = build_result(
            args,
            samples,
            trace["initial_count"],
            source={"kind": "video", **metadata},
            trace=trace,
        )
        if args.render_dir is not None:
            labels = [row["state"] for row in payload["evaluation"]["rows"]]
            path = render_samples(frames, samples, visuals, labels, args.render_dir)
            if path is not None:
                payload["source"]["montage"] = str(path)

    if args.output is not None:
        atomic_write_json(args.output, payload)
    evaluation = payload["evaluation"]
    summary = payload["summary"]
    if not args.quiet:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(
            f"initial_count={evaluation['initial_count']} "
            f"mode={evaluation['deviation_mode']} rule={evaluation['occlusion_rule']} "
            f"event={evaluation['event']} onset={evaluation['onset_sample_index']} "
            f"mlr%={summary['mlr_percent']}"
        )
    if args.compare_all:
        print_comparison(payload["comparison"])
    if args.output is not None:
        print(f"Wrote {args.output}", flush=True)


# --------------------------------------------------------------------------- #
# offline logic checks
# --------------------------------------------------------------------------- #


def _instance(
    instance_id: int = 1,
    logit: float = 9.0,
    area: int = 1000,
    initial_area: int = 1000,
    overlap: float = 0.0,
    contact: bool = False,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "logit": logit,
        "area": area,
        "initial_area": initial_area,
        "overlap": overlap,
        "contact": contact,
    }


def _sample(
    index: int,
    raw_count: int,
    count: int | None = None,
    instances: list[dict[str, Any]] | None = None,
    contact: bool | None = None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "sample_index": index,
        "source_frame": index,
        "raw_count": raw_count,
        "count": raw_count if count is None else count,
    }
    if instances is not None:
        sample["instances"] = [dict(entry) for entry in instances]
    if contact is not None:
        sample["contact"] = contact
    return sample


def _self_test() -> None:
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: Any) -> None:
        checks.append((name, bool(condition)))

    # sampling axis
    check("sample_round_axis", sample_indices(12, 5, "round") == [0, 3, 6, 8, 11])
    check("sample_linspace_axis", sample_indices(12, 5, "linspace") == [0, 2, 5, 8, 11])
    axis = sample_indices(93, 24, "round")
    check("sample_axis_endpoints", len(axis) == 24 and axis[0] == 0 and axis[-1] == 92)
    check("sample_axis_monotonic", all(b >= a for a, b in zip(axis, axis[1:])))
    check("sample_single_frame", sample_indices(1, 24, "round") == [0] * 24)

    # persistence
    check("persistence_needs_consecutive", not has_persistent_run([True, False, True], 2))
    check("persistence_event", has_persistent_run([False, True, True], 2))
    check("persistence_onset", first_persistent_onset([False, True, True], 2) == 1)
    check("persistence_max_run", longest_run([True, True, False, True]) == 2)

    # over-count is always evidence
    over = [_sample(index, 3, 4) for index in range(3)]
    check(
        "over_count_is_evidence",
        evaluate_samples(over, 3, deviation_mode="symmetric", occlusion_rule="paper_overlap")[
            "event"
        ],
    )
    check(
        "over_count_is_evidence_legacy",
        evaluate_samples(over, 3, deviation_mode="over_only", occlusion_rule="none")["event"],
    )

    # under-count without occlusion evidence is evidence
    under = [_sample(index, 2, 2) for index in range(3)]
    check(
        "under_count_without_evidence",
        evaluate_samples(under, 3, deviation_mode="symmetric", occlusion_rule="paper_overlap")[
            "event"
        ],
    )

    # occlusion-supported under-count: one unreliable instance overlaps the robot
    occluded = [
        _instance(instance_id=1, logit=-1.0, area=80, initial_area=1000, overlap=0.42),
        _instance(instance_id=2, logit=9.0, area=1000, initial_area=1000),
        _instance(instance_id=3, logit=9.0, area=1000, initial_area=1000),
    ]
    exempt_samples = [_sample(index, 2, 2, instances=occluded) for index in range(2)]
    result = evaluate_samples(
        exempt_samples, 3, deviation_mode="symmetric", occlusion_rule="paper_overlap"
    )
    check("occlusion_exempts_under_count", not result["event"])
    check("occlusion_adjusted_count", set(result["adjusted_counts"]) == {3})
    check("occlusion_is_not_a_deviation", result["deviation_flags"] == [False, False])

    # tau_occ boundary is inclusive
    boundary = [dict(entry) for entry in occluded]
    boundary[0]["overlap"] = 0.15
    result = evaluate_samples(
        [_sample(index, 2, 2, instances=boundary) for index in range(2)],
        3,
        deviation_mode="symmetric",
        occlusion_rule="paper_overlap",
        tau_occ=0.15,
    )
    check("tau_occ_boundary_inclusive", not result["event"])
    below = [dict(entry) for entry in occluded]
    below[0]["overlap"] = 0.149
    result = evaluate_samples(
        [_sample(index, 2, 2, instances=below) for index in range(2)],
        3,
        deviation_mode="symmetric",
        occlusion_rule="paper_overlap",
        tau_occ=0.15,
    )
    check("tau_occ_boundary_excluded", result["event"])

    # an unreliable instance with too little overlap does not exempt the under-count
    thin = [dict(entry) for entry in occluded]
    thin[0]["overlap"] = 0.05
    check(
        "insufficient_overlap_not_exempt",
        evaluate_samples(
            [_sample(index, 2, 2, instances=thin) for index in range(2)],
            3,
            deviation_mode="symmetric",
            occlusion_rule="paper_overlap",
        )["event"],
    )
    # a collapsed mask area marks the instance unreliable even at high logit
    collapsed = [dict(entry) for entry in occluded]
    collapsed[0]["logit"] = 9.0
    collapsed[0]["area"] = 100
    check(
        "unreliable_by_area",
        not evaluate_samples(
            [_sample(index, 2, 2, instances=collapsed) for index in range(2)],
            3,
            deviation_mode="symmetric",
            occlusion_rule="paper_overlap",
        )["event"],
    )

    # sam_presence
    low_presence = [dict(entry) for entry in occluded]
    low_presence[0]["logit"] = -2.0
    low_presence[0]["overlap"] = 0.0
    check(
        "sam_presence_exempts",
        not evaluate_samples(
            [_sample(index, 2, 2, instances=low_presence) for index in range(2)],
            3,
            deviation_mode="symmetric",
            occlusion_rule="sam_presence",
        )["event"],
    )
    confident = [dict(entry) for entry in occluded]
    confident[0]["logit"] = 9.0
    confident[0]["overlap"] = 0.0
    check(
        "sam_presence_needs_low_logit",
        evaluate_samples(
            [_sample(index, 2, 2, instances=confident) for index in range(2)],
            3,
            deviation_mode="symmetric",
            occlusion_rule="sam_presence",
        )["event"],
    )

    # contact_recovery
    contact_instance = [_instance(instance_id=1, logit=5.0, contact=True)]
    plain_instance = [_instance(instance_id=1, logit=5.0)]
    recovering = [
        _sample(0, 2, 2, instances=contact_instance),
        _sample(1, 2, 2, instances=contact_instance),
        _sample(2, 3, 3),
    ]
    check(
        "contact_run_with_recovery_exempt",
        not evaluate_samples(
            recovering, 3, deviation_mode="symmetric", occlusion_rule="contact_recovery"
        )["event"],
    )
    check(
        "contact_run_without_recovery",
        evaluate_samples(
            recovering[:2], 3, deviation_mode="symmetric", occlusion_rule="contact_recovery"
        )["event"],
    )
    check(
        "run_without_contact",
        evaluate_samples(
            [
                _sample(0, 2, 2, instances=plain_instance),
                _sample(1, 2, 2, instances=plain_instance),
                _sample(2, 3, 3),
            ],
            3,
            deviation_mode="symmetric",
            occlusion_rule="contact_recovery",
        )["event"],
    )

    # an exempt frame resets the persistence criterion
    mixed = [_sample(0, 4, 4), _sample(1, 2, 2, instances=occluded), _sample(2, 4, 4)]
    result = evaluate_samples(
        mixed, 3, deviation_mode="symmetric", occlusion_rule="paper_overlap"
    )
    check("exemption_resets_persistence", not result["event"])
    check("exemption_resets_flags", result["deviation_flags"] == [True, False, True])
    check("exemption_onset_none", result["onset_sample_index"] is None)

    # over_only ignores under-counts
    check(
        "over_only_ignores_under_count",
        not evaluate_samples(
            [_sample(index, 2, 2) for index in range(4)],
            3,
            deviation_mode="over_only",
            occlusion_rule="none",
        )["event"],
    )

    # sampled count basis: filtered vs raw
    basis = [_sample(0, 3, 2), _sample(1, 3, 2)]
    check(
        "filtered_basis_reads_merged_count",
        evaluate_samples(basis, 3, deviation_mode="symmetric", occlusion_rule="none")["event"],
    )
    check(
        "raw_basis_reads_raw_count",
        not evaluate_samples(
            basis, 3, deviation_mode="symmetric", occlusion_rule="none", count_source="raw"
        )["event"],
    )
    mixed_basis = [_sample(index, 4, 2, instances=occluded) for index in range(2)]
    check(
        "occlusion_exemption_follows_basis",
        not evaluate_samples(
            mixed_basis, 3, deviation_mode="symmetric", occlusion_rule="paper_overlap"
        )["event"]
        and evaluate_samples(
            mixed_basis,
            3,
            deviation_mode="symmetric",
            occlusion_rule="paper_overlap",
            count_source="raw",
        )["event"],
    )
    flat = [normalize_sample(entry, index) for index, entry in enumerate([3, 2, 2, 3])]
    check(
        "single_count_trace_is_basis_independent",
        evaluate_samples(flat, 3, deviation_mode="symmetric", occlusion_rule="none")["event"]
        and evaluate_samples(
            flat, 3, deviation_mode="symmetric", occlusion_rule="none", count_source="raw"
        )["event"],
    )

    # bare count traces and scoring
    entries = [normalize_sample(entry, index) for index, entry in enumerate([3, 3, 2])]
    check(
        "bare_count_trace",
        evaluate_samples(entries, 2, deviation_mode="symmetric", occlusion_rule="none")["event"],
    )
    check(
        "ineligible_clip",
        not evaluate_samples(entries, 0, deviation_mode="symmetric", occlusion_rule="none")[
            "eligible"
        ],
    )
    summary = summarize(
        [
            {"eligible": True, "event": True, "onset_sample_index": 4},
            {"eligible": True, "event": False, "onset_sample_index": None},
            {"eligible": False, "event": False, "onset_sample_index": None},
        ]
    )
    check("summary_mlr_percent", summary["mlr_percent"] == 50.0)
    check("summary_coverage_percent", abs(summary["coverage_percent"] - 200.0 / 3) < 1e-9)
    comparison = compare_rules([_sample(index, 2, 2, instances=occluded) for index in range(3)], 3)
    check("compare_rules_covers_every_combination", len(comparison) == 8)
    check(
        "compare_rules_disagrees_across_modes",
        comparison["over_only|paper_overlap"]["event"] is False
        and comparison["symmetric|none"]["event"] is True,
    )

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit(f"self-test failures: {', '.join(failed)}")


if __name__ == "__main__":
    main()
