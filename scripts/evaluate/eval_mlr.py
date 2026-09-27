#!/usr/bin/env python3
"""Score generated clips with the Missing-instance rate (MLR).

The script resolves the frozen protocol of ``--config``, scores the clips of ``--pred-dir``
with ``eveworld.evaluation.mlr`` and writes the aggregate to ``--output``::

    python scripts/evaluate/eval_mlr.py \
        --config configs/eval/mlr/dreamgen.yaml \
        --pred-dir outputs/dreamgen_eveworld/generated_only \
        --metadata data/metadata/dreamgenbench \
        --output results/item_level/dreamgen/mlr_eveworld.json

Every clip is scored on the sampled timestamps of the protocol, ``eval.sample_count`` of them
in ``eval.sample_mode`` order: Grounding-DINO counts the instances of the target named by the
clip's instruction, SAM2 tracks the frame-0 detections of the target and of the robot gripper
through the clip, an instance whose mask is covered by the gripper mask by more than
``eval.tau_occ`` is exempt at that timestamp, and a deviation from the frame-0 inventory
becomes an event once it persists over ``eval.k`` consecutive sampled timestamps. The counter
reads its checkpoint and config from ``$GROUNDING_DINO_CONFIG`` and ``$GROUNDING_DINO_WEIGHTS``,
the tracker from ``$SAM2_CHECKPOINT`` unless ``--sam2-checkpoint`` names one; both run on
``--device``.

``--pred-dir`` is either the directory holding the generated MP4s or the layout root that keeps
them in ``generated_only/``. ``--metadata`` is the directory (or the file) of the per-clip
metadata: the stem of a clip keys its entry and the ``target`` of the entry is the object the
counter prompts with. The entry decides the prompt-level eligibility of the clip - the
``eligible`` flag of DreamGenBench, a resolved target and mover on WorldArena, a resolved
source and destination on RoboTwin, a resolved target elsewhere. A clip whose instruction
parsed is scored, and joins the MLR denominator when its first sampled timestamp holds at least
one instance; a clip whose first frame shows no instance of the target, and one the metadata
already rejects, are ineligible and leave the denominator. The ``coverage`` of every row is the
eligible share of the benchmark, read from ``eligible_ids.json`` when the metadata directory
publishes one, otherwise the eligible share of the run itself.

``--output`` receives the summary and ``--rows`` the per-clip rows, one JSON object per line in
the schema of ``results/item_level/README.md``; ``--rows`` defaults to ``--output`` with a
``.jsonl`` suffix. ``--limit`` caps the number of clips for a smoke run, and ``--dry-run``
prints the plan and the resolved clips without building a detector or touching a device.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".avi")
GENERATED_SUBDIR = "generated_only"
METADATA_FILE = "target_queries.json"
ELIGIBLE_FILE = "eligible_ids.json"
ROWS_SUFFIX = ".jsonl"
BENCHMARK_KEYS = ("num_eligible", "num_total", "num_resolved", "coverage")
PREVIEW = 3

# The profile keys of the evaluation configurations, under the names the audit block records.
PROTOCOL_KEYS = (
    ("eval.deviation_mode", "deviation_mode"),
    ("eval.occlusion_rule", "occlusion_rule"),
    ("eval.tau_occ", "tau_occ"),
    ("eval.k", "persistence"),
    ("eval.sample_count", "sample_count"),
    ("eval.sample_mode", "sample_mode"),
    ("eval.object_topk", "object_topk"),
    ("eval.object_box_threshold", "object_box_threshold"),
    ("eval.robot_prompt", "robot_prompt"),
    ("eval.robot_topk", "robot_topk"),
    ("eval.robot_box_threshold", "robot_box_threshold"),
    ("eval.min_center_distance", "min_center_distance"),
    ("eval.gripper_overlap_threshold", "gripper_overlap_threshold"),
    ("eval.reliable_area_ratio", "reliable_area_ratio"),
    ("eval.reliable_logit", "reliable_logit"),
    ("eval.presence_logit", "presence_logit"),
    ("eval.contact_margin_ratio", "contact_margin_ratio"),
    ("eval.initial_count_source", "initial_count_source"),
    ("eval.sampled_count_source", "sampled_count_source"),
    ("eval.sam2_checkpoint", "sam2_checkpoint"),
    ("eval.sam2_model_config", "sam2_model_config"),
    ("eval.profile", "profile_name"),
)

LAYOUT = """\
files written by a run:

    --output    the aggregate summary, a JSON object
    --rows      the rows it was computed from, one JSON object per clip

A row carries request_id, model, mlr, eligible, events and coverage next to the counting detail
of the clip, as documented in results/item_level/README.md.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the MLR evaluation script."""
    parser = argparse.ArgumentParser(
        description="Score generated clips with the Missing-instance rate (MLR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help=(
            "evaluation configuration, e.g. configs/eval/mlr/dreamgen.yaml "
            "(default: $EVEWORLD_CONFIG)"
        ),
    )
    run.add_argument(
        "--pred-dir",
        help="directory of the generated clips, or the layout root holding generated_only/",
    )
    run.add_argument("--metadata", help="directory (or file) of the per-clip metadata")
    run.add_argument("--output", help="JSON file the aggregate summary is written to")
    run.add_argument(
        "--rows",
        help="JSONL file of the per-clip rows (default: --output with a .jsonl suffix)",
    )
    run.add_argument("--limit", type=int, help="score at most this many clips")
    run.add_argument(
        "--model",
        help="arm that generated the clips, e.g. eveworld (default: the directory name)",
    )
    protocol = parser.add_argument_group("protocol")
    protocol.add_argument("--device", help="torch device of the detector and the tracker")
    protocol.add_argument("--sam2-checkpoint", help="override eval.sam2_checkpoint")
    protocol.add_argument("--sam2-config", help="override eval.sam2_model_config")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the resolved clips, without scoring them",
    )
    return parser.parse_args(argv)


def repo_root() -> Path:
    """Repository root: ``eveworld.utils.io.repo_root`` when importable, else the pyproject walk."""
    try:
        from eveworld.utils.io import repo_root as _library_root
    except ImportError:
        pass
    else:
        return Path(_library_root())
    path = Path(__file__).resolve()
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def load_config(config_path: Path, overrides: Sequence[str]) -> Any:
    """Parse ``config_path`` with ``overrides`` applied on top of it."""
    from eveworld.utils.config import load_config as _load_config

    return _load_config(config_path, overrides)


def cfg_value(config: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` out of a parsed configuration without raising on a missing key."""
    from eveworld.utils.config import cfg_get

    return cfg_get(config, key, default)


def protocol_settings(config: Any, args: argparse.Namespace, device: str) -> dict[str, Any]:
    """The protocol settings of the run, under the names the audit block records.

    A setting the configuration does not carry is left out, so the audit block falls back to
    the frozen default of the protocol for it.
    """
    settings: dict[str, Any] = {"device": device}
    for key, name in PROTOCOL_KEYS:
        value = cfg_value(config, key)
        if value is not None:
            settings[name] = value
    if args.sam2_checkpoint:
        settings["sam2_checkpoint"] = args.sam2_checkpoint
    if args.sam2_config:
        settings["sam2_model_config"] = args.sam2_config
    return settings


def resolve_pred_dir(named: str | None, config: Any, root: Path) -> Path:
    """Directory of the generated clips: the named one, or its ``generated_only/`` subdirectory."""
    from eveworld.utils.config import resolve_path

    if named:
        candidate = resolve_path(named, root)
    else:
        candidate = resolve_path(cfg_value(config, "output_dir", "outputs"), root) / GENERATED_SUBDIR
    nested = candidate / GENERATED_SUBDIR
    return nested if nested.is_dir() else candidate


def select_clips(pred_dir: Path, limit: int | None) -> list[Path]:
    """Generated clips of ``pred_dir``, in name order, capped at ``limit``."""
    if not pred_dir.is_dir():
        return []
    clips = sorted(
        (
            path
            for path in pred_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    return clips if limit is None else clips[: int(limit)]


def load_metadata(metadata: Path) -> dict[str, Any]:
    """Per-clip metadata entries of the benchmark, keyed by request id; empty when there is none."""
    from eveworld.utils.io import read_json

    for candidate in (metadata, metadata / METADATA_FILE):
        if candidate.is_file():
            document = read_json(candidate)
            if not isinstance(document, dict):
                raise ValueError(f"metadata file {candidate} is not a JSON object")
            return document
    return {}


def benchmark_block(metadata: Path) -> dict[str, Any]:
    """The eligible set the benchmark publishes, or an empty mapping when it publishes none."""
    from eveworld.utils.io import read_json

    candidates = [metadata / ELIGIBLE_FILE, metadata] if metadata.is_dir() else [metadata]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        document = read_json(candidate)
        block: dict[str, Any] = {"file": str(candidate)}
        if isinstance(document, dict):
            for key in BENCHMARK_KEYS:
                if key in document:
                    block[key] = document[key]
        return block
    return {}


def entry_target(entry: Any) -> str | None:
    """Target object the instruction of a metadata entry asks for, when it parsed to one."""
    if not isinstance(entry, dict):
        return None
    value = entry.get("target")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def prompt_eligible(entry: Any) -> bool:
    """Whether a metadata entry names an instruction the protocol can score.

    DreamGenBench marks the entry itself, WorldArena publishes the parsed target next to the
    mover that acts on it, RoboTwin the source and the destination of the motion; an entry
    with none of those fields is eligible when its instruction parsed to a target.
    """
    if not isinstance(entry, dict):
        return False
    if "eligible" in entry:
        return bool(entry["eligible"])
    if "mover" in entry:
        return bool(entry_target(entry)) and bool(entry.get("mover"))
    if "source" in entry or "destination" in entry:
        return bool(entry.get("source")) and bool(entry.get("destination"))
    return bool(entry_target(entry))


@dataclass
class EvalPlan:
    """Everything a run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration, the stem of the default output directory.
        protocol: The frozen protocol of the run, i.e. the settings every clip is scored with.
        pred_dir: Directory the generated clips are read from.
        clips: Clips selected for scoring, in name order.
        limit: Number of clips ``--limit`` caps the run at, or ``None`` for all of them.
        model: Arm the clips were generated by, as recorded on every row.
        metadata: Directory or file the per-clip metadata is read from.
        entries: Metadata entries keyed by request id, empty when there is no metadata file.
        benchmark: The eligible set the benchmark publishes, empty when it publishes none.
        output: JSON file the aggregate is written to, or ``None`` for the standard output only.
        rows: JSONL file the per-clip rows are written to, or ``None`` for no row file.
        command: Command line of the run, quoted, as recorded in the summary.
        dry_run: Whether the run stops after the plan.
    """

    config_path: Path
    config: Any
    name: str
    protocol: dict[str, Any]
    pred_dir: Path
    clips: list[Path]
    limit: int | None
    model: str
    metadata: Path
    entries: dict[str, Any]
    benchmark: dict[str, Any]
    output: Path | None
    rows: Path | None
    command: str
    dry_run: bool


def build_plan(args: argparse.Namespace, config_path: Path, argv: Sequence[str]) -> EvalPlan:
    """Resolve the configuration, the clips and the protocol into an :class:`EvalPlan`."""
    from eveworld.evaluation.mlr.merge import protocol_metadata
    from eveworld.utils.config import resolve_path

    root = repo_root()
    config = load_config(config_path, [])
    pred_dir = resolve_pred_dir(args.pred_dir, config, root)
    metadata = resolve_path(
        args.metadata or str(cfg_value(config, "data.metadata", "data/metadata")), root
    )
    device = args.device or str(cfg_value(config, "eval.device", "cuda"))
    protocol = protocol_metadata(protocol_settings(config, args, device), None)
    if pred_dir.name == GENERATED_SUBDIR:
        model = args.model or pred_dir.parent.name
    else:
        model = args.model or pred_dir.name
    output = None if not args.output else resolve_path(args.output, root)
    if args.rows:
        rows = resolve_path(args.rows, root)
    elif output is not None:
        rows = output.with_suffix(ROWS_SUFFIX)
    else:
        rows = None
    command = " ".join(
        shlex.quote(part) for part in ("python", "scripts/evaluate/eval_mlr.py", *argv)
    )
    return EvalPlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        protocol=protocol,
        pred_dir=pred_dir,
        clips=select_clips(pred_dir, args.limit),
        limit=args.limit,
        model=model,
        metadata=metadata,
        entries=load_metadata(metadata),
        benchmark=benchmark_block(metadata),
        output=output,
        rows=rows,
        command=command,
        dry_run=bool(args.dry_run),
    )


def benchmark_line(block: dict[str, Any]) -> str:
    """Human-readable eligible set of the benchmark, or a note that it publishes none."""
    if not block:
        return "no eligible set published"
    parts = [f"{key} {block[key]}" for key in BENCHMARK_KEYS if key in block]
    return ", ".join(parts)


def print_plan(plan: EvalPlan) -> None:
    """Print the run, one aligned field per line."""
    protocol = plan.protocol
    found = "" if plan.pred_dir.is_dir() else " (not found)"
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(
        f"protocol:     {protocol['protocol']}, {int(protocol['frame_count'])} timestamps "
        f"sampled {protocol['sample_mode']}, k {int(protocol['persistence_samples'])}, "
        f"tau_occ {float(protocol['tau_occ']):g}"
    )
    print(f"pred dir:     {plan.pred_dir}{found}")
    print(f"model:        {plan.model}")
    print(f"metadata:     {plan.metadata}")
    print(
        f"entries:      {len(plan.entries)} clips, "
        f"{sum(1 for entry in plan.entries.values() if prompt_eligible(entry))} prompt-eligible"
    )
    print(f"benchmark:    {benchmark_line(plan.benchmark)}")
    limit = "" if plan.limit is None else f", limit {plan.limit}"
    print(f"clips:        {len(plan.clips)} selected{limit}")
    for path in plan.clips[:PREVIEW]:
        target = entry_target(plan.entries.get(path.stem))
        print(f"clip:         {path.name} -> {target or '(no target)'}")
    if len(plan.clips) > PREVIEW:
        print(f"clip:         ... and {len(plan.clips) - PREVIEW} more")
    print(
        f"detector:     Grounding-DINO on {protocol['device']}, object topk "
        f"{int(protocol['object_prompt_topk'])} at box {float(protocol['object_box_threshold']):g}"
    )
    print(
        f"tracker:      SAM2 {protocol['sam2_model_config']} at "
        f"{protocol['sam2_checkpoint'] or '$SAM2_CHECKPOINT'}"
    )
    print(f"output:       {plan.output or 'standard output'}")
    print(f"rows:         {plan.rows or 'none'}")
    print(f"dry run:      {'yes' if plan.dry_run else 'no'}")


def build_counter(protocol: dict[str, Any]) -> Any:
    """Grounding-DINO counter of the protocol, with its backend built before the first clip."""
    from eveworld.evaluation.mlr import InstanceCounter

    counter = InstanceCounter(
        device=str(protocol["device"]),
        box_threshold=float(protocol["object_box_threshold"]),
        robot_prompt=str(protocol["robot_prompt"]),
        object_topk=int(protocol["object_prompt_topk"]),
        robot_topk=int(protocol["robot_topk"]),
        robot_box_threshold=float(protocol["robot_box_threshold"]),
        min_dist=float(protocol["min_center_distance"]),
        grip_overlap_thr=float(protocol["gripper_overlap_threshold"]),
    )
    try:
        backend = counter.detector
    except (ImportError, FileNotFoundError, RuntimeError) as error:
        raise RuntimeError(f"the Grounding-DINO detector is unavailable: {error}") from error
    if backend is None:
        raise RuntimeError("the Grounding-DINO detector could not be built")
    return counter


def build_tracker(protocol: dict[str, Any]) -> Any:
    """SAM2 tracker of the protocol, built from the checkpoint of the run."""
    from eveworld.data.tracking.sam2_tracker import Sam2Tracker

    kwargs: dict[str, Any] = {}
    if protocol.get("sam2_model_config"):
        kwargs["model_cfg"] = str(protocol["sam2_model_config"])
    try:
        return Sam2Tracker(
            checkpoint=protocol.get("sam2_checkpoint"),
            device=str(protocol["device"]),
            **kwargs,
        )
    except (ImportError, FileNotFoundError, RuntimeError) as error:
        raise RuntimeError(f"the SAM2 tracker is unavailable: {error}") from error


def boxes_of(detections: Sequence[Any]) -> Any:
    """``(N, 4)`` float32 xyxy boxes of a detection list, e.g. of ``InstanceCounter.detect``."""
    import numpy as np

    boxes: list[Any] = []
    for detection in detections:
        value = detection.get("box") if isinstance(detection, dict) else getattr(detection, "box", None)
        if value is None:
            continue
        box = np.asarray(value, dtype=np.float32).reshape(-1)
        if box.size == 4:
            boxes.append(box)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(boxes).astype(np.float32)


def row_of(
    plan: EvalPlan,
    path: Path,
    *,
    target: str | None = None,
    eligible: bool = False,
    reference_count: int | None = None,
    counts: list[int] | None = None,
    exemptions: list[bool] | None = None,
    events: list[bool] | None = None,
    mlr: float | None = None,
    skipped: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """One row of the result file: the clip, its counting detail and its outcome."""
    flags = list(events or [])
    row: dict[str, Any] = {
        "request_id": path.stem,
        "model": plan.model,
        "clip": path.name,
        "target": target,
        "eligible": bool(eligible) and error is None,
        "reference_count": reference_count,
        "counts": counts,
        "exemptions": exemptions,
        "events": flags,
        "event": any(flags),
        "mlr": None if error is not None else mlr,
        "coverage": None,
        "protocol": str(plan.protocol["protocol"]),
    }
    if skipped is not None:
        row["skipped"] = skipped
    if error is not None:
        row["error"] = error
    return row


def score_clip(
    plan: EvalPlan,
    path: Path,
    target: str,
    counter: Any,
    tracker: Any,
) -> dict[str, Any]:
    """Score one clip of the plan and return its row."""
    import numpy as np

    from eveworld.data.transforms.video import load_video
    from eveworld.evaluation.mlr import (
        adjust_counts,
        align_timestamps,
        clip_mlr,
        detect_events,
        merge_track_masks,
    )
    from eveworld.evaluation.mlr.occlusion import exempt_targets

    protocol = plan.protocol
    frames = load_video(path)
    indices = align_timestamps(
        int(frames.shape[0]), int(protocol["frame_count"]), str(protocol["sample_mode"])
    )
    sampled = frames[indices]
    counts = counter.counts(
        sampled, target, box_threshold=float(protocol["object_box_threshold"])
    )
    counts_list = [int(value) for value in counts]
    reference = int(counts[0]) if counts.size else 0
    if reference <= 0:
        return row_of(
            plan,
            path,
            target=target,
            reference_count=reference,
            counts=counts_list,
            skipped="no instance of the target in the first timestamp",
        )
    target_boxes = boxes_of(
        counter.detect(
            sampled[0:1],
            target,
            topk=int(protocol["object_prompt_topk"]),
            box_threshold=float(protocol["object_box_threshold"]),
        )[0]
    )
    if target_boxes.shape[0] == 0:
        raise ValueError(f"no box of the target {target!r} on the first frame of {path.name}")
    robot_boxes = boxes_of(
        counter.detect(
            sampled[0:1],
            str(protocol["robot_prompt"]),
            topk=int(protocol["robot_topk"]),
            box_threshold=float(protocol["robot_box_threshold"]),
        )[0]
    )
    target_masks = merge_track_masks(tracker.track(frames, target_boxes).masks, indices)
    if robot_boxes.shape[0]:
        robot_masks = merge_track_masks(tracker.track(frames, robot_boxes).masks, indices)
        robot_mask = robot_masks.any(axis=1)
    else:
        robot_mask = np.zeros(
            (indices.size, int(frames.shape[1]), int(frames.shape[2])), dtype=bool
        )
    exempt = exempt_targets(target_masks, robot_mask, tau_occ=float(protocol["tau_occ"]))
    adjusted = adjust_counts(counts, exempt, reference)
    events = detect_events(
        adjusted != reference, exemptions=adjusted != counts, k=int(protocol["persistence_samples"])
    )
    flags = [bool(value) for value in events]
    return row_of(
        plan,
        path,
        target=target,
        eligible=True,
        reference_count=reference,
        counts=counts_list,
        exemptions=[bool(value) for value in adjusted != counts],
        events=flags,
        mlr=clip_mlr(flags, True),
    )


def score_clips(plan: EvalPlan, counter: Any, tracker: Any) -> list[dict[str, Any]]:
    """Score every selected clip of the plan and return one row per clip.

    A clip the metadata does not cover, or whose instruction parsed to no target, is recorded
    as an ineligible row; a clip whose scoring raises is recorded with the error, so that one
    unreadable file does not lose the rest of the run.
    """
    rows: list[dict[str, Any]] = []
    for path in plan.clips:
        entry = plan.entries.get(path.stem)
        if entry is None:
            rows.append(row_of(plan, path, error="no metadata entry for this clip"))
            continue
        target = entry_target(entry)
        if target is None:
            rows.append(row_of(plan, path, skipped="no target in the metadata entry"))
            continue
        if not prompt_eligible(entry):
            rows.append(row_of(plan, path, target=target, skipped="ineligible in the metadata"))
            continue
        try:
            rows.append(score_clip(plan, path, target, counter, tracker))
        except (OSError, ValueError, RuntimeError) as error:
            rows.append(
                row_of(plan, path, target=target, error=f"{type(error).__name__}: {error}")
            )
    return rows


def summarize(plan: EvalPlan, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the rows into the summary written to ``--output``.

    The reported MLR is the share of the eligible clips with at least one event, the paper's
    per-clip outcome; ``mean_clip_mlr`` is the pooled per-timestamp rate that
    ``eveworld.evaluation.mlr.metric.clip_mlr`` returns for a single clip.
    """
    from eveworld.evaluation.mlr import coverage
    from eveworld.evaluation.mlr.metric import wilson_interval

    errors = sorted(str(row["error"]) for row in rows if row.get("error"))
    eligible = [row for row in rows if row["eligible"]]
    with_events = [row for row in eligible if row["event"]]
    run_coverage = coverage(len(eligible), len(rows))
    published = plan.benchmark.get("coverage")
    share = run_coverage if published is None else float(published)
    rates = [float(row["mlr"]) for row in eligible if row.get("mlr") is not None]
    mlr = 0.0 if not eligible else 100.0 * len(with_events) / len(eligible)
    interval = wilson_interval(len(with_events), len(eligible))
    return {
        "config": str(plan.config_path),
        "name": plan.name,
        "model": plan.model,
        "protocol": str(plan.protocol["protocol"]),
        "protocol_settings": plan.protocol,
        "command": plan.command,
        "pred_dir": str(plan.pred_dir),
        "selected_prompts": len(plan.clips),
        "records": len(rows),
        "eligible": len(eligible),
        "coverage": share,
        "run_coverage": run_coverage,
        "events": len(with_events),
        "mlr": mlr,
        "mean_clip_mlr": 0.0 if not rates else sum(rates) / len(rates),
        "wilson_95": [float(interval[0]), float(interval[1])],
        "benchmark": dict(plan.benchmark),
        "errors": errors,
    }


def run_evaluation(plan: EvalPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score the clips of the plan and return the summary with the rows."""
    if not plan.pred_dir.is_dir():
        raise FileNotFoundError(f"prediction directory {plan.pred_dir} does not exist")
    if not plan.entries:
        raise FileNotFoundError(f"no metadata entries under {plan.metadata}")
    counter = build_counter(plan.protocol)
    tracker = build_tracker(plan.protocol)
    rows = score_clips(plan, counter, tracker)
    summary = summarize(plan, rows)
    for row in rows:
        row["coverage"] = summary["coverage"]
    return summary, rows


def write_results(plan: EvalPlan, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the aggregate and the per-clip rows, when the run named files for them."""
    from eveworld.utils.io import write_json, write_jsonl

    if plan.output is not None:
        write_json(summary, plan.output)
    if plan.rows is not None:
        write_jsonl(rows, plan.rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: resolve the run, then score the clips or dry-run them."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(arguments)
    if not args.config:
        print("error: --config is required (or export EVEWORLD_CONFIG)", file=sys.stderr)
        return 2
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        print(f"error: config file {config_path} does not exist", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2
    try:
        plan = build_plan(args, config_path, arguments)
        print_plan(plan)
        if plan.dry_run:
            print("dry run: no clip scored")
            return 0
        summary, rows = run_evaluation(plan)
        write_results(plan, summary, rows)
        print(json.dumps(summary, indent=2))
        return 0
    except ImportError as error:
        print(f"error: the eveworld package is not importable: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
