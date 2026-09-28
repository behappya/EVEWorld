"""Five-row aggregate of the RoboTwin cross-backbone study (Table 5).

Each variant is transferred to the 250 held-out RoboTwin episodes (50 tasks, episodes 45 to 49)
and scored clip by clip against the reference recording of the same episode: PSNR, SSIM and LPIPS
for appearance and flow end-point error for motion. EVEWorld finishes at 12.765 / 0.769 / 0.365 /
2.207 against 12.218 / 0.748 / 0.383 / 3.033 for the FlowWAM Stage-1 control, and the ranking on
flow error is the point of the table: injecting the interaction-aware objective keeps motion
closer to the demonstration than either baseline.

:func:`evaluate_clip` works from paths because the generated and reference clips live in separate
directories of the campaign layout, and because decoding a clip once per metric would multiply
the cost of a 250-clip sweep. MLR is the exception: it needs the per-timestamp event array
produced by annotation, so the caller passes ``events`` and gets the clip's MLR back. Rows that
carry the ``mlr_*`` keys are pooled by :func:`aggregate` into the MLR columns of the table.

The module runs as a script::

    python -m eveworld.evaluation.robotwin.evaluate \\
        --pair eve=/results/heldout/eve:/data/robotwin/reference

``--pair`` follows the ``--variant NAME=VIDEO_DIR`` flag of the FlowWAM campaign scripts but takes
the reference directory as well, so one invocation scores a whole variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from eveworld.utils.io import list_files, write_json
from eveworld.utils.logging import get_logger

from .flow_epe import compute_flow, flow_epe
from .psnr import DEFAULT_LPIPS_SAMPLES, DEFAULT_SSIM_SAMPLES, lpips_clip, psnr_clip, ssim_clip

logger = get_logger(__name__)

__all__ = ["VARIANTS", "aggregate", "evaluate_clip", "main"]

VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "stage1", "label": "FlowWAM Stage-1", "igr": False, "tia": False},
    {"name": "sft", "label": "Standard SFT", "igr": False, "tia": False},
    {"name": "igr", "label": "+ IGR", "igr": True, "tia": False},
    {"name": "tia", "label": "+ TIA", "igr": False, "tia": True},
    {"name": "eve", "label": "EVEWorld", "igr": True, "tia": True},
)

DEFAULT_METRICS: tuple[str, ...] = ("psnr", "ssim", "lpips", "flow_epe")
DEFAULT_OUTPUT = Path("aggregate/table5.json")
PROTOCOL = "robotwin_table5_v1"
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")
_CSV_HEADER = (
    "Variant",
    "IGR",
    "TIA",
    "PSNR",
    "SSIM",
    "LPIPS",
    "Flow-EPE",
    "MLR (%)",
    "MLR events",
    "MLR eligible",
)


def evaluate_clip(
    pred_path: str | Path,
    target_path: str | Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ssim_samples: int = DEFAULT_SSIM_SAMPLES,
    lpips_samples: int = DEFAULT_LPIPS_SAMPLES,
    events: np.ndarray | None = None,
    mlr_eligible: bool = True,
    raft: Any = None,
    device: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Score one generated clip against the reference recording of the same episode.

    Both clips are truncated to the frames they share, so a generated clip that ends early is
    still scorable and the metric never compares frame ``i`` of one clip with a later frame of
    the other.

    Args:
        pred_path: Generated clip, any format the video reader accepts.
        target_path: Reference clip for the same episode.
        metrics: Subset of :data:`DEFAULT_METRICS` to compute; each entry costs one pass over
            the decoded clips, with ``flow_epe`` the expensive one.
        ssim_samples: Frames sampled for SSIM.
        lpips_samples: Frames sampled for LPIPS.
        events: Per-timestamp MLR event flags ``(T,)`` bool from the annotation stage. When it is
            given, the row carries the clip's MLR, its event and timestamp counts and whether the
            clip counts towards the eligible set of the table.
        mlr_eligible: Whether this clip belongs to the MLR eligible set, that is, whether the
            instruction names instances whose presence could be checked at all.
        raft: Flow extractor for ``flow_epe``; defaults to the FlowWAM RAFT model.
        device: Torch device for LPIPS and for the default flow extractor.
        variant: Variant key or label to record on the row, used by :func:`aggregate`.

    Returns:
        A row with the clip paths, the frame counts and one float per requested metric, plus
        ``flow_pairs`` when flow was requested and the ``mlr_*`` keys when ``events`` was given.
    """
    requested = _check_metrics(metrics)
    pred_file = Path(pred_path)
    target_file = Path(target_path)
    for path in (pred_file, target_file):
        if not path.is_file():
            raise FileNotFoundError(f"clip not found: {path}")

    from eveworld.data.transforms.video import load_video

    pred = load_video(pred_file)
    target = load_video(target_file)
    pred_frames = len(pred)
    target_frames = len(target)
    aligned = min(pred_frames, target_frames)
    if aligned == 0:
        raise ValueError(f"clip pair has no frames: {pred_file} vs {target_file}")
    pred = pred[:aligned]
    target = target[:aligned]

    row: dict[str, Any] = {
        "pred_path": str(pred_file),
        "target_path": str(target_file),
        "pred_frames": int(pred_frames),
        "target_frames": int(target_frames),
        "aligned_frames": int(aligned),
    }
    if variant is not None:
        row["variant"] = variant
    if "psnr" in requested:
        row["psnr"] = psnr_clip(pred, target)
    if "ssim" in requested:
        row["ssim"] = ssim_clip(pred, target, samples=ssim_samples)
    if "lpips" in requested:
        row["lpips"] = lpips_clip(pred, target, samples=lpips_samples, device=device)
    if "flow_epe" in requested:
        if aligned < 2:
            raise ValueError(f"need at least two aligned frames for flow: {pred_file}")
        pred_flow = compute_flow(pred, raft=raft, device=device)
        target_flow = compute_flow(target, raft=raft, device=device)
        row["flow_epe"] = flow_epe(pred_flow, target_flow)
        row["flow_pairs"] = int(len(pred_flow))
    if events is not None:
        row.update(_mlr_row(events, mlr_eligible))
    logger.debug("scored %s against %s (%d aligned frames)", pred_file.name, target_file.name, aligned)
    return row


def aggregate(rows: Iterable[dict[str, Any]], *, output: str | Path | None = None) -> dict[str, Any]:
    """Pool per-clip rows into the five rows of Table 5.

    Each metric is the mean over the clips of the variant. MLR is pooled instead of averaged, so
    that a variant with more eligible clips does not get the same weight per clip as a shorter
    one: the fraction is ``events / timestamps`` summed over the eligible clips. Pass ``output``
    to write the report as JSON, with the CSV rendering of the same table next to it.

    Args:
        rows: Rows from :func:`evaluate_clip`, each carrying a ``variant`` key.
        output: Path of the JSON report, or of a directory to hold ``table5.json``.

    Returns:
        ``{"protocol", "metrics", "rows"}``, where every row holds ``key``, ``variant`` (the
        display label), ``igr``, ``tia``, ``clips``, one float per metric and the MLR columns.
        Metrics missing from the input rows are reported as ``None``.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("variant") or "").strip()
        if not key:
            raise ValueError("every row needs a 'variant' key, e.g. from --pair NAME=...")
        grouped.setdefault(_variant_key(key), []).append(row)

    report_rows = []
    for key in _ordered_keys(grouped):
        report_rows.append(_aggregate_group(key, grouped[key]))
    report = {"protocol": PROTOCOL, "metrics": list(DEFAULT_METRICS), "rows": report_rows}
    if output is not None:
        json_path, csv_path = _write_table(report, Path(output))
        logger.info("wrote %s and %s", json_path, csv_path)
    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; scores every ``--pair`` and writes ``aggregate/table5.json``."""
    parser = argparse.ArgumentParser(
        prog="eveworld.evaluation.robotwin.evaluate",
        description="Score the five Table 5 variants on the held-out RoboTwin episodes.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="NAME=PRED_DIR:TARGET_DIR",
        help="variant key and the generated and reference directories to score; repeatable",
    )
    parser.add_argument("--limit", type=int, default=None, help="score at most this many clips per pair")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report to write")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="comma separated metric names")
    parser.add_argument("--ssim-samples", type=int, default=DEFAULT_SSIM_SAMPLES)
    parser.add_argument("--lpips-samples", type=int, default=DEFAULT_LPIPS_SAMPLES)
    parser.add_argument("--device", default=None, help="torch device for LPIPS and RAFT")
    args = parser.parse_args(argv)

    if not args.pair:
        print(
            "error: at least one --pair NAME=PRED_DIR:TARGET_DIR is required; NAME is one of " + ", ".join(variant["name"] for variant in VARIANTS),
            file=sys.stderr,
        )
        return 2

    metrics = tuple(part.strip() for part in args.metrics.split(",") if part.strip())
    rows: list[dict[str, Any]] = []
    for spec in args.pair:
        key, pred_dir, target_dir = _parse_pair(spec)
        pairs = _match_pairs(pred_dir, target_dir)
        if args.limit is not None:
            pairs = pairs[: args.limit]
        logger.info("scoring %d clips for %s", len(pairs), key)
        for pred_path, target_path in pairs:
            try:
                row = evaluate_clip(
                    pred_path,
                    target_path,
                    metrics=metrics,
                    ssim_samples=args.ssim_samples,
                    lpips_samples=args.lpips_samples,
                    device=args.device,
                    variant=key,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                logger.error("skipping %s: %s", pred_path, exc)
                continue
            rows.append(row)
    report = aggregate(rows, output=args.output)
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _mlr_row(events: np.ndarray, eligible: bool) -> dict[str, Any]:
    """Score the per-timestamp MLR events of one clip."""
    flags = np.asarray(events).astype(bool)
    if flags.ndim != 1:
        raise ValueError(f"events must be a (T,) bool array, got {flags.shape}")
    clip_mlr = _import_clip_mlr()
    return {
        "mlr": float(clip_mlr(flags, eligible=eligible)),
        "mlr_events": int(flags.sum()),
        "mlr_timestamps": int(flags.size),
        "mlr_eligible": bool(eligible),
    }


def _import_clip_mlr() -> Any:
    """Return ``mlr.metric.clip_mlr``, naming the missing module when the extra is not installed."""
    try:
        from eveworld.evaluation.mlr.metric import clip_mlr
    except ImportError as exc:
        raise RuntimeError(
            "MLR scoring needs eveworld.evaluation.mlr.metric, which is missing from this "
            'checkout; install the evaluation subpackages with pip install -e ".[eval]"'
        ) from exc
    return clip_mlr


def _aggregate_group(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean the metrics of one variant and pool its MLR columns."""
    meta = _variant_meta(key)
    entry: dict[str, Any] = {
        "key": key,
        "variant": meta["label"],
        "igr": meta["igr"],
        "tia": meta["tia"],
        "clips": len(rows),
    }
    for metric in DEFAULT_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        entry[metric] = float(np.mean(values)) if values else None
    eligible = [row for row in rows if row.get("mlr_eligible")]
    events = sum(int(row.get("mlr_events") or 0) for row in eligible)
    timestamps = sum(int(row.get("mlr_timestamps") or 0) for row in eligible)
    entry["mlr_percent"] = 100.0 * events / timestamps if timestamps else None
    entry["mlr_events"] = events if timestamps else None
    entry["mlr_eligible"] = len(eligible) if timestamps else None
    return entry


def _write_table(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Write the JSON report and the matching CSV rendering of Table 5."""
    json_path = output if output.suffix == ".json" else output / "table5.json"
    csv_path = json_path.with_suffix(".csv")
    write_json(report, json_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_HEADER)
        for row in report["rows"]:
            writer.writerow(
                [
                    str(row["variant"]),
                    str(int(bool(row["igr"]))),
                    str(int(bool(row["tia"]))),
                    _format_number(row.get("psnr"), 3),
                    _format_number(row.get("ssim"), 3),
                    _format_number(row.get("lpips"), 3),
                    _format_number(row.get("flow_epe"), 3),
                    _format_number(row.get("mlr_percent"), 2),
                    _format_number(row.get("mlr_events")),
                    _format_number(row.get("mlr_eligible")),
                ]
            )
    return json_path, csv_path


def _format_number(value: Any, digits: int = 0) -> str:
    """Render a cell of the CSV table; a metric that was not computed leaves the cell empty."""
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _check_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    """Validate a metric selection against :data:`DEFAULT_METRICS`."""
    requested = tuple(str(name) for name in metrics)
    unknown = [name for name in requested if name not in DEFAULT_METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; supported: {list(DEFAULT_METRICS)}")
    if not requested:
        raise ValueError("at least one metric is required")
    return requested


def _variant_key(value: str) -> str:
    """Map a variant key or display label onto the key of :data:`VARIANTS`."""
    for variant in VARIANTS:
        if value in (variant["name"], variant["label"]):
            return str(variant["name"])
    return value


def _variant_meta(key: str) -> dict[str, Any]:
    """Return the label and the objective flags of a variant, warning when it is not in the table."""
    for variant in VARIANTS:
        if variant["name"] == key:
            return dict(variant)
    logger.warning("%s is not a variant of Table 5", key)
    return {"name": key, "label": key, "igr": False, "tia": False}


def _ordered_keys(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Order the variants as in the table, with any variant outside it after them."""
    known = [str(variant["name"]) for variant in VARIANTS if str(variant["name"]) in grouped]
    return known + [key for key in grouped if key not in known]


def _parse_pair(spec: str) -> tuple[str, Path, Path]:
    """Split ``NAME=PRED_DIR:TARGET_DIR`` into its three parts."""
    name, separator, paths = spec.partition("=")
    if not separator or not name.strip():
        raise ValueError(f"invalid --pair {spec!r}; expected NAME=PRED_DIR:TARGET_DIR")
    pred_dir, separator, target_dir = paths.partition(":")
    if not separator or not pred_dir.strip() or not target_dir.strip():
        raise ValueError(f"invalid --pair {spec!r}; expected NAME=PRED_DIR:TARGET_DIR")
    return name.strip(), Path(pred_dir.strip()), Path(target_dir.strip())


def _match_pairs(pred_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    """Pair every generated clip with the reference clip of the same relative path or name."""
    videos = [path for path in list_files(pred_dir) if path.name.endswith(VIDEO_SUFFIXES)]
    pairs = []
    for pred_path in videos:
        relative = target_dir / pred_path.relative_to(pred_dir)
        target_path = relative if relative.is_file() else target_dir / pred_path.name
        if not target_path.is_file():
            logger.warning("no reference clip for %s", pred_path)
            continue
        pairs.append((pred_path, target_path))
    return pairs


if __name__ == "__main__":
    raise SystemExit(main())
