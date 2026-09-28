#!/usr/bin/env python3
"""Score the generated clips of a RoboTwin arm with the metrics of paper Table 5.

The script resolves the protocol of ``--config``, scores every generated clip against the
reference recording of the same episode and writes the five-row table to ``--output``::

    python scripts/evaluate/eval_robotwin.py \
        --config configs/eval/mlr/robotwin.yaml \
        --pred-dir outputs/robotwin_eveworld/generated_only \
        --target-dir data/robotwin/heldout \
        --output results/item_level/robotwin/table5.json

Every clip pair is truncated to the frames the two recordings share and scored with the metrics of
``--metrics``: PSNR with a data range of 255, SSIM over ``--ssim-samples`` frames, LPIPS over
``--lpips-samples`` frames and the end-point error of the optical flow averaged over the pixels
with a valid flow. The rows are pooled per variant, each metric as the mean over the clips. MLR
comes from the annotation stage rather than from the pixels: ``--mlr-rows`` points at the rows
``scripts/evaluate/eval_mlr.py`` wrote for the same clips, their per-timestamp ``events`` and their
``eligible`` flag are joined onto the clip by request id, and the aggregate pools the fraction as
``events / timestamps`` over the eligible clips. Without ``--mlr-rows`` the MLR columns of the
table stay empty, and a ``--metadata`` entry with ``eligible: false`` marks the clip as ineligible.

``--pair NAME=PRED_DIR:TARGET_DIR`` scores several arms in one run and is repeatable, with the
``NAME`` of :data:`eveworld.evaluation.robotwin.evaluate.VARIANTS` selecting the label and the
IGR / TIA flags of the row. Without ``--pair`` the run scores one ``--variant`` from
``--pred-dir`` and ``--target-dir``. Clips are paired on their relative path inside the two
directories, falling back to their name, so a layout that keeps one directory per task works as
well as a flat one.

``--output`` receives the five-row report as JSON with the CSV rendering of the table next to it,
and ``--rows`` the per-clip rows, one JSON object per line in the schema of
``results/item_level/README.md``; ``--rows`` defaults to ``--output`` with a ``.jsonl`` suffix. A
clip that could not be scored is reported under ``errors`` and kept in the row file with its
error, while the table is pooled from the clips that were scored. ``--limit`` caps the clips of
every pair for a smoke run, and ``--dry-run`` prints the plan and the matched clips without
decoding a single frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")
GENERATED_SUBDIR = "generated_only"
ROWS_SUFFIX = ".jsonl"
PREVIEW = 3
PROTOCOL = "robotwin_table5_v1"
DEFAULT_VARIANT = "eve"
DEFAULT_OUTPUT = "results/item_level/robotwin/table5.json"
METRICS = ("psnr", "ssim", "lpips", "flow_epe")
DEFAULT_SSIM_SAMPLES = 16
DEFAULT_LPIPS_SAMPLES = 8
MLR_ROW_KEYS = ("events", "eligible", "mlr")

LAYOUT = """\
files written by a run:

    --output    the five-row table as JSON, with the CSV rendering next to it
    --rows      the clips it was pooled from, one JSON object per clip

A row carries request_id, model and the variant of the clip next to the metrics of the table and
the mlr keys of the annotation stage, as documented in results/item_level/README.md.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the RoboTwin evaluation script."""
    parser = argparse.ArgumentParser(
        description="Score generated RoboTwin clips with the metrics of paper Table 5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help="evaluation configuration, e.g. configs/eval/mlr/robotwin.yaml (default: $EVEWORLD_CONFIG)",
    )
    run.add_argument(
        "--pred-dir",
        help="directory of the generated clips, or the layout root holding generated_only/",
    )
    run.add_argument("--target-dir", help="directory of the reference recordings of the episodes")
    run.add_argument(
        "--metadata",
        help="directory (or file) of the per-clip metadata, read for the MLR eligibility flags",
    )
    run.add_argument("--output", help=f"report of the run (default: {DEFAULT_OUTPUT})")
    run.add_argument(
        "--rows",
        help="JSONL file of the per-clip rows (default: --output with a .jsonl suffix)",
    )
    run.add_argument("--limit", type=int, help="score at most this many clips per pair")
    protocol = parser.add_argument_group("protocol")
    protocol.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="NAME=PRED_DIR:TARGET_DIR",
        help="arm and the generated and reference directories to score; repeatable",
    )
    protocol.add_argument(
        "--variant",
        help=f"name of the single-arm run, e.g. eve (default: {DEFAULT_VARIANT})",
    )
    protocol.add_argument(
        "--metrics",
        help=f"comma separated metrics (default: {','.join(METRICS)})",
    )
    protocol.add_argument("--ssim-samples", type=int, help="frames sampled for SSIM")
    protocol.add_argument("--lpips-samples", type=int, help="frames sampled for LPIPS")
    protocol.add_argument("--device", help="torch device of LPIPS and of the flow extractor")
    protocol.add_argument(
        "--mlr-rows",
        help="JSONL rows of scripts/evaluate/eval_mlr.py, joined onto the clips by request id",
    )
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the matched clips, without decoding a clip",
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


def resolve_pred_dir(named: str | None, config: Any, root: Path) -> Path:
    """Directory of the generated clips: the named one, or its ``generated_only/`` subdirectory."""
    from eveworld.utils.config import resolve_path

    if named:
        candidate = resolve_path(named, root)
    else:
        candidate = resolve_path(cfg_value(config, "output_dir", "outputs"), root) / GENERATED_SUBDIR
    nested = candidate / GENERATED_SUBDIR
    return nested if nested.is_dir() else candidate


def metric_names(args: argparse.Namespace, config: Any) -> tuple[str, ...]:
    """Metrics of the run: the named ones, or the metrics of the configuration in table order."""
    if args.metrics:
        requested = tuple(part.strip() for part in args.metrics.split(",") if part.strip())
    else:
        configured = cfg_value(config, "eval.metrics", ())
        if isinstance(configured, str):
            configured = [configured]
        requested = tuple(name for name in METRICS if name in {str(item) for item in configured})
    unknown = [name for name in requested if name not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; supported: {list(METRICS)}")
    if not requested:
        raise ValueError("at least one metric is required")
    return requested


def parse_pair(spec: str) -> tuple[str, Path, Path]:
    """Split ``NAME=PRED_DIR:TARGET_DIR`` into its three parts."""
    name, separator, paths = spec.partition("=")
    if not separator or not name.strip():
        raise ValueError(f"invalid --pair {spec!r}; expected NAME=PRED_DIR:TARGET_DIR")
    pred_dir, separator, target_dir = paths.partition(":")
    if not separator or not pred_dir.strip() or not target_dir.strip():
        raise ValueError(f"invalid --pair {spec!r}; expected NAME=PRED_DIR:TARGET_DIR")
    return name.strip(), Path(pred_dir.strip()).expanduser(), Path(target_dir.strip()).expanduser()


def match_pairs(pred_dir: Path, target_dir: Path, limit: int | None) -> list[tuple[Path, Path]]:
    """Pair every generated clip with the reference recording of the same path or name."""
    from eveworld.utils.io import list_files

    pairs: list[tuple[Path, Path]] = []
    for pred_path in list_files(pred_dir):
        if not pred_path.name.endswith(VIDEO_SUFFIXES):
            continue
        relative = target_dir / pred_path.relative_to(pred_dir)
        target_path = relative if relative.is_file() else target_dir / pred_path.name
        if not target_path.is_file():
            continue
        pairs.append((pred_path, target_path))
    return pairs if limit is None else pairs[: int(limit)]


def load_mlr_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    """Rows of the MLR pass keyed by request id, empty when the run named no row file."""
    from eveworld.utils.io import read_jsonl

    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"the MLR rows {path} do not exist; run eval_mlr.py first")
    return {str(row["request_id"]): dict(row) for row in read_jsonl(path) if row.get("request_id")}


def load_entries(metadata: Path) -> dict[str, Any]:
    """Per-clip metadata entries of the benchmark, keyed by request id; empty when there is none."""
    from eveworld.utils.io import read_json

    candidates = [metadata] if metadata.is_file() else sorted(metadata.glob("*.json"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        document = read_json(candidate)
        if isinstance(document, dict) and any(isinstance(entry, dict) for entry in document.values()):
            return {str(key): entry for key, entry in document.items()}
    return {}


def entry_eligible(entry: Any) -> bool:
    """Whether a metadata entry leaves the clip in the eligible set of the MLR columns."""
    if not isinstance(entry, dict) or "eligible" not in entry:
        return True
    return bool(entry["eligible"])


@dataclass
class PairSpec:
    """One arm of the table and the clips it is scored on.

    Attributes:
        name: Variant key or label of the arm, e.g. ``eve``.
        pred_dir: Directory the generated clips of the arm are read from.
        target_dir: Directory the reference recordings are read from, or ``None`` when the run
            names no references for the arm.
        clips: Matched clip pairs, in name order.
    """

    name: str
    pred_dir: Path
    target_dir: Path | None
    clips: list[tuple[Path, Path]] = field(default_factory=list)


@dataclass
class EvalPlan:
    """Everything a run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration, the stem of the default output directory.
        protocol: The frozen protocol of the table, i.e. the settings every clip is scored with.
        pairs: The arms of the run with their matched clips.
        limit: Number of clips ``--limit`` caps every pair at, or ``None`` for all of them.
        entries: Metadata entries keyed by request id, read for the eligibility flags.
        mlr: Rows of the MLR pass keyed by request id, empty when the run has no ``--mlr-rows``.
        output: JSON report of the table, or ``None`` for the standard output only.
        rows: JSONL file the per-clip rows are written to, or ``None`` for no row file.
        command: Command line of the run, quoted, as recorded in the report.
        dry_run: Whether the run stops after the plan.
    """

    config_path: Path
    config: Any
    name: str
    protocol: dict[str, Any]
    pairs: list[PairSpec]
    limit: int | None
    entries: dict[str, Any]
    mlr: dict[str, dict[str, Any]]
    output: Path | None
    rows: Path | None
    command: str
    dry_run: bool


def build_plan(args: argparse.Namespace, config_path: Path, argv: Sequence[str]) -> EvalPlan:
    """Resolve the configuration, the arms and the protocol into an :class:`EvalPlan`."""
    from eveworld.utils.config import resolve_path

    root = repo_root()
    config = load_config(config_path, [])
    if args.pair:
        specs = [
            PairSpec(
                name=name,
                pred_dir=resolve_path(str(pred_dir), root),
                target_dir=resolve_path(str(target_dir), root),
            )
            for name, pred_dir, target_dir in (parse_pair(spec) for spec in args.pair)
        ]
    else:
        target = resolve_path(args.target_dir, root) if args.target_dir else None
        specs = [
            PairSpec(
                name=args.variant or DEFAULT_VARIANT,
                pred_dir=resolve_pred_dir(args.pred_dir, config, root),
                target_dir=target,
            )
        ]
    for spec in specs:
        if spec.target_dir is not None:
            spec.clips = match_pairs(spec.pred_dir, spec.target_dir, args.limit)
    protocol: dict[str, Any] = {
        "protocol": PROTOCOL,
        "metrics": list(metric_names(args, config)),
        "ssim_samples": int(args.ssim_samples if args.ssim_samples is not None else cfg_value(config, "eval.ssim_samples", DEFAULT_SSIM_SAMPLES)),
        "lpips_samples": int(
            args.lpips_samples if args.lpips_samples is not None else cfg_value(config, "eval.lpips_samples", DEFAULT_LPIPS_SAMPLES)
        ),
        "device": args.device or str(cfg_value(config, "eval.device", "cuda")),
    }
    for key, name in (
        ("eval.sample_count", "mlr_sample_count"),
        ("eval.flow_cond", "flow_cond"),
        ("eval.cfg_scale", "cfg_scale"),
        ("eval.num_steps", "num_steps"),
        ("eval.seed", "seed"),
        ("data.metadata", "metadata"),
        ("data.split", "split"),
    ):
        value = cfg_value(config, key)
        if value is not None:
            protocol[name] = value
    metadata = resolve_path(args.metadata or str(cfg_value(config, "data.metadata", "data/metadata/robotwin")), root)
    mlr_path = resolve_path(args.mlr_rows, root) if args.mlr_rows else None
    output = resolve_path(args.output or DEFAULT_OUTPUT, root)
    rows = resolve_path(args.rows, root) if args.rows else output.with_suffix(ROWS_SUFFIX)
    return EvalPlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        protocol=protocol,
        pairs=specs,
        limit=args.limit,
        entries=load_entries(metadata),
        mlr=load_mlr_rows(mlr_path),
        output=output,
        rows=rows,
        command=" ".join(shlex.quote(part) for part in ("python", "scripts/evaluate/eval_robotwin.py", *argv)),
        dry_run=bool(args.dry_run),
    )


def metrics_line(protocol: dict[str, Any]) -> str:
    """Human-readable metric selection of the run."""
    samples = [f"{name} {int(protocol[f'{name}_samples'])} samples" for name in ("ssim", "lpips") if name in protocol["metrics"]]
    return ", ".join(protocol["metrics"]) + (f" ({', '.join(samples)})" if samples else "")


def print_plan(plan: EvalPlan) -> None:
    """Print the run, one aligned field per line."""
    protocol = plan.protocol
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(f"protocol:     {protocol['protocol']}, MLR over {protocol.get('mlr_sample_count', '-')} samples")
    print(f"metrics:      {metrics_line(protocol)}")
    print(f"device:       {protocol['device']}")
    print(f"metadata:     {len(plan.entries)} clips, read for the eligibility flags")
    print(
        f"mlr rows:     {len(plan.mlr)} clips with per-timestamp events"
        if plan.mlr
        else "mlr rows:     none, the MLR columns of the table stay empty"
    )
    print(f"pairs:        {len(plan.pairs)}")
    for spec in plan.pairs:
        source = str(spec.pred_dir) + ("" if spec.pred_dir.is_dir() else " (not found)")
        if spec.target_dir is None:
            target = "(not given)"
        else:
            target = str(spec.target_dir) + ("" if spec.target_dir.is_dir() else " (not found)")
        print(f"pair:         {spec.name}: {source}")
        print(f"paired with:  {target}")
        print(f"clips:        {len(spec.clips)} matched")
        for pred_path, _ in spec.clips[:PREVIEW]:
            print(f"clip:         {pred_path.name}")
        if len(spec.clips) > PREVIEW:
            print(f"clip:         ... and {len(spec.clips) - PREVIEW} more")
    print(f"output:       {plan.output or 'standard output'}")
    print(f"rows:         {plan.rows or 'none'}")
    print(f"dry run:      {'yes' if plan.dry_run else 'no'}")


def mlr_of(plan: EvalPlan, pred_path: Path) -> Any:
    """Per-timestamp MLR events of a clip and its eligibility, from the joined rows."""
    import numpy as np

    row = plan.mlr.get(pred_path.stem)
    if row is None:
        entry = plan.entries.get(pred_path.stem)
        return None, entry_eligible(entry)
    events = row.get("events")
    flags = np.asarray(events, dtype=bool) if events is not None else None
    eligible = bool(row.get("eligible", True))
    return flags, eligible


def score_pair(plan: EvalPlan, spec: PairSpec) -> list[dict[str, Any]]:
    """Score the clips of one arm, keeping the failures in the rows instead of raising."""
    from eveworld.evaluation.robotwin import evaluate_clip

    protocol = plan.protocol
    rows: list[dict[str, Any]] = []
    for pred_path, target_path in spec.clips:
        events, eligible = mlr_of(plan, pred_path)
        try:
            row = evaluate_clip(
                pred_path,
                target_path,
                metrics=protocol["metrics"],
                ssim_samples=int(protocol["ssim_samples"]),
                lpips_samples=int(protocol["lpips_samples"]),
                events=events,
                mlr_eligible=eligible,
                device=str(protocol["device"]),
                variant=spec.name,
            )
        except (OSError, ValueError, RuntimeError) as error:
            rows.append(
                {
                    "request_id": pred_path.stem,
                    "model": spec.name,
                    "variant": spec.name,
                    "pred_path": str(pred_path),
                    "target_path": str(target_path),
                    "protocol": PROTOCOL,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        row["request_id"] = pred_path.stem
        row["model"] = spec.name
        row["protocol"] = PROTOCOL
        rows.append(row)
    return rows


def run_evaluation(plan: EvalPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score the clips of the plan and return the table with the rows it was pooled from."""
    from eveworld.evaluation.robotwin import aggregate

    for spec in plan.pairs:
        if not spec.pred_dir.is_dir():
            raise FileNotFoundError(f"prediction directory {spec.pred_dir} does not exist")
        if spec.target_dir is None:
            raise FileNotFoundError(f"no reference directory for {spec.name}; pass --target-dir or --pair")
        if not spec.target_dir.is_dir():
            raise FileNotFoundError(f"reference directory {spec.target_dir} does not exist")
        if not spec.clips:
            raise FileNotFoundError(f"no clip of {spec.pred_dir} matches a reference recording under {spec.target_dir}")
    rows: list[dict[str, Any]] = []
    for spec in plan.pairs:
        rows.extend(score_pair(plan, spec))
    scored = [row for row in rows if "error" not in row]
    report = aggregate(scored, output=plan.output)
    errors = sorted(str(row["error"]) for row in rows if row.get("error"))
    report["arms"] = [spec.name for spec in plan.pairs]
    report["clips"] = len(rows)
    report["errors"] = errors
    return report, rows


def write_rows(plan: EvalPlan, rows: list[dict[str, Any]]) -> None:
    """Write the per-clip rows, when the run named a file for them."""
    from eveworld.utils.io import write_jsonl

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
        report, rows = run_evaluation(plan)
        write_rows(plan, rows)
        print(json.dumps(report, indent=2))
        return 0
    except ImportError as error:
        print(f"error: the eveworld package is not importable: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
