#!/usr/bin/env python3
"""Rebuild the columns of paper Table 3 from the WorldArena metric files.

The released WorldArena evaluator writes one JSON file per metric per model under
``<root>/<model>/core/<metric>.json`` and reports the eight core metrics with Overall as their
equal-weight mean. This script locates that root, reads it with
:mod:`eveworld.evaluation.worldarena` and writes the Table 3 columns as JSON::

    python scripts/evaluate/eval_worldarena.py \
        --config configs/eval/worldarena.yaml \
        --pred-dir outputs/worldarena_eveworld/generated_only \
        --output outputs/evaluation/item_level/worldarena/table3.json

``--scores`` names the evaluation root, one model directory under it or a single metric file when
the default does not find them. Without it the script looks for metric files in ``--pred-dir``,
its parent, and their ``evaluation``, ``scores``, ``metrics`` and ``core`` subdirectories, and
takes the first location that holds them. ``--model`` selects one model directory of the root the
way the adapter names it; without it every model directory holding metric files is read.

The table is the eight metrics and Overall on the 0-100 scale of the paper, the same numbers the
released evaluator prints. MLR is reported beside this table rather than in it and is computed by
``scripts/evaluate/eval_mlr.py``. ``--output`` receives the table as JSON, ``--rows`` the per-clip
records behind it, one JSON object per line, ``--limit`` caps the clips for a smoke run and
``--dry-run`` prints the plan and the metric files the run would read, without reading one.
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

PROTOCOL = "worldarena_table3_v1"
CORE_METRICS = (
    "image_quality",
    "aesthetic_quality",
    "dynamic_degree",
    "flow_score",
    "motion_smoothness",
    "subject_consistency",
    "background_consistency",
    "photometric_smoothness",
)
CANDIDATE_SUBDIRS = ("evaluation", "scores", "metrics", "core")
PREVIEW = 3
ROWS_SUFFIX = ".jsonl"
DEFAULT_OUTPUT = "outputs/evaluation/item_level/worldarena/table3.json"
UNKNOWN = "unknown"

LAYOUT = """\
files written by a run:

    --rows      the per-clip records behind the table, one JSON object per clip
    --output    the Table 3 columns, on the 0-100 scale of the paper

The evaluator scores the eight core metrics on [0, 1] and reports Overall as their equal-weight
mean; the table multiplies every value by 100, the scale of paper Table 3. MLR is reported beside
the table and is computed by scripts/evaluate/eval_mlr.py.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the WorldArena evaluation script."""
    parser = argparse.ArgumentParser(
        description="Rebuild the columns of paper Table 3 from the WorldArena metric files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help="evaluation configuration, e.g. configs/eval/worldarena.yaml (default: $EVEWORLD_CONFIG)",
    )
    run.add_argument(
        "--pred-dir",
        help="layout directory of the generated clips, e.g. outputs/worldarena_eveworld/generated_only",
    )
    run.add_argument("--scores", help="evaluation root, model directory or single metric file")
    run.add_argument("--model", help="model directory under the root, e.g. eveworld")
    run.add_argument(
        "--output",
        help=f"summary of the table (default: {DEFAULT_OUTPUT})",
    )
    run.add_argument(
        "--rows",
        help="JSONL file of the per-clip records (default: --output with a .jsonl suffix)",
    )
    run.add_argument("--limit", type=int, help="score at most this many clips")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the metric files the run would read, without reading one",
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


def metric_dir(directory: Path) -> Path:
    """The directory holding the ``<metric>.json`` files of one model."""
    core = directory / "core"
    return core if core.is_dir() else directory


def metric_files(directory: Path) -> list[str]:
    """The core metric files a metrics directory holds, in the order of the table."""
    return [name for name in CORE_METRICS if (directory / f"{name}.json").is_file()]


def holds_metrics(directory: Path, model: str | None = None) -> bool:
    """Whether the adapter would read metric files under ``directory``."""
    if model:
        return bool(metric_files(metric_dir(directory / model)))
    if metric_files(metric_dir(directory)):
        return True
    if not directory.is_dir():
        return False
    return any(metric_files(metric_dir(child)) for child in directory.iterdir() if child.is_dir())


def model_name(metrics_dir: Path, root: Path) -> str:
    """Name the model a metrics directory belongs to, or ``unknown`` when the layout says nothing."""
    if metrics_dir.name == "core":
        return metrics_dir.parent.name
    if metrics_dir != root:
        return metrics_dir.name
    return root.name if root.is_dir() else UNKNOWN


def model_dirs(source: Path, model: str | None) -> list[tuple[str, Path]]:
    """The ``(model, metrics directory)`` pairs a run reads, in the order the adapter reads them."""
    if source.is_file():
        return [(model or source.parent.name, metric_dir(source.parent))]
    if model:
        target = source / model
        return [(model, metric_dir(target))] if metric_files(metric_dir(target)) else []
    if not source.is_dir():
        return []
    if metric_files(metric_dir(source)):
        metrics_dir = metric_dir(source)
        return [(model_name(metrics_dir, source), metrics_dir)]
    names = sorted(child.name for child in source.iterdir() if child.is_dir() and metric_files(metric_dir(child)))
    return [(name, metric_dir(source / name)) for name in names]


def expand_target(target: Path) -> list[Path]:
    """The target itself, followed by the subdirectories a run looks for metric files in."""
    if target.is_file():
        return [target]
    return [target, *(target / name for name in CANDIDATE_SUBDIRS)]


def find_scores(
    layout: Path,
    named: str | None,
    model: str | None,
    root: Path,
) -> tuple[Path | None, list[Path]]:
    """Locate the WorldArena metric files; returns them with every location that was looked at."""
    from eveworld.utils.config import resolve_path

    if named:
        target = resolve_path(named, root)
        if target.is_file():
            return target, [target]
        searched = expand_target(target)
        for candidate in searched:
            if holds_metrics(candidate, model):
                return candidate, searched
        return target, searched
    searched: list[Path] = []
    for base in (layout, layout.parent):
        for candidate in expand_target(base):
            if candidate in searched:
                continue
            searched.append(candidate)
            if holds_metrics(candidate, model):
                return candidate, searched
    return None, searched


@dataclass
class EvalPlan:
    """Everything a run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration, the stem of the default output directory.
        layout: Layout directory of the generated clips, the anchor of the metric search.
        scores: Evaluation root, model directory or metric file, or ``None`` when the search
            did not find one.
        searched: Every location the search looked at, in the order it looked.
        model: Model directory to read under the root, or ``None`` to read every model.
        limit: Number of clips ``--limit`` caps the run at, or ``None`` for all of them.
        output: JSON summary of the table, or ``None`` to only print it.
        rows: JSONL file the per-clip records are written to, or ``None`` for no row file.
        command: Command line of the run, quoted, as recorded in the report.
        dry_run: Whether the run stops after the plan.
    """

    config_path: Path
    config: Any
    name: str
    layout: Path
    scores: Path | None
    searched: list[Path]
    model: str | None
    limit: int | None
    output: Path | None
    rows: Path | None
    command: str
    dry_run: bool


def build_plan(args: argparse.Namespace, config_path: Path, argv: Sequence[str]) -> EvalPlan:
    """Resolve the configuration, the layout and the metric files into an :class:`EvalPlan`."""
    from eveworld.utils.config import resolve_path

    root = repo_root()
    config = load_config(config_path, [])
    if args.pred_dir:
        layout = resolve_path(args.pred_dir, root)
    else:
        layout = resolve_path(str(cfg_value(config, "output_dir", "outputs")), root)
    scores, searched = find_scores(layout, args.scores, args.model, root)
    output = resolve_path(args.output or DEFAULT_OUTPUT, root)
    rows = resolve_path(args.rows, root) if args.rows else output.with_suffix(ROWS_SUFFIX)
    return EvalPlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        layout=layout,
        scores=scores,
        searched=searched,
        model=args.model,
        limit=args.limit,
        output=output,
        rows=rows,
        command=" ".join(shlex.quote(part) for part in ("python", "scripts/evaluate/eval_worldarena.py", *argv)),
        dry_run=bool(args.dry_run),
    )


def print_plan(plan: EvalPlan) -> None:
    """Print the run, one aligned field per line."""
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(f"protocol:     {PROTOCOL}, the eight core metrics and Overall of Table 3")
    found = "" if plan.layout.is_dir() else " (not found)"
    print(f"layout:       {plan.layout}{found}")
    if plan.scores is None:
        print(f"scores:       no metric files in the {len(plan.searched)} locations searched")
        for path in plan.searched[:PREVIEW]:
            print(f"searched:     {path}")
        if len(plan.searched) > PREVIEW:
            print(f"searched:     ... and {len(plan.searched) - PREVIEW} more")
    else:
        print(f"scores:       {plan.scores}")
        pairs = model_dirs(plan.scores, plan.model)
        for name, directory in pairs:
            present = metric_files(directory)
            detail = f"{len(present)}/{len(CORE_METRICS)} metrics"
            missing = [metric for metric in CORE_METRICS if metric not in present]
            if missing:
                detail += f", missing {', '.join(missing)}"
            print(f"model dir:    {name} -> {directory} ({detail})")
        if not pairs:
            print("model dir:    no metric files at this location")
    print(f"model:        {plan.model or '(every model directory)'}")
    print(f"limit:        {plan.limit if plan.limit is not None else 'none'}")
    print(f"output:       {plan.output or 'standard output'}")
    print(f"rows:         {plan.rows or 'none'}")
    print(f"dry run:      {'yes' if plan.dry_run else 'no'}")


def run_evaluation(plan: EvalPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the metric files and return the report of the table with the per-clip records."""
    from eveworld.evaluation.worldarena import evaluate, load_scores

    if plan.scores is None:
        raise FileNotFoundError(
            f"no WorldArena metric files found; looked at {len(plan.searched)} locations "
            f"starting from {plan.searched[0] if plan.searched else plan.layout}"
        )
    payload = load_scores(plan.scores, model=plan.model)
    records = payload["records"]
    if plan.limit is not None:
        records = records[: int(plan.limit)]
    if not records:
        raise ValueError(f"no scored clips in {plan.scores}")
    result = evaluate(records)
    summary = result["scores"]
    report = {
        "config": str(plan.config_path),
        "name": plan.name,
        "protocol": PROTOCOL,
        "command": plan.command,
        "pred_dir": str(plan.layout),
        "scores_root": str(plan.scores),
        "model": summary.get("model") or plan.model,
        "models": summary.get("models"),
        "records": len(result["rows"]),
        "scores": summary,
    }
    return report, result["rows"]


def write_results(plan: EvalPlan, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the summary and the per-clip records, when the run named files for them."""
    from eveworld.utils.io import write_json, write_jsonl

    if plan.output is not None:
        write_json(report, plan.output)
    if plan.rows is not None:
        write_jsonl(rows, plan.rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: resolve the run, then read the metric files or dry-run it."""
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
            print("dry run: no score read")
            return 0
        report, rows = run_evaluation(plan)
        write_results(plan, report, rows)
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
