#!/usr/bin/env python3
"""Rebuild the columns of paper Table 4 from the official EWMBench result CSV.

The official EWMBench toolkit scores the generated AgiBot clips and merges the component metrics
into one CSV whose ``MEAN`` row holds the per-column means. This script locates that CSV under the
layout directory, rebuilds the ``MEAN`` row from the data rows with
:mod:`eveworld.evaluation.ewmbench` and writes the table as JSON::

    python scripts/evaluate/eval_ewmbench.py \
        --config configs/eval/ewmbench.yaml \
        --pred-dir eval_layout/eveworld_dataset \
        --output results/item_level/ewmbench/ewmbench_eveworld.json

``--scores`` names the CSV, or the directory holding it, when the layout is not the one the
toolkit wrote. Without it the script looks for ``final_results.csv``, ``results.csv`` and
``ewmbench.csv`` in ``--pred-dir`` and its parent, then for a CSV under ``--pred-dir`` whose first
non-comment row starts with a ``task_id`` header, up to two levels deep. ``--model`` sets the
model name the clip ids are built from and defaults to the name of ``--pred-dir`` without its
``_dataset`` suffix, the way the official toolkit names the clips.

The reported Motion, Semantics and Scene columns are the official aggregate scores, DYN, HSR,
nDTW and the components behind them come from the CSV, and the values stay in the raw units of
the file: paper Table 4 prints Motion multiplied by 100 and Semantics and Overall by 10.
``--output`` receives the table as JSON, ``--rows`` the per-clip records the summary was pooled
from, one JSON object per line, and ``--dry-run`` prints the plan and the located CSV without
reading the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROTOCOL = "ewmbench_table4_v1"
SCORE_NAMES = ("final_results.csv", "results.csv", "ewmbench.csv")
HEADER_KEY = "task_id"
SEARCH_DEPTH = 2
PREVIEW = 3
MODEL_SUFFIX = "_dataset"
DEFAULT_LAYOUT = "eval_layout"
DEFAULT_OUTPUT = "results/item_level/ewmbench/table4.json"
ROWS_SUFFIX = ".jsonl"

LAYOUT = """\
files written by a run:

    --rows      the per-clip records behind the summary, one JSON object per clip
    --output    the Table 4 columns, rebuilt from the data rows of the official CSV

The values stay in the raw units of the official CSV; paper Table 4 prints Motion multiplied by
100 and Semantics and Overall by 10. HSR and the CSV column hsd are the same hand-object score.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the EWMBench evaluation script."""
    parser = argparse.ArgumentParser(
        description="Rebuild the columns of paper Table 4 from an EWMBench result CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help="evaluation configuration, e.g. configs/eval/ewmbench.yaml (default: $EVEWORLD_CONFIG)",
    )
    run.add_argument(
        "--pred-dir",
        help="layout directory of the generated clips, e.g. eval_layout/eveworld_dataset",
    )
    run.add_argument("--scores", help="official result CSV, or the directory holding it")
    run.add_argument("--model", help="model name of the clip ids, e.g. eveworld")
    run.add_argument(
        "--output",
        help=f"summary of the table (default: {DEFAULT_OUTPUT})",
    )
    run.add_argument(
        "--rows",
        help="JSONL file of the per-clip records (default: --output with a .jsonl suffix)",
    )
    run.add_argument("--limit", type=int, help="score at most this many rows of the CSV")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the located CSV, without reading a score",
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


def model_of(layout: Path, named: str | None) -> str | None:
    """Model name of the clip ids: the named one, or the layout directory without its suffix."""
    if named:
        return named
    name = layout.name
    if name.endswith(MODEL_SUFFIX) and name != MODEL_SUFFIX:
        return name[: -len(MODEL_SUFFIX)]
    return None


def csv_starts_with_header(path: Path) -> bool:
    """Whether the first data row of ``path`` starts with a ``task_id`` column."""
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for cells in csv.reader(handle):
                first = cells[0].strip() if cells else ""
                if not first or first.startswith("#") or first.upper() == "MEAN":
                    continue
                return first.lower() == HEADER_KEY
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    return False


def search_directory(directory: Path) -> tuple[Path | None, list[Path]]:
    """Locate the official CSV in ``directory`` or its parent, else by its ``task_id`` header."""
    searched: list[Path] = []
    for base in (directory, directory.parent):
        for name in SCORE_NAMES:
            candidate = base / name
            searched.append(candidate)
            if candidate.is_file():
                return candidate, searched
    if directory.is_dir():
        for candidate in sorted(directory.rglob("*.csv")):
            relative = candidate.relative_to(directory)
            if len(relative.parts) > SEARCH_DEPTH or candidate in searched:
                continue
            searched.append(candidate)
            if csv_starts_with_header(candidate):
                return candidate, searched
    return None, searched


def find_scores(layout: Path, named: str | None, root: Path) -> tuple[Path | None, list[Path]]:
    """Locate the official result CSV of a run; returns it with every location that was looked at."""
    from eveworld.utils.config import resolve_path

    if named:
        target = resolve_path(named, root)
        if target.is_file():
            return target, [target]
        if target.is_dir():
            found, searched = search_directory(target)
            return found, [target, *searched]
        return None, [target]
    return search_directory(layout)


def count_rows(path: Path) -> int:
    """Number of data rows a run would read from the official CSV."""
    count = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for cells in csv.reader(handle):
            first = cells[0].strip() if cells else ""
            if not first or first.startswith("#") or first.upper() == "MEAN":
                continue
            count += 1
    return max(count - 1, 0)


@dataclass
class EvalPlan:
    """Everything a run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration, the stem of the default output directory.
        layout: Layout directory of the generated clips, the anchor of the CSV search.
        scores: Official result CSV, or ``None`` when the search did not find one.
        searched: Every location the search looked at, in the order it looked.
        model: Model name the clip ids are built from, or ``None`` to keep the CSV tail.
        limit: Number of CSV rows ``--limit`` caps the run at, or ``None`` for all of them.
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
    """Resolve the configuration, the layout and the result CSV into an :class:`EvalPlan`."""
    from eveworld.utils.config import resolve_path

    root = repo_root()
    config = load_config(config_path, [])
    if args.pred_dir:
        layout = resolve_path(args.pred_dir, root)
    else:
        layout = resolve_path(str(cfg_value(config, "eval.layout", DEFAULT_LAYOUT)), root)
    scores, searched = find_scores(layout, args.scores, root)
    output = resolve_path(args.output, root) if args.output else resolve_path(
        DEFAULT_OUTPUT, root
    )
    rows = resolve_path(args.rows, root) if args.rows else output.with_suffix(ROWS_SUFFIX)
    return EvalPlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        layout=layout,
        scores=scores,
        searched=searched,
        model=model_of(layout, args.model),
        limit=args.limit,
        output=output,
        rows=rows,
        command=" ".join(
            shlex.quote(part) for part in ("python", "scripts/evaluate/eval_ewmbench.py", *argv)
        ),
        dry_run=bool(args.dry_run),
    )


def print_plan(plan: EvalPlan) -> None:
    """Print the run, one aligned field per line."""
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(f"protocol:     {PROTOCOL}, Motion / Semantics / Scene of Table 4")
    found = "" if plan.layout.is_dir() else " (not found)"
    print(f"layout:       {plan.layout}{found}")
    if plan.scores is None:
        print(f"scores:       no result CSV in the {len(plan.searched)} locations searched")
        for path in plan.searched[:PREVIEW]:
            print(f"searched:     {path}")
        if len(plan.searched) > PREVIEW:
            print(f"searched:     ... and {len(plan.searched) - PREVIEW} more")
    else:
        print(f"scores:       {plan.scores}")
        print(f"csv rows:     {count_rows(plan.scores)}")
    print(f"model:        {plan.model or '(from the clip ids)'}")
    print(f"limit:        {plan.limit if plan.limit is not None else 'none'}")
    print(f"output:       {plan.output or 'standard output'}")
    print(f"rows:         {plan.rows or 'none'}")
    print(f"dry run:      {'yes' if plan.dry_run else 'no'}")


def run_evaluation(plan: EvalPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the official CSV and return the report of the table with the per-clip records."""
    from eveworld.evaluation.ewmbench import evaluate, load_scores

    if plan.scores is None:
        raise FileNotFoundError(
            f"no EWMBench result CSV found; looked at {len(plan.searched)} locations "
            f"starting from {plan.searched[0] if plan.searched else plan.layout}"
        )
    if not plan.scores.is_file():
        raise FileNotFoundError(f"the EWMBench result CSV {plan.scores} does not exist")
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
        "scores_csv": str(plan.scores),
        "model": summary.get("model") or plan.model,
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
    """Entry point: resolve the run, then read the CSV or dry-run it."""
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
