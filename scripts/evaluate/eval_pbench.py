#!/usr/bin/env python3
"""Rebuild the horizon sweep of the PBench physical-commonsense benchmark.

The generation sweep writes one prediction directory per horizon tier and this script scores each
tier with :mod:`eveworld.evaluation.pbench`, then averages the tiers into the Domain, Phys., Space
and Time columns of the paper's horizon table::

    python scripts/evaluate/eval_pbench.py \
        --config configs/eval/pbench.yaml \
        --pred-dir outputs/pbench_eveworld \
        --output outputs/pbench_eveworld/horizon.json

``--pred-dir`` is the sweep root: every directory under it that is named after a tier or holds a
PBench score file is one tier. The name carries the tier either as a frame count or as the
duration in seconds at 16 FPS, so ``61``, ``frames_61`` and ``3.8s`` all name the 61-frame tier.
``--scores`` names one score file, or the sweep root, when the default search does not find it,
and ``--scores-name`` overrides the file name probed inside a tier. A tier is probed for the
standard score file names first and, when none of them is there, for the first JSON or JSONL file
of the tier whose first record looks like PBench data.

A tier with no score file is listed in the report and left out of the table, and the run stops
only when no tier of the sweep has one. ``--weighting`` chooses the released per-clip estimator
(``sample``) or question pooling. ``--output`` receives one block of scores per tier plus the
means the paper reports, ``--rows`` the per-question records of every tier with their tier and
protocol, one JSON object per line, ``--limit`` caps the questions a tier is scored on and
``--dry-run`` prints the plan and the score file of every tier without scoring one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL = "pbench_horizon_v1"
FPS = 16.0
TIER_FRAMES = (61, 93, 157, 253, 317, 413, 477)
REPORTED_TIERS = (61, 93, 157, 253, 317)
LONG_TIERS = (157, 253, 317)
SCORE_NAMES = (
    "scores.jsonl",
    "scores.json",
    "pbench.jsonl",
    "pbench.json",
    "results.jsonl",
    "results.json",
    "predictions.jsonl",
    "predictions.json",
)
PROBE_DIRS = ("evaluation", "scores", "eval", "results")
SNIFF_SUFFIXES = (".jsonl", ".json")
DURATION_IN_NAME = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*s(?:ec)?(?![a-z])", re.IGNORECASE)
PREVIEW = 3
ROWS_SUFFIX = ".jsonl"
DEFAULT_OUTPUT_NAME = "horizon.json"
UNSORTED = 10**6

LAYOUT = """\
files written by a run:

    --rows      the per-question records of every tier, one JSON object per question
    --output    one block of scores per tier plus the means over the tiers

A tier with no score file is listed in the report and left out of the table; the run stops with an
error only when no tier of the sweep has one. The means are the two rows of the paper:
``five_shortest`` covers the five tiers from 3.8 s to 19.8 s (61 to 317 frames) and
``three_longest`` the three longest of them, 9.8 s to 19.8 s. ``all_tiers`` averages every tier
the sweep holds and is a diagnostic beside the paper's rows, not one of them.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the PBench evaluation script."""
    parser = argparse.ArgumentParser(
        description="Rebuild the horizon table of the PBench benchmark from a tier sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help="evaluation configuration, e.g. configs/eval/pbench.yaml (default: $EVEWORLD_CONFIG)",
    )
    run.add_argument(
        "--pred-dir",
        help="sweep root of the generated clips, e.g. outputs/pbench_eveworld",
    )
    run.add_argument("--scores", help="one score file, or the sweep root holding the tiers")
    run.add_argument("--scores-name", help="score file name probed inside a tier directory")
    run.add_argument(
        "--weighting",
        default="sample",
        choices=["sample", "question"],
        help="sample averages the per-clip accuracies, question pools all questions (default: sample)",
    )
    run.add_argument(
        "--output",
        help=f"summary of the horizon table (default: <config output_dir>/{DEFAULT_OUTPUT_NAME})",
    )
    run.add_argument(
        "--rows",
        help="JSONL file of the per-question records (default: --output with a .jsonl suffix)",
    )
    run.add_argument("--limit", type=int, help="score at most this many questions per tier")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the score file of every tier, without scoring one",
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


def duration_of(frames: int | None) -> float | None:
    """Duration of a tier in seconds, or ``None`` when its frame count is unknown."""
    return None if frames is None else round(frames / FPS, 1)


def parse_frames(name: str) -> int | None:
    """Tier frame count of a directory name, from a digit run or from a duration in seconds."""
    for run in re.findall(r"\d+", name):
        if int(run) in TIER_FRAMES:
            return int(run)
    for match in DURATION_IN_NAME.finditer(name):
        frames = int(round(float(match.group(1)) * FPS))
        if frames in TIER_FRAMES:
            return frames
    return None


def probe_dirs(tier: Path) -> list[Path]:
    """The tier directory itself and the subdirectories a score file is probed in."""
    return [tier, *(tier / name for name in PROBE_DIRS)]


def standard_score(tier: Path, scores_name: str | None) -> tuple[Path | None, list[Path]]:
    """Locate a score file of ``tier`` by name; returns it with every location that was looked at."""
    searched: list[Path] = []
    names = (scores_name,) if scores_name else SCORE_NAMES
    for directory in probe_dirs(tier):
        for name in names:
            candidate = directory / name
            searched.append(candidate)
            if candidate.is_file():
                return candidate, searched
    return None, searched


def as_mapping(value: Any) -> dict[str, Any] | None:
    """``value`` as a plain dict when it is a mapping, else ``None``."""
    return dict(value) if isinstance(value, Mapping) else None


def first_record(path: Path) -> dict[str, Any] | None:
    """The first record of a JSON or JSONL file, or ``None`` when the file holds no record."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    return as_mapping(json.loads(text))
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return as_mapping(payload[0]) if payload else None
    record = as_mapping(payload)
    if record is None:
        return None
    inner = record.get("records") or record.get("samples")
    if isinstance(inner, list):
        return as_mapping(inner[0]) if inner else None
    return record


def is_pbench_record(record: Any) -> bool:
    """Whether ``record`` looks like a PBench clip record or a PBench question record."""
    if not isinstance(record, Mapping):
        return False
    if isinstance(record.get("qa_pairs"), list):
        return True
    return "question" in record and any(key in record for key in ("gold_answer", "answer", "pred_answer"))


def looks_like_pbench(path: Path) -> bool:
    """Whether the first record of ``path`` carries PBench question fields."""
    try:
        record = first_record(path)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return is_pbench_record(record)


def sniff_score(tier: Path) -> tuple[Path | None, list[Path]]:
    """Locate a score file by looking at the first record of the JSON files of a tier."""
    searched: list[Path] = []
    for directory in probe_dirs(tier):
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() not in SNIFF_SUFFIXES:
                continue
            searched.append(candidate)
            if looks_like_pbench(candidate):
                return candidate, searched
    return None, searched


def tier_score(tier: Path, scores_name: str | None) -> tuple[Path | None, list[Path]]:
    """Locate the score file of one tier, by name first and by sniffing its JSON files after."""
    found, searched = standard_score(tier, scores_name)
    if found is not None or scores_name:
        return found, searched
    sniffed, extra = sniff_score(tier)
    searched.extend(path for path in extra if path not in searched)
    return sniffed, searched


@dataclass
class Tier:
    """One horizon tier of the sweep and the score file it is read from.

    Attributes:
        directory: Directory of the tier under the sweep root.
        frames: Frame count the tier directory name carries, or ``None`` when it carries none.
        score: Score file of the tier, or ``None`` when the tier holds no PBench data.
        searched: Every location the tier was probed at, in the order it was probed.
    """

    directory: Path
    frames: int | None
    score: Path | None
    searched: list[Path]


def tier_key(tier: Tier) -> str:
    """Report key of a tier: its frame count, or the directory name when it carries none."""
    return str(tier.frames) if tier.frames is not None else tier.directory.name


def tier_label(tier: Tier) -> str:
    """Human label of a tier: its frames and duration, or the directory name."""
    if tier.frames is None:
        return f"{tier.directory.name} (frames unknown)"
    return f"{tier.frames} frames ({duration_of(tier.frames)} s)"


def sort_tiers(tiers: Sequence[Tier]) -> list[Tier]:
    """Tiers in frame order, with the tiers of an unknown frame count last."""
    return sorted(tiers, key=lambda tier: (tier.frames if tier.frames is not None else UNSORTED, tier.directory.name))


def resolve_tiers(sweep: Path, scores_name: str | None) -> tuple[list[Tier], list[Path]]:
    """Tier directories of a sweep root with the score file of each, and every location probed.

    A directory under the root is a tier when its name carries a tier frame count or it holds a
    PBench score file; a root that holds a score file directly and has no tier directory is a
    single tier of its own. An empty tier list means the sweep root holds no PBench data.
    """
    tiers: list[Tier] = []
    searched: list[Path] = []
    if not sweep.is_dir():
        return [], list(standard_score(sweep, scores_name)[1])
    for child in sorted(sweep.iterdir()):
        if not child.is_dir():
            continue
        frames = parse_frames(child.name)
        score, probed = tier_score(child, scores_name)
        searched.extend(path for path in probed if path not in searched)
        if frames is not None or score is not None:
            tiers.append(Tier(child, frames, score, probed))
    if tiers:
        return sort_tiers(tiers), searched
    score, probed = tier_score(sweep, scores_name)
    searched.extend(path for path in probed if path not in searched)
    if score is not None:
        tiers.append(Tier(sweep, parse_frames(sweep.name), score, probed))
    return sort_tiers(tiers), searched


@dataclass
class EvalPlan:
    """Everything a run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration.
        layout: Layout directory of the generated clips, as given or configured.
        sweep: Directory the tier sweep was searched in.
        tiers: Horizon tiers of the sweep, in frame order, with the score file of each.
        searched: Every location the sweep was probed at, in the order it was probed.
        weighting: ``sample`` or ``question``, the estimator the tiers are scored with.
        limit: Number of questions ``--limit`` caps a tier at, or ``None`` for all of them.
        output: JSON summary of the horizon table.
        rows: JSONL file the per-question records are written to, or ``None`` for no row file.
        command: Command line of the run, quoted, as recorded in the report.
        dry_run: Whether the run stops after the plan.
    """

    config_path: Path
    config: Any
    name: str
    layout: Path
    sweep: Path
    tiers: list[Tier]
    searched: list[Path]
    weighting: str
    limit: int | None
    output: Path
    rows: Path
    command: str
    dry_run: bool


def build_plan(args: argparse.Namespace, config_path: Path, argv: Sequence[str]) -> EvalPlan:
    """Resolve the configuration, the sweep root and the tier score files into an :class:`EvalPlan`."""
    from eveworld.utils.config import resolve_path

    root = repo_root()
    config = load_config(config_path, [])
    scratch = resolve_path(str(cfg_value(config, "output_dir", "outputs")), root)
    layout = resolve_path(args.pred_dir, root) if args.pred_dir else scratch
    if args.scores:
        target = resolve_path(args.scores, root)
        if target.is_file():
            sweep = target.parent
            searched = [target]
            tiers = [Tier(sweep, parse_frames(sweep.name), target, [target])]
        else:
            sweep = target
            tiers, searched = resolve_tiers(sweep, args.scores_name)
        if not args.pred_dir:
            layout = sweep
    else:
        sweep = layout
        tiers, searched = resolve_tiers(sweep, args.scores_name)
    output = resolve_path(args.output, root) if args.output else scratch / DEFAULT_OUTPUT_NAME
    rows = resolve_path(args.rows, root) if args.rows else output.with_suffix(ROWS_SUFFIX)
    return EvalPlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        layout=layout,
        sweep=sweep,
        tiers=tiers,
        searched=searched,
        weighting=args.weighting,
        limit=args.limit,
        output=output,
        rows=rows,
        command=" ".join(shlex.quote(part) for part in ("python", "scripts/evaluate/eval_pbench.py", *argv)),
        dry_run=bool(args.dry_run),
    )


def print_plan(plan: EvalPlan) -> None:
    """Print the run, one aligned field per line."""
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(f"protocol:     {PROTOCOL}, Domain / Phys. / Space / Time per tier at {FPS} FPS")
    found = "" if plan.sweep.is_dir() else " (not found)"
    print(f"sweep:        {plan.sweep}{found}")
    if not plan.tiers:
        print("tier:         no tier directory with a score file")
    for tier in plan.tiers:
        if tier.score is None:
            print(f"tier:         {tier_label(tier)} -> no score file under {tier.directory}")
        else:
            print(f"tier:         {tier_label(tier)} -> {tier.score}")
    missing = [tier for tier in plan.tiers if tier.score is None]
    if missing and any(tier.score is not None for tier in plan.tiers):
        frames = ", ".join(str(tier.frames) for tier in missing if tier.frames is not None)
        detail = f" ({frames})" if frames else ""
        print(f"missing:      {len(missing)} of {len(plan.tiers)} tiers have no score file{detail}")
    if not any(tier.score is not None for tier in plan.tiers):
        print(f"searched:     no score file in the {len(plan.searched)} locations probed")
        for path in plan.searched[:PREVIEW]:
            print(f"searched:     {path}")
        if len(plan.searched) > PREVIEW:
            print(f"searched:     ... and {len(plan.searched) - PREVIEW} more")
    print(f"weighting:    {plan.weighting}")
    print(f"limit:        {plan.limit if plan.limit is not None else 'none'}")
    print(f"output:       {plan.output}")
    print(f"rows:         {plan.rows}")
    print(f"dry run:      {'yes' if plan.dry_run else 'no'}")


def mean_block(scored: Mapping[str, Any], frames: Sequence[int]) -> dict[str, Any]:
    """Mean of each metric over the given tiers the run scored, and the tiers that were pooled."""
    from eveworld.evaluation.pbench.evaluate import METRICS

    present = [frame for frame in frames if str(frame) in scored]
    block: dict[str, Any] = {"tiers": present}
    for metric in METRICS:
        values = [float(scored[str(frame)]["scores"][metric]) for frame in present if scored[str(frame)]["scores"].get(metric) is not None]
        block[metric] = round(sum(values) / len(values), 2) if values else None
    return block


def run_evaluation(plan: EvalPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every tier that holds a score file; returns the horizon table with the scored rows."""
    from eveworld.evaluation.pbench import evaluate, load_scores

    if not any(tier.score is not None for tier in plan.tiers):
        first = plan.searched[0] if plan.searched else plan.sweep
        raise FileNotFoundError(f"no PBench score file found; looked at {len(plan.searched)} locations " f"starting from {first}")
    scored: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for tier in plan.tiers:
        if tier.score is None:
            continue
        payload = load_scores(tier.score)
        records = payload["records"]
        if plan.limit is not None:
            records = records[: int(plan.limit)]
        if not records:
            raise ValueError(f"no questions in {tier.score}")
        result = evaluate(records, weighting=plan.weighting)
        key = tier_key(tier)
        scored[key] = {
            "frames": tier.frames,
            "duration": duration_of(tier.frames),
            "path": str(tier.score),
            "scores": result["scores"],
        }
        rows.extend({**row, "tier": key, "protocol": PROTOCOL} for row in result["rows"])
    report = {
        "config": str(plan.config_path),
        "name": plan.name,
        "protocol": PROTOCOL,
        "command": plan.command,
        "pred_dir": str(plan.layout),
        "sweep_root": str(plan.sweep),
        "fps": FPS,
        "weighting": plan.weighting,
        "records": len(rows),
        "tiers": scored,
        "missing_tiers": [{"frames": tier.frames, "dir": str(tier.directory)} for tier in plan.tiers if tier.score is None],
        "means": {
            "five_shortest": mean_block(scored, REPORTED_TIERS),
            "three_longest": mean_block(scored, LONG_TIERS),
            "all_tiers": mean_block(scored, TIER_FRAMES),
        },
    }
    return report, rows


def write_results(plan: EvalPlan, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the horizon table and the per-question records of every tier."""
    from eveworld.utils.io import write_json, write_jsonl

    write_json(report, plan.output)
    write_jsonl(rows, plan.rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: resolve the run, then score every tier or dry-run it."""
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
