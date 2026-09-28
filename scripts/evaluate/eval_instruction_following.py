#!/usr/bin/env python3
"""Score generated clips with the instruction-following judges of Table 1 and Table 6.

The script resolves the judging protocol of ``--config``, asks the judge named by
``eval.metrics`` over the clips of ``--pred-dir`` and writes the aggregate to ``--output``::

    python scripts/evaluate/eval_instruction_following.py \
        --config configs/eval/instruction_following/qwen_if.yaml \
        --pred-dir outputs/dreamgen_eveworld/generated_only \
        --output results/item_level/dreamgen/qwen_if_eveworld.json

Every clip is decoded to ``eval.frame_count`` frames whose longest side is at most
``eval.max_side`` pixels, JPEG-encoded at quality 85 and sent to the judge together with the
instruction of the clip. The judge answers one binary question - does the clip follow the
instruction - at ``eval.temperature`` with a budget of ``eval.max_tokens`` tokens, ``eval.retries``
attempts and a ``eval.timeout`` second deadline per request. ``--judge qwen_if`` talks to the
OpenAI-compatible endpoint of ``eval.base_url`` with ``$DASHSCOPE_API_KEY`` or
``$OPENAI_API_KEY``; ``--judge gemini_if`` talks to the endpoint of ``$GEMINI_BASE_URL`` /
``$DIFROST_GENAI_BASE_URL`` with ``$DIFROST_API_TOKEN``. The judge class of the library judges the
records one after another, so the ``eval.concurrency`` of the configuration is recorded in the
audit block but not applied.

The instruction of a clip is read from ``--metadata``: the stem of a clip keys its entry and the
``prompt``, ``instruction`` or ``question`` of the entry is the instruction. Without such an entry
the clip name is read as ``<index>_<instruction with underscores>``. A clip whose entry carries a
category, a split or a task type is counted under it, so a benchmark that publishes its
Environment / Object / Behavior split gets the breakdown of the released table.

``--output`` receives the summary and ``--rows`` the per-clip rows, one JSON object per line in
the schema of ``results/item_level/README.md``; ``--rows`` defaults to ``--output`` with a
``.jsonl`` suffix. ``--resume`` reads the rows of an earlier run back, skips the clips they
decided and carries those rows into the new row file, while a clip whose earlier judgement errored
is tried again; that is how the release ran a judge over a benchmark in several sittings.
``--limit`` caps the number of clips for a smoke run, and ``--dry-run`` prints the plan and the
resolved clips without calling a judge.
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
ROWS_SUFFIX = ".jsonl"
PREVIEW = 3
PROTOCOL = "instruction_following_v1"
QWEN_JUDGE = "qwen_if"
GEMINI_JUDGE = "gemini_if"
JUDGES = (QWEN_JUDGE, GEMINI_JUDGE)

# The frozen protocol of the released table, used when the configuration does not carry a setting.
DEFAULT_FRAME_COUNT = 49
DEFAULT_MAX_SIDE = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 32000
DEFAULT_TIMEOUT = 600.0
DEFAULT_RETRIES = 3
DEFAULT_JPEG_QUALITY = 85
DEFAULT_METADATA = "data/metadata/dreamgenbench"
DEFAULT_JUDGE_MODEL = {QWEN_JUDGE: "qwen-plus", GEMINI_JUDGE: "gemini-2.5-flash"}

# The instruction keys of a metadata entry, in the order the library reads them.
INSTRUCTION_KEYS = ("prompt", "instruction", "question")
CATEGORY_KEYS = ("category", "split", "task_type", "generalization", "task")

LAYOUT = """\
files written by a run:

    --output    the aggregate summary, a JSON object
    --rows      the rows it was computed from, one JSON object per clip

A row carries request_id, model and the judge key of the run - qwen_if or gemini_if, 100.0 for a
followed instruction, 0.0 for a broken one and null for an undecided clip - next to the verdict
and the error of the clip, as documented in results/item_level/README.md.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the instruction-following evaluation script."""
    parser = argparse.ArgumentParser(
        description="Score generated clips with the instruction-following judges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help=("evaluation configuration, e.g. configs/eval/instruction_following/qwen_if.yaml " "(default: $EVEWORLD_CONFIG)"),
    )
    run.add_argument(
        "--pred-dir",
        help="directory of the generated clips, or the layout root holding generated_only/",
    )
    run.add_argument("--metadata", help="directory (or file) of the per-clip instructions")
    run.add_argument("--output", help="JSON file the aggregate summary is written to")
    run.add_argument(
        "--rows",
        help="JSONL file of the per-clip rows (default: --output with a .jsonl suffix)",
    )
    run.add_argument("--limit", type=int, help="judge at most this many clips")
    run.add_argument(
        "--model",
        help="arm that generated the clips, e.g. eveworld (default: the directory name)",
    )
    protocol = parser.add_argument_group("protocol")
    protocol.add_argument(
        "--judge",
        choices=JUDGES,
        help="judge to ask (default: the judge eval.metrics of the configuration names)",
    )
    protocol.add_argument("--judge-model", help="model name of the judge, e.g. qwen-plus")
    protocol.add_argument("--frame-count", type=int, help="frames sent to the judge per clip")
    protocol.add_argument("--max-side", type=int, help="longest side of those frames in pixels")
    protocol.add_argument(
        "--resume",
        action="store_true",
        help="skip the clips the rows of an earlier run already decided",
    )
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the resolved clips, without calling a judge",
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


def judge_of(config: Any, named: str | None) -> str:
    """Judge of the run: the named one, or the judge ``eval.metrics`` of the configuration names."""
    if named:
        return str(named)
    metrics = cfg_value(config, "eval.metrics", ())
    if isinstance(metrics, str):
        metrics = [metrics]
    for name in metrics or ():
        text = str(name).strip().lower()
        if text in JUDGES:
            return text
    return QWEN_JUDGE


def protocol_settings(config: Any, args: argparse.Namespace, judge: str) -> dict[str, Any]:
    """The judging protocol of the run, with the command-line overrides applied.

    A setting the configuration does not carry falls back to the frozen protocol of the released
    table, so the audit block always names the frames, the budget and the retries of the run.
    """
    settings: dict[str, Any] = {
        "protocol": PROTOCOL,
        "judge": judge,
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "frame_count": int(args.frame_count if args.frame_count is not None else cfg_value(config, "eval.frame_count", DEFAULT_FRAME_COUNT)),
        "max_side": int(args.max_side if args.max_side is not None else cfg_value(config, "eval.max_side", DEFAULT_MAX_SIDE)),
        "temperature": float(cfg_value(config, "eval.temperature", DEFAULT_TEMPERATURE)),
        "max_tokens": int(cfg_value(config, "eval.max_tokens", DEFAULT_MAX_TOKENS)),
        "timeout": float(cfg_value(config, "eval.timeout", DEFAULT_TIMEOUT)),
        "retries": int(cfg_value(config, "eval.retries", DEFAULT_RETRIES)),
        "model": str(args.judge_model or cfg_value(config, "eval.model", DEFAULT_JUDGE_MODEL.get(judge, ""))),
        "concurrency": int(cfg_value(config, "eval.concurrency", 1)),
        "disable_thinking": bool(cfg_value(config, "eval.disable_thinking", True)),
    }
    for key, name in (
        ("eval.base_url", "base_url"),
        ("eval.env_endpoint", "env_endpoint"),
        ("eval.thinking_level", "thinking_level"),
        ("data.prompts", "prompt_template"),
        ("data.metadata", "metadata"),
    ):
        value = cfg_value(config, key)
        if value is not None:
            settings[name] = value
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
        (path for path in pred_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: path.name,
    )
    return clips if limit is None else clips[: int(limit)]


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


def entry_instruction(entry: Any) -> str | None:
    """Instruction a metadata entry carries, when it carries one."""
    if not isinstance(entry, dict):
        return None
    for key in INSTRUCTION_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def entry_category(entry: Any) -> str | None:
    """Category, split or task type a metadata entry carries, when it carries one."""
    if not isinstance(entry, dict):
        return None
    for key in CATEGORY_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def prompt_from_path(path: Path) -> str:
    """Recover the instruction from a clip name such as ``0007_pick_up_the_cup.mp4``."""
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return stem.replace("_", " ")


def record_of(plan: EvalPlan, path: Path) -> dict[str, Any]:
    """Judging record of one clip: its instruction, its category and the arm it came from."""
    entry = plan.entries.get(path.stem)
    instruction = entry_instruction(entry)
    record: dict[str, Any] = {
        "request_id": path.stem,
        "model": plan.model,
        "video": str(path),
        "question": instruction or prompt_from_path(path),
    }
    category = entry_category(entry)
    if category:
        record["category"] = category
    return record


def read_rows(path: Path | None) -> list[dict[str, Any]]:
    """Decided rows of an earlier run, empty when the run wrote none or named no file.

    A row whose judge call errored or answered ambiguously carries no verdict and is left out, so
    a resumed run tries that clip again instead of carrying the undecided row into the new file.
    """
    from eveworld.utils.io import read_jsonl

    if path is None or not path.is_file():
        return []
    return [dict(row) for row in read_jsonl(path) if row.get("verdict") is not None]


@dataclass
class EvalPlan:
    """Everything a run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration, the stem of the default output directory.
        judge: Judge of the run, ``qwen_if`` or ``gemini_if``.
        protocol: The judging protocol of the run, i.e. the settings every clip is judged with.
        pred_dir: Directory the generated clips are read from.
        clips: Clips selected for judging, in name order.
        records: Records to judge, i.e. the clips the resume file did not already decide.
        resumed: Rows of an earlier run, carried into the new row file unchanged.
        limit: Number of clips ``--limit`` caps the run at, or ``None`` for all of them.
        model: Arm the clips were generated by, as recorded on every row.
        metadata: Directory or file the per-clip instructions are read from.
        entries: Metadata entries keyed by request id, empty when there is no metadata file.
        output: JSON file the aggregate is written to, or ``None`` for the standard output only.
        rows: JSONL file the per-clip rows are written to, or ``None`` for no row file.
        command: Command line of the run, quoted, as recorded in the summary.
        dry_run: Whether the run stops after the plan.
    """

    config_path: Path
    config: Any
    name: str
    judge: str
    protocol: dict[str, Any]
    pred_dir: Path
    clips: list[Path]
    records: list[dict[str, Any]]
    resumed: list[dict[str, Any]]
    limit: int | None
    model: str
    metadata: Path
    entries: dict[str, Any]
    output: Path | None
    rows: Path | None
    command: str
    dry_run: bool


def build_plan(args: argparse.Namespace, config_path: Path, argv: Sequence[str]) -> EvalPlan:
    """Resolve the configuration, the clips and the judging protocol into an :class:`EvalPlan`."""
    from eveworld.utils.config import resolve_path

    root = repo_root()
    config = load_config(config_path, [])
    pred_dir = resolve_pred_dir(args.pred_dir, config, root)
    metadata = resolve_path(args.metadata or str(cfg_value(config, "data.metadata", DEFAULT_METADATA)), root)
    judge = judge_of(config, args.judge)
    protocol = protocol_settings(config, args, judge)
    output = None if not args.output else resolve_path(args.output, root)
    if args.rows:
        rows_path: Path | None = resolve_path(args.rows, root)
    elif output is not None:
        rows_path = output.with_suffix(ROWS_SUFFIX)
    else:
        rows_path = None
    if pred_dir.name == GENERATED_SUBDIR:
        model = args.model or pred_dir.parent.name
    else:
        model = args.model or pred_dir.name
    resumed = read_rows(rows_path) if args.resume else []
    decided = {str(row.get("request_id")) for row in resumed if row.get("request_id") is not None}
    clips = select_clips(pred_dir, args.limit)
    plan = EvalPlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        judge=judge,
        protocol=protocol,
        pred_dir=pred_dir,
        clips=clips,
        records=[],
        resumed=resumed,
        limit=args.limit,
        model=model,
        metadata=metadata,
        entries=load_entries(metadata),
        output=output,
        rows=rows_path,
        command=" ".join(shlex.quote(part) for part in ("python", "scripts/evaluate/eval_instruction_following.py", *argv)),
        dry_run=bool(args.dry_run),
    )
    plan.records = [record_of(plan, path) for path in clips if path.stem not in decided]
    return plan


def endpoint_line(protocol: dict[str, Any], judge: str) -> str:
    """Endpoint the judge of the run talks to, as far as the configuration names one."""
    if protocol.get("base_url"):
        return str(protocol["base_url"])
    if protocol.get("env_endpoint"):
        return f"${protocol['env_endpoint']}"
    if judge == QWEN_JUDGE:
        return "$QWEN_BASE_URL or the DashScope compatible mode"
    return "$GEMINI_BASE_URL or $DIFROST_GENAI_BASE_URL"


def print_plan(plan: EvalPlan) -> None:
    """Print the run, one aligned field per line."""
    protocol = plan.protocol
    found = "" if plan.pred_dir.is_dir() else " (not found)"
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(
        f"protocol:     {protocol['protocol']}, {int(protocol['frame_count'])} frames, "
        f"JPEG quality {int(protocol['jpeg_quality'])}, temperature "
        f"{float(protocol['temperature']):g}"
    )
    print(f"judge:        {plan.judge} over {protocol['model']}")
    print(f"endpoint:     {endpoint_line(protocol, plan.judge)}")
    print(
        f"judge call:   max tokens {int(protocol['max_tokens'])}, retries "
        f"{int(protocol['retries'])}, timeout {float(protocol['timeout']):g} s, "
        f"frames at most {int(protocol['max_side'])} px on the longest side"
    )
    print(f"concurrency:  {int(protocol['concurrency'])} configured, the judge class " "asks one clip after another")
    print(f"pred dir:     {plan.pred_dir}{found}")
    print(f"model:        {plan.model}")
    print(f"metadata:     {plan.metadata}")
    print(
        f"instructions: {len(plan.entries)} entries, "
        f"{sum(1 for entry in plan.entries.values() if entry_instruction(entry))} with an "
        "instruction"
    )
    limit = "" if plan.limit is None else f", limit {plan.limit}"
    skipped = len(plan.clips) - len(plan.records)
    print(f"clips:        {len(plan.clips)} selected{limit}, {skipped} already decided")
    for record in plan.records[:PREVIEW]:
        print(f"clip:         {Path(str(record['video'])).name} -> {record['question']}")
    if len(plan.records) > PREVIEW:
        print(f"clip:         ... and {len(plan.records) - PREVIEW} more")
    print(f"to judge:     {len(plan.records)} clips")
    print(f"output:       {plan.output or 'standard output'}")
    print(f"rows:         {plan.rows or 'none'}")
    print(f"dry run:      {'yes' if plan.dry_run else 'no'}")


def build_judge(plan: EvalPlan) -> Any:
    """Judge of the run, built from the protocol settings and the endpoint environment."""
    from eveworld.evaluation.instruction_following import gemini_if, qwen_if

    settings = plan.protocol
    kwargs: dict[str, Any] = {
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "timeout": settings["timeout"],
        "retries": settings["retries"],
        "frame_count": settings["frame_count"],
        "max_side": settings["max_side"],
    }
    if settings.get("prompt_template"):
        kwargs["prompt_template"] = settings["prompt_template"]
    if plan.judge == QWEN_JUDGE:
        return qwen_if.QwenIFJudge(
            model=str(settings["model"]),
            base_url=settings.get("base_url"),
            disable_thinking=bool(settings.get("disable_thinking", True)),
            **kwargs,
        )
    gemini_kwargs = dict(kwargs)
    if settings.get("thinking_level"):
        gemini_kwargs["thinking_level"] = str(settings["thinking_level"])
    return gemini_if.GeminiIFJudge(
        model=str(settings["model"]),
        base_url=settings.get("base_url"),
        include_thoughts=bool(settings.get("include_thoughts", False)),
        **gemini_kwargs,
    )


def row_of(scored: dict[str, Any], plan: EvalPlan) -> dict[str, Any]:
    """Item-level row of one judged clip, under the schema of ``results/item_level/README.md``."""
    verdict = scored.get("verdict")
    row: dict[str, Any] = {
        "request_id": scored.get("request_id", ""),
        "model": plan.model,
        "video": scored.get("video", ""),
        "judge": plan.judge,
        plan.judge: None if verdict is None else (100.0 if verdict else 0.0),
        "verdict": verdict,
        "protocol": PROTOCOL,
    }
    category = scored.get("category")
    if category:
        row["category"] = str(category)
    error = scored.get("error")
    if error:
        row["error"] = str(error)
    if verdict is None or error:
        row["raw_text"] = str(scored.get("raw_text", ""))
    return row


def judge_clips(plan: EvalPlan, judge: Any) -> list[dict[str, Any]]:
    """Judge the records of the plan and return one item-level row per clip."""
    from eveworld.evaluation.instruction_following import score_records

    scores = score_records(plan.records, judge)
    records = list(scores["records"])
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(plan.records):
        scored = dict(records[index]) if index < len(records) else dict(record)
        scored.setdefault("request_id", record.get("request_id", ""))
        scored.setdefault("video", record.get("video", ""))
        if "category" not in scored and "category" in record:
            scored["category"] = record["category"]
        rows.append(row_of(scored, plan))
    return rows


def summarize(plan: EvalPlan, rows: list[dict[str, Any]], judged: int) -> dict[str, Any]:
    """Aggregate the rows into the summary written to ``--output``.

    The overall accuracy and the per-category breakdown are the ones
    ``eveworld.evaluation.instruction_following.summarize`` reports: the denominator is the set of
    decided clips, and a judge call that errored is reported under ``errors`` instead of counting
    as a broken instruction.
    """
    from eveworld.evaluation.instruction_following import summarize as summarize_scores

    scores = summarize_scores(rows)
    return {
        "config": str(plan.config_path),
        "name": plan.name,
        "judge": plan.judge,
        "judge_model": str(plan.protocol["model"]),
        "model": plan.model,
        "protocol": PROTOCOL,
        "protocol_settings": plan.protocol,
        "command": plan.command,
        "pred_dir": str(plan.pred_dir),
        "metadata": str(plan.metadata),
        "selected_prompts": len(plan.clips),
        "records": len(rows),
        "judged": judged,
        "resumed": len(plan.resumed),
        "errors": int(scores["errors"]),
        "scores": scores,
    }


def run_evaluation(plan: EvalPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Judge the clips of the plan and return the summary with the rows of the whole set."""
    if not plan.pred_dir.is_dir():
        raise FileNotFoundError(f"prediction directory {plan.pred_dir} does not exist")
    if not plan.clips:
        raise FileNotFoundError(f"no generated clip under {plan.pred_dir}")
    judged = judge_clips(plan, build_judge(plan)) if plan.records else []
    rows = [*plan.resumed, *judged]
    return summarize(plan, rows, len(judged)), rows


def write_results(plan: EvalPlan, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the aggregate and the per-clip rows, when the run named files for them."""
    from eveworld.utils.io import write_json, write_jsonl

    if plan.output is not None:
        write_json(summary, plan.output)
    if plan.rows is not None:
        write_jsonl(rows, plan.rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: resolve the run, then judge the clips or dry-run them."""
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
    if args.frame_count is not None and args.frame_count < 1:
        print("error: --frame-count must be positive", file=sys.stderr)
        return 2
    try:
        plan = build_plan(args, config_path, arguments)
        print_plan(plan)
        if plan.dry_run:
            print("dry run: no clip judged")
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
