"""EWMBench adapter: read the official result CSV into the paper's Table 4 columns.

EWMBench scores generated AgiBot clips with the official metric suite -- BLEU, CLIP and diversity
under ``semantics``, dynamic degree, hand-object interaction and nDTW under
``trajectory_consistency``, the logic constraints, the scene consistency, PSNR and SSIM -- and
merges everything into one CSV whose ``MEAN`` row holds the mean of every column.
:func:`load_scores` reads that CSV and :func:`aggregate` rebuilds the ``MEAN`` row from the data
rows, so a filtered subset can be re-scored without the official toolkit::

    python -m eveworld.evaluation.ewmbench.evaluate --input final_results.csv --rows rows.jsonl

The reported table is Motion / Semantics / DYN / HSD / nDTW / Scene / Overall with
``motion = dyn + hsr + ndtw``, ``semantics = bleu + clip + diversity + logic`` and
``overall = motion + semantics + scene``, every term a mean over clips. Values stay in the raw
units of the CSV; paper Table 4 and its appendix print the component columns and Motion multiplied
by 100 and Semantics and Overall by 10. The CSV spells the hand-object score ``hsd`` where the
scorer key and the tables say HSR, so both column names are accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from eveworld.utils.io import write_json, write_jsonl
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "COMPONENTS",
    "CSV_COLUMNS",
    "METRICS",
    "MOTION_TERMS",
    "SEMANTIC_TERMS",
    "aggregate",
    "evaluate",
    "load_scores",
    "main",
]

METRICS = ("motion", "semantics", "dyn", "hsr", "ndtw", "scene", "overall")

COMPONENTS = ("bleu", "clip", "diversity", "dyn", "hsr", "ndtw", "logic", "scene", "psnr", "ssim")

MOTION_TERMS = ("dyn", "hsr", "ndtw")

SEMANTIC_TERMS = ("bleu", "clip", "diversity", "logic")

OVERALL_TERMS = ("motion", "semantics", "scene")

CSV_COLUMNS = {
    "BLEUScore": "bleu",
    "CLIPScore": "clip",
    "diversity": "diversity",
    "dyn": "dyn",
    "hsd": "hsr",
    "hsr": "hsr",
    "ndtw": "ndtw",
    "logic_constraints": "logic",
    "scene_consistency": "scene",
    "psnr": "psnr",
    "ssim": "ssim",
}

_DERIVED = (("motion", MOTION_TERMS), ("semantics", SEMANTIC_TERMS), ("overall", OVERALL_TERMS))


def load_scores(path: str | Path, model: str | None = None) -> dict[str, Any]:
    """Read an EWMBench result CSV.

    The official toolkit writes the header, one row per clip and trial, a ``task_id`` row of
    ``MEAN`` holding the per-column means, a blank row and a ``#`` comment row; the latter three are
    skipped here. A cell that is empty or holds ``-`` -- the ``diversity`` cells outside the first
    trial -- is read as absent.

    Args:
        path: The result CSV.
        model: Model name to build the clip ids from. The official toolkit names a clip
            ``<model>_dataset_<task>_<episode>_<trial>``; without a name the ids keep the
            ``<task>_<episode>_<trial>`` tail.

    Returns:
        ``{"path", "model", "clips", "tasks", "trials", "records"}``. Every record carries
        ``request_id``, ``model``, ``task_id``, ``episode_id`` and ``trial_id``, one field per
        entry of :data:`COMPONENTS` (``None`` for a column the file does not carry) and the derived
        ``motion``, ``semantics`` and ``overall`` scores.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"no EWMBench result CSV at {source}; run the official EWMBench toolkit first"
        )

    lines: list[list[str]] = []
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        for cells in csv.reader(handle):
            first = _cell(cells, 0).strip()
            if not first or first.startswith("#"):
                continue
            if first.upper() == "MEAN":
                logger.debug("skipping the EWMBench MEAN row of %s", source)
                continue
            lines.append(cells)
    if not lines:
        raise ValueError(f"{source}: no scored clips in the EWMBench CSV")

    header = [name.strip() for name in lines[0]]
    if not header or header[0].lower() != "task_id":
        raise ValueError(f"{source}: expected a task_id column, got {header[0] if header else ''!r}")
    index = {name: position for position, name in enumerate(header)}
    records = [_row_from_cells(cells, index, model) for cells in lines[1:]]
    counts = _counts(records)
    logger.debug("read %d EWMBench rows over %d clips from %s", len(records), counts["clips"], source)
    return {
        "path": str(source),
        "model": model,
        **counts,
        "records": records,
    }


def evaluate(records: Any, *, model: str | None = None) -> dict[str, Any]:
    """Summarize EWMBench records into the paper's Table 4 columns.

    Args:
        records: The payload of :func:`load_scores`, a list of its records, or the path of the CSV
            to read them from.
        model: Model name for the clip ids when the records are read from a path.

    Returns:
        ``{"rows", "scores", "clips", "tasks", "trials"}``. ``rows`` holds one record per clip with
        the components and the derived scores, and ``scores`` is the summary of :func:`aggregate`.
    """
    rows = [_derive_row(row) for row in _as_rows(records, model=model)]
    return {
        "rows": rows,
        "scores": aggregate(rows),
        **_counts(rows),
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the per-clip rows into the Motion/Semantics/DYN/HSR/nDTW/Scene/Overall table.

    Each component is the mean of the values the rows carry for it. The derived scores sum those
    means rather than the per-clip sums, the way the official toolkit builds its ``MEAN`` row, and
    they are built in order so ``overall`` adds the ``motion`` and ``semantics`` scores rather than
    the components behind them. A derived score whose terms are all absent is ``None``.

    Args:
        rows: Records as :func:`load_scores` returns them.

    Returns:
        One value per entry of :data:`METRICS`, the remaining entries of :data:`COMPONENTS`,
        ``None`` for a column the rows do not cover, and the counts ``rows``, ``clips``, ``tasks``
        and ``trials`` plus the ``model`` name when the rows agree on one.
    """
    items = [_derive_row(row) for row in rows]
    means = {name: _column_mean(items, name) for name in COMPONENTS}
    derived: dict[str, float | None] = {}
    pool: dict[str, Any] = dict(means)
    for name, terms in _DERIVED:
        value = _sum_terms(pool, terms)
        derived[name] = value
        pool[name] = value

    summary: dict[str, Any] = {}
    for name in METRICS:
        summary[name] = derived[name] if name in derived else means[name]
    for name in COMPONENTS:
        summary.setdefault(name, means[name])
    summary["model"] = _single([row.get("model") for row in items])
    summary["rows"] = len(items)
    summary.update(_counts(items))
    return summary


def main(argv: list[str] | None = None) -> int:
    """Summarize an EWMBench result CSV and write the summary."""
    parser = argparse.ArgumentParser(
        prog="eveworld.evaluation.ewmbench.evaluate",
        description="Summarize an EWMBench result CSV into the paper's Table 4 columns.",
    )
    parser.add_argument("--input", type=Path, required=True, help="EWMBench result CSV")
    parser.add_argument("--output", type=Path, default=None, help="summary JSON to write")
    parser.add_argument("--rows", type=Path, default=None, help="per-clip rows to write as JSONL")
    parser.add_argument("--model", default=None, help="model name for the clip ids")
    args = parser.parse_args(argv)

    try:
        payload = load_scores(args.input, model=args.model)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not payload["records"]:
        print(f"error: no scored clips in {args.input}", file=sys.stderr)
        return 2

    result = evaluate(payload["records"])
    summary = result["scores"]
    if args.rows is not None:
        write_jsonl(result["rows"], args.rows)
        logger.info("wrote %s", args.rows)
    if args.output is not None:
        write_json(summary, args.output)
        logger.info("wrote %s", args.output)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _row_from_cells(
    cells: Sequence[str],
    index: Mapping[str, int],
    model: str | None,
) -> dict[str, Any]:
    """Turn one CSV row into a record with its components and derived scores."""
    task_id = _identifier(_cell(cells, index.get("task_id", 0)))
    episode_id = _cell(cells, index.get("episode_id", 1)).strip()
    trial_id = _identifier(_cell(cells, index.get("trial_id", 2)))
    row: dict[str, Any] = {
        "request_id": _request_id(model, task_id, episode_id, trial_id),
        "model": model,
        "task_id": task_id,
        "episode_id": episode_id,
        "trial_id": trial_id,
    }
    for name in COMPONENTS:
        row[name] = None
    for column, name in CSV_COLUMNS.items():
        position = index.get(column)
        if position is None:
            continue
        value = _parse_cell(_cell(cells, position))
        if value is not None:
            row[name] = value
    return _derive_row(row)


def _derive_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Fill the derived Motion, Semantics and Overall scores of one record.

    A derived score is the sum of the terms the record carries and stays ``None`` only when it
    carries none of them, so a partial CSV still yields the sums it can support. A value the record
    already carries and that the components cannot rebuild is kept.
    """
    item = dict(row)
    for name, terms in _DERIVED:
        value = _sum_terms(item, terms)
        if value is not None:
            item[name] = value
        elif name not in item:
            item[name] = None
    return item


def _as_rows(records: Any, *, model: str | None = None) -> list[dict[str, Any]]:
    """Accept the payload of :func:`load_scores`, a bare record list, or a path to read them from."""
    if isinstance(records, (str, Path)):
        return list(load_scores(records, model=model)["records"])
    if isinstance(records, Mapping):
        inner = records.get("records")
        if not isinstance(inner, list):
            raise ValueError("a record mapping needs a 'records' list")
        return [dict(row) for row in inner]
    return [dict(row) for row in records]


def _request_id(model: str | None, task_id: Any, episode_id: str, trial_id: Any) -> str:
    """Build a clip id in the official ``<model>_dataset_<task>_<episode>_<trial>`` form."""
    tail = f"{task_id}_{episode_id}_{trial_id}"
    return f"{model}_dataset_{tail}" if model else tail


def _identifier(text: str) -> int | str:
    """Read an id cell as an integer when it is one, else as the stripped text."""
    value = text.strip()
    try:
        return int(value)
    except ValueError:
        return value


def _parse_cell(text: str) -> float | None:
    """Read a CSV cell as a float; an empty cell, ``-`` and anything non-numeric are absent."""
    value = text.strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        logger.debug("ignoring non-numeric EWMBench cell %r", text)
        return None


def _cell(cells: Sequence[str], position: int) -> str:
    """Read one cell of a CSV row, which may be shorter than the header."""
    if 0 <= position < len(cells):
        return cells[position]
    return ""


def _sum_terms(values: Mapping[str, Any], terms: Sequence[str]) -> float | None:
    """Sum the numeric fields ``terms`` of ``values``, or ``None`` when it carries none."""
    present = [float(values[term]) for term in terms if _is_number(values.get(term))]
    return round(sum(present), 6) if present else None


def _column_mean(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
    """Mean of the values the rows carry for one component, or ``None`` when they carry none."""
    values = [float(row[name]) for row in rows if _is_number(row.get(name))]
    return round(sum(values) / len(values), 6) if values else None


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count the clips, tasks and trials the records cover."""
    labelled = [row for row in rows if row.get("task_id") is not None]
    return {
        "clips": len({(row.get("task_id"), row.get("episode_id")) for row in labelled}),
        "tasks": len({row.get("task_id") for row in labelled}),
        "trials": len({row.get("trial_id") for row in rows if row.get("trial_id") is not None}),
    }


def _single(values: Sequence[Any]) -> str | None:
    """The one non-empty value of a sequence, or ``None`` when it holds none or several."""
    present = {str(value) for value in values if value}
    return present.pop() if len(present) == 1 else None


def _is_number(value: Any) -> bool:
    """Whether a value is a real number rather than a bool or a string."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
