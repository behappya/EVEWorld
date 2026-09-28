"""WorldArena adapter: read the per-metric result JSON files into the paper's Table 3 columns.

WorldArena 1.0 scores every prompt with eight core metrics -- image quality, aesthetic quality,
dynamic degree, flow score, motion smoothness, subject consistency, background consistency and
photometric consistency -- and reports Overall as their equal-weight mean. The released evaluator
writes one file per metric per model, ``<root>/<model>/core/<metric>.json``, each holding
``[<video names>, <per-clip results>]`` with values normalized to ``[0, 1]``.
:func:`load_scores` turns them into one record per clip and :func:`aggregate` reports the table on
the 0-100 scale the paper prints::

    python -m eveworld.evaluation.worldarena.evaluate --input outputs/worldarena --model eveworld

Clips are generated at ``480 x 768`` and scored on the prompts that pass the shared
detector-eligible filter, after normalization to the ``640 x 480``, 121 frames, 24 fps contract of
the cross-model comparison. The per-clip records keep the evaluator's ``[0, 1]`` values and carry
``overall`` as the legacy EWMScore-local-8; MLR is reported beside this table and is computed by
:mod:`eveworld.evaluation.mlr`, not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from eveworld.utils.io import read_json, write_json, write_jsonl
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CORE_METRICS",
    "DISPLAY_NAMES",
    "METRICS",
    "aggregate",
    "evaluate",
    "load_scores",
    "main",
]

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

METRICS = CORE_METRICS + ("overall",)

DISPLAY_NAMES = {
    "image_quality": "Image Quality",
    "aesthetic_quality": "Aesthetic Quality",
    "dynamic_degree": "Dynamic Degree",
    "flow_score": "Flow Score",
    "motion_smoothness": "Motion Smoothness",
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "photometric_smoothness": "Photometric Consistency",
    "overall": "Overall",
}

_UNKNOWN = "unknown"


def load_scores(path: str | Path, model: str | None = None) -> dict[str, Any]:
    """Read the WorldArena metric files of one or more models.

    Args:
        path: An evaluation root holding one ``<model>/core/<metric>.json`` tree per model, a
            ``core`` or model directory holding the metric files directly, or a single metric file,
            which is read together with the metrics beside it.
        model: Model directory under the root to read. Without it every model directory holding
            metric files is read.

    Returns:
        ``{"path", "models", "videos", "coverage", "records"}``. ``coverage`` counts the clips
        each metric covers per model and ``videos`` the distinct clips per model; every record
        carries ``request_id``, ``model``, ``video_path``, one field per metric the run produced
        and the derived ``overall``. A metric file the run did not write is reported as a warning
        and simply absent from the records.
    """
    source = Path(path)
    records: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, int]] = {}
    for name, directory in _result_dirs(source, model):
        rows, counts = _read_directory(directory, name)
        records.extend(rows)
        coverage[name] = counts
    videos = {name: len({row["request_id"] for row in records if _model_of(row) == name}) for name in coverage}
    logger.debug("read %d WorldArena clips over %d models from %s", len(records), len(coverage), source)
    return {
        "path": str(source),
        "models": sorted(coverage),
        "videos": videos,
        "coverage": coverage,
        "records": records,
    }


def evaluate(records: Any, *, model: str | None = None) -> dict[str, Any]:
    """Summarize WorldArena records into the paper's Table 3 columns.

    Args:
        records: The payload of :func:`load_scores`, a list of its records, or the path of the
            result tree to read them from.
        model: Model directory to read when the records come from a path.

    Returns:
        ``{"rows", "scores"}``. ``rows`` holds one record per clip with the evaluator's ``[0, 1]``
        values and ``scores`` the summary of :func:`aggregate` on the 0-100 scale.
    """
    rows = [_normalize_row(row) for row in _as_rows(records, model=model)]
    return {"rows": rows, "scores": aggregate(rows)}


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the per-clip records into the eight core metrics and Overall.

    Every metric is the mean of the values the records carry, multiplied by 100, and ``overall``
    the equal-weight mean of the eight, which is the EWMScore-local-8 of the released evaluator.

    Args:
        records: Records as :func:`load_scores` returns them.

    Returns:
        One value per entry of :data:`CORE_METRICS` and ``overall``, ``None`` for a metric the
        records do not cover, the ``rows`` and ``videos`` counts, a per-metric ``coverage`` count,
        the single ``model`` name when the records agree on one, the ``models`` they cover and
        ``by_model`` with the same table restricted to one model each. Records that carry no model
        are grouped under ``unknown``.
    """
    items = [_normalize_row(row) for row in rows]
    summary = _table(items)
    models = sorted({_model_of(row) for row in items})
    summary["model"] = models[0] if len(models) == 1 and models[0] != _UNKNOWN else None
    summary["models"] = models
    summary["by_model"] = {name: _table(_of_model(items, name)) for name in models}
    return summary


def main(argv: list[str] | None = None) -> int:
    """Summarize WorldArena metric files and write the summary."""
    parser = argparse.ArgumentParser(
        prog="eveworld.evaluation.worldarena.evaluate",
        description="Summarize WorldArena core-8 metric files into the paper's Table 3 columns.",
    )
    parser.add_argument("--input", type=Path, required=True, help="evaluation root, metric directory or metric file")
    parser.add_argument("--model", default=None, help="model directory under the evaluation root")
    parser.add_argument("--output", type=Path, default=None, help="summary JSON to write")
    parser.add_argument("--rows", type=Path, default=None, help="per-clip records to write as JSONL")
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


def _result_dirs(source: Path, model: str | None) -> list[tuple[str, Path]]:
    """List the ``(model, directory)`` pairs whose metric files the input names."""
    if source.is_file():
        directory = source.parent
        return [(model or _model_name(directory, source), directory)]
    if not source.is_dir():
        raise FileNotFoundError(f"no WorldArena metric files at {source}")
    if model:
        target = source / model
        if not _holds_metrics(target):
            raise FileNotFoundError(f"no WorldArena metric files under {target}")
        return [(model, _metric_dir(target))]
    if _holds_metrics(source):
        metrics_dir = _metric_dir(source)
        return [(_model_name(metrics_dir, source), metrics_dir)]
    names = sorted(child.name for child in source.iterdir() if child.is_dir() and _holds_metrics(child))
    if not names:
        raise FileNotFoundError(f"no WorldArena metric files under {source}")
    return [(name, _metric_dir(source / name)) for name in names]


def _read_directory(
    directory: Path,
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read every metric file of one model into one record per clip."""
    records: dict[str, dict[str, Any]] = {}
    coverage: dict[str, int] = {}
    for metric in CORE_METRICS:
        path = directory / f"{metric}.json"
        if not path.is_file():
            logger.warning("no %s metric at %s", metric, path)
            continue
        points = _metric_values(path, metric)
        coverage[metric] = len(points)
        for request_id, (value, video_path) in points.items():
            record = records.setdefault(
                request_id,
                {"request_id": request_id, "model": model, "video_path": video_path},
            )
            record[metric] = value
    return [_normalize_row(records[key]) for key in sorted(records)], coverage


def _metric_values(path: Path, metric: str) -> dict[str, tuple[float, str]]:
    """Read one metric file into ``{request_id: (value, video_path)}``."""
    payload = read_json(path)
    if not isinstance(payload, Mapping) or metric not in payload:
        raise ValueError(f"{path}: no {metric!r} result")
    values: dict[str, tuple[float, str]] = {}
    for entry in _entries(payload[metric], path, metric):
        video_path = str(entry.get("video_path") or entry.get("request_id") or "")
        request_id = Path(video_path).stem
        if not request_id:
            raise ValueError(f"{path}: {metric} entry without a video path")
        value = _as_number(entry.get("video_results_normalized", entry.get("video_results")))
        if value is None:
            raise ValueError(f"{path}: {metric} has no numeric value for {request_id}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{path}: {metric} value {value} outside [0, 1] for {request_id}")
        values[request_id] = (value, video_path)
    return values


def _entries(result: Any, path: Path, metric: str) -> list[Mapping[str, Any]]:
    """Flatten the ``[names, details]`` payload of one metric into its per-clip entries."""
    details = result[1] if isinstance(result, list) and len(result) > 1 else None
    if not isinstance(details, list):
        raise ValueError(f"{path}: invalid {metric} result")
    flat: list[Any] = []
    for detail in details:
        flat.extend(detail if isinstance(detail, list) else [detail])
    entries: list[Mapping[str, Any]] = []
    for entry in flat:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: invalid {metric} entry {entry!r}")
        entries.append(entry)
    return entries


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one record: string ids, a float per core metric and the legacy overall score."""
    item = dict(row)
    item["request_id"] = str(item.get("request_id") or "")
    item["model"] = str(item.get("model") or "")
    for metric in CORE_METRICS:
        item[metric] = _as_number(item.get(metric))
    core = [item[metric] for metric in CORE_METRICS if item[metric] is not None]
    if core:
        item["overall"] = round(100.0 * sum(core) / len(core), 4)
    elif "overall" not in item:
        item["overall"] = None
    return item


def _table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one group of clips into the 0-100 table of the paper."""
    summary: dict[str, Any] = {}
    for metric in CORE_METRICS:
        values = [float(row[metric]) for row in rows if _is_number(row.get(metric))]
        summary[metric] = round(100.0 * sum(values) / len(values), 4) if values else None
    covered = [summary[metric] for metric in CORE_METRICS if summary[metric] is not None]
    summary["overall"] = round(sum(covered) / len(covered), 4) if covered else None
    summary["rows"] = len(rows)
    summary["videos"] = len({str(row.get("request_id") or "") for row in rows})
    summary["coverage"] = {metric: sum(1 for row in rows if _is_number(row.get(metric))) for metric in CORE_METRICS}
    return summary


def _as_rows(records: Any, *, model: str | None = None) -> list[dict[str, Any]]:
    """Accept the payload of :func:`load_scores`, a bare record list, or a path to read from."""
    if isinstance(records, (str, Path)):
        return list(load_scores(records, model=model)["records"])
    if isinstance(records, Mapping):
        inner = records.get("records")
        if not isinstance(inner, list):
            raise ValueError("a record mapping needs a 'records' list")
        return [dict(row) for row in inner]
    return [dict(row) for row in records]


def _metric_dir(directory: Path) -> Path:
    """The directory holding the ``<metric>.json`` files of one model."""
    core = directory / "core"
    return core if core.is_dir() else directory


def _holds_metrics(directory: Path) -> bool:
    """Whether a directory holds WorldArena metric files."""
    metrics_dir = _metric_dir(directory)
    return any((metrics_dir / f"{metric}.json").is_file() for metric in CORE_METRICS)


def _model_name(metrics_dir: Path, root: Path) -> str:
    """Name the model a metrics directory belongs to, or ``unknown`` when the layout says nothing."""
    if metrics_dir.name == "core":
        return metrics_dir.parent.name
    if metrics_dir != root:
        return metrics_dir.name
    return root.name if root.is_dir() else _UNKNOWN


def _model_of(row: Mapping[str, Any]) -> str:
    """The model bucket of one record."""
    return str(row.get("model") or _UNKNOWN)


def _of_model(rows: Sequence[Mapping[str, Any]], model: str) -> list[Mapping[str, Any]]:
    """The records of one model bucket."""
    return [row for row in rows if _model_of(row) == model]


def _as_number(value: Any) -> float | None:
    """Read a metric value as a float, or ``None`` when it is not a number."""
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _is_number(value: Any) -> bool:
    """Whether a value is a real number rather than a bool or a string."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
