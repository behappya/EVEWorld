"""PBench adapter: turn per-question robot predictions into the paper's four numbers.

PBench asks yes/no questions about a generated robot clip and splits them over three physical
categories -- ``Space``, ``Time`` and ``Fundamental Physics`` -- whose subcategories are the
Interaction, Geometry, Actions, Order, Camera, States, Object Permanence, Relationship, Attributes
and Object Permanence rows of the released metadata. The paper reports the benchmark as Domain
(all questions), Physical, Spatial and Temporal, which is the mapping in
:data:`CATEGORY_TO_METRIC`: the three categories map onto the latter three columns and Domain is
everything.

The released metadata file holds one record per clip with a list of ``qa_pairs``; a scored run
holds one record per question, with the model's answer. :func:`load_scores` reads either shape and
:func:`evaluate` normalizes the answers to ``yes``/``no``, marks them against the gold label and
summarizes the result. A question the model never answered stays in the denominator and is counted
as wrong, as the benchmark protocol requires. The two estimators the released numbers use are kept
apart: ``weighting="sample"`` averages the per-clip accuracies (the released score) and
``weighting="question"`` pools all questions::

    python -m eveworld.evaluation.pbench.evaluate --input predictions.jsonl --output pbench.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from eveworld.utils.io import read_json, read_jsonl, write_json, write_jsonl
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CATEGORY_TO_METRIC",
    "METRICS",
    "aggregate",
    "evaluate",
    "load_scores",
    "main",
    "normalize_yes_no",
]

METRICS = ("domain", "physical", "spatial", "temporal")

CATEGORY_TO_METRIC = {
    "fundamental physics": "physical",
    "space": "spatial",
    "time": "temporal",
}

_SAMPLE_KEYS = ("pbench_id", "sample_id", "id")
_QUESTION_KEYS = ("question", "prompt", "text")
_GOLD_KEYS = ("gold_answer", "answer", "label", "target")
_PREDICTION_KEYS = ("pred_answer", "prediction", "pred", "model_answer", "response", "raw_text")
_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"?(yes|no)"?', flags=re.IGNORECASE)
_YES = re.compile(r"\byes\b")
_NO = re.compile(r"\bno\b")


def normalize_yes_no(value: Any) -> str:
    """Normalize a yes/no answer to ``"yes"``, ``"no"`` or ``""`` when it says neither."""
    text = "" if value is None else str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return ""


def load_scores(path: str | Path) -> dict[str, Any]:
    """Read PBench records from a JSON or JSONL file.

    Two shapes are accepted. The released metadata file holds one record per clip with a list of
    ``qa_pairs``, which is expanded into one row per question with an empty prediction; a scored
    run holds one row per question and usually carries a prediction and a gold label.

    Args:
        path: A ``.jsonl`` file with one record per line, or a ``.json`` file holding a list of
            records or a single record.

    Returns:
        ``{"path", "samples", "questions", "records"}``, where ``records`` is a list of dicts with
        ``pbench_id``, ``question_index``, ``question``, ``gold_answer``, ``pred_answer``,
        ``category`` and ``subcategory``, plus ``prediction`` when the input carried one.
    """
    source = Path(path)
    rows = _read_rows(source)
    records = _expand(rows)
    samples = {str(record["pbench_id"]) for record in records}
    logger.debug("read %d questions over %d clips from %s", len(records), len(samples), source)
    return {"path": str(source), "samples": len(samples), "questions": len(records), "records": records}


def evaluate(records: Any, *, weighting: str = "sample") -> dict[str, Any]:
    """Score predictions against the gold labels and summarize them.

    Args:
        records: The payload of :func:`load_scores`, a list of its records, or the path of the file
            to read them from.
        weighting: ``"sample"`` for the released estimator, which averages the per-clip accuracies,
            or ``"question"`` to pool all questions.

    Returns:
        ``{"rows", "scores", "samples", "questions", "answered", "errors"}``. ``rows`` is the
        per-question scoring table with the normalized answers and the ``is_correct`` flag, and
        ``scores`` the summary of :func:`aggregate`.
    """
    rows = _as_rows(records)
    scored = [_score_row(row) for row in rows]
    return {
        "rows": scored,
        "scores": aggregate(scored, weighting=weighting),
        "samples": len({str(row["pbench_id"]) for row in scored}),
        "questions": len(scored),
        "answered": sum(1 for row in scored if row["pred_answer"]),
        "errors": sum(1 for row in scored if row.get("error")),
    }


def aggregate(rows: Sequence[Mapping[str, Any]], *, weighting: str = "sample") -> dict[str, Any]:
    """Summarize the per-question rows into the Domain/Physical/Spatial/Temporal table.

    Args:
        rows: Rows as :func:`evaluate` returns them; rows that are not scored yet are scored here.
        weighting: ``"sample"`` or ``"question"``, as in :func:`evaluate`.

    Returns:
        One percentage per entry of :data:`METRICS`, ``None`` for a metric the rows do not cover,
        plus ``weighting``, the counts ``samples``, ``questions``, ``answered``, ``correct`` and
        ``errors``, and the ``by_category`` and ``by_subcategory`` breakdowns in the same units.
    """
    if weighting not in ("sample", "question"):
        raise ValueError(f"weighting must be 'sample' or 'question', got {weighting!r}")
    items = [_score_row(row) if "is_correct" not in row else dict(row) for row in rows]
    by_sample = _group_by_sample(items)

    summary: dict[str, Any] = {metric: _metric_accuracy(by_sample, metric, weighting) for metric in METRICS}
    summary["weighting"] = weighting
    summary["samples"] = len(by_sample)
    summary["questions"] = len(items)
    summary["answered"] = sum(1 for row in items if row["pred_answer"])
    summary["correct"] = sum(1 for row in items if row["is_correct"])
    summary["errors"] = sum(1 for row in items if row.get("error"))
    summary["by_category"] = _breakdown(items, lambda row: str(row.get("category") or "unknown"))
    summary["by_subcategory"] = _breakdown(
        items,
        lambda row: f"{row.get('category') or 'unknown'}/{row.get('subcategory') or 'unknown'}",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Score a PBench prediction file and write the summary."""
    parser = argparse.ArgumentParser(
        prog="eveworld.evaluation.pbench.evaluate",
        description="Summarize PBench robot predictions into the paper's four scores.",
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL or JSON of PBench records")
    parser.add_argument("--output", type=Path, default=None, help="summary JSON to write")
    parser.add_argument("--rows", type=Path, default=None, help="per-question rows to write as JSONL")
    parser.add_argument("--weighting", default="sample", choices=["sample", "question"])
    parser.add_argument("--limit", type=int, default=None, help="score at most this many questions")
    args = parser.parse_args(argv)

    try:
        payload = load_scores(args.input)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    records = payload["records"]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"error: no questions in {args.input}", file=sys.stderr)
        return 2

    result = evaluate(records, weighting=args.weighting)
    summary = result["scores"]
    if args.rows is not None:
        write_jsonl(result["rows"], args.rows)
        logger.info("wrote %s", args.rows)
    if args.output is not None:
        write_json(summary, args.output)
        logger.info("wrote %s", args.output)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read the records of a JSON or JSONL file, whichever suffix the file carries."""
    if not path.is_file():
        raise FileNotFoundError(f"no PBench records at {path}")
    if path.suffix.lower() == ".jsonl":
        rows: Iterable[Any] = read_jsonl(path)
    else:
        payload = read_json(path)
        if isinstance(payload, Mapping):
            inner = payload.get("records") or payload.get("samples")
            rows = inner if isinstance(inner, list) else [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"{path}: expected a list of records, got {type(payload).__name__}")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: record {index} is not an object")
        out.append(dict(row))
    return out


def _expand(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand clip records with ``qa_pairs`` into one row per question."""
    records: list[dict[str, Any]] = []
    for row in rows:
        pairs = row.get("qa_pairs")
        sample = _first_text(row, _SAMPLE_KEYS)
        if isinstance(pairs, list) and pairs:
            for index, pair in enumerate(pairs):
                if not isinstance(pair, Mapping):
                    raise ValueError(f"{sample}: qa_pairs[{index}] is not an object")
                records.append(_question_row(pair, row, sample, index))
            continue
        records.append(_question_row(row, row, sample, int(row.get("question_index") or 0)))
    return records


def _question_row(
    pair: Mapping[str, Any],
    sample: Mapping[str, Any],
    sample_id: str,
    index: int,
) -> dict[str, Any]:
    """Flatten one question and its metadata into a row."""
    row: dict[str, Any] = {
        "pbench_id": sample_id,
        "question_index": index,
        "question": _first_text(pair, _QUESTION_KEYS),
        "gold_answer": normalize_yes_no(_first_text(pair, _GOLD_KEYS)),
        "pred_answer": _prediction(pair) or _prediction(sample),
        "category": _first_text(pair, ("category",)) or _first_text(sample, ("category",)),
        "subcategory": _first_text(pair, ("subcategory",)) or _first_text(sample, ("subcategory",)),
    }
    if not row["question"]:
        raise ValueError(f"{sample_id}: question {index} has no text")
    if not row["gold_answer"]:
        raise ValueError(f"{sample_id}: question {index} has no yes/no gold answer")
    prediction = _first_text(pair, ("prediction_text", "raw_text")) or _first_text(sample, ("raw_text",))
    if prediction:
        row["prediction_text"] = prediction
    return row


def _score_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the answers of one row and mark it against its gold label."""
    item = dict(row)
    item["pbench_id"] = str(item.get("pbench_id") or "unknown")
    item["question_index"] = int(item.get("question_index") or 0)
    item["gold_answer"] = normalize_yes_no(item.get("gold_answer"))
    item["pred_answer"] = _prediction(item)
    item["category"] = str(item.get("category") or "unknown")
    item["subcategory"] = str(item.get("subcategory") or "unknown")
    item["metric"] = CATEGORY_TO_METRIC.get(item["category"].strip().lower())
    item["answered"] = bool(item["pred_answer"])
    item["is_correct"] = bool(item["pred_answer"]) and item["pred_answer"] == item["gold_answer"]
    return item


def _prediction(row: Mapping[str, Any]) -> str:
    """Read the model's yes/no answer out of a scored row."""
    for key in _PREDICTION_KEYS:
        if key not in row:
            continue
        answer = _answer_from_text(row.get(key))
        if answer:
            return answer
    return ""


def _answer_from_text(value: Any) -> str:
    """Read a yes/no answer out of a raw model response such as ``{"answer": "yes"}``."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    direct = normalize_yes_no(text)
    if direct:
        return direct
    match = _JSON_ANSWER.search(text)
    if match:
        return match.group(1).lower()
    lowered = text.lower()
    has_yes = _YES.search(lowered) is not None
    has_no = _NO.search(lowered) is not None
    if has_yes != has_no:
        return "yes" if has_yes else "no"
    return ""


def _as_rows(records: Any) -> list[dict[str, Any]]:
    """Accept the payload of :func:`load_scores`, a bare row list, or a path to read them from."""
    if isinstance(records, (str, Path)):
        return list(load_scores(records)["records"])
    if isinstance(records, Mapping):
        inner = records.get("records")
        if not isinstance(inner, list):
            raise ValueError("a record mapping needs a 'records' list")
        return [dict(row) for row in inner]
    return [dict(row) for row in records]


def _group_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Group the rows of one evaluation by clip id."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("pbench_id") or "unknown")].append(row)
    return dict(groups)


def _metric_accuracy(
    by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    metric: str,
    weighting: str,
) -> float | None:
    """Accuracy of one metric in percent, or ``None`` when no row belongs to it."""
    per_sample: list[float] = []
    correct = 0
    total = 0
    for rows in by_sample.values():
        subset = [row for row in rows if metric == "domain" or row.get("metric") == metric]
        if not subset:
            continue
        sample_correct = sum(1 for row in subset if row.get("is_correct"))
        per_sample.append(sample_correct / len(subset))
        correct += sample_correct
        total += len(subset)
    if weighting == "sample":
        if not per_sample:
            return None
        return round(100.0 * sum(per_sample) / len(per_sample), 2)
    if not total:
        return None
    return round(100.0 * correct / total, 2)


def _breakdown(
    rows: Sequence[Mapping[str, Any]],
    key: Any,
) -> dict[str, dict[str, Any]]:
    """Accuracy table over a grouping key, with ``None`` accuracy for an empty group."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        entry = counts[str(key(row))]
        entry[0] += 1
        entry[1] += int(bool(row.get("is_correct")))
    return {
        name: {
            "correct": entry[1],
            "total": entry[0],
            "accuracy": round(100.0 * entry[1] / entry[0], 2) if entry[0] else None,
        }
        for name, entry in sorted(counts.items())
    }


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first non-empty string field of ``record`` among ``keys``, else an empty string."""
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
