#!/usr/bin/env python3
"""Summarise the layer-wise consistency probe into the attachment-layer table.

The TIA adapter is attached to one Transformer block, and the block is chosen
offline by running a matching probe at every candidate layer: for each layer,
the target feature of frame ``t - 1`` retrieves its best local match among the
candidate cells of frame ``t``, and the endpoint error (EPE) of that match
against the localised target position is averaged over the clip. The layer with
the lowest EPE wins.

The probe writes one JSON per candidate block into a per-domain directory,
``<root>/<domain>/blockNN.json``, with the fields

```text
{"block": 8, "epe": 1.23, "reliability": {"metric": 0.12, "...": 0.34}}
```

``reliability`` is optional and either a mapping of setting-specific scores or a
single number; a mapping becomes one column per key. The script prints one
table per domain, sorted by block, with the lowest EPE inside ``--block-range``
in bold, followed by the blocks of the range that the domain did not report.
The winner is selected inside the range only; blocks outside it are still
printed for inspection.

Run ``python summarize.py --root experiments/analysis/tia_layer_probe``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["main", "load_domain", "missing_blocks"]

FILE_PATTERN = re.compile(r"block[-_]?(\d+)\.json$")
DEFAULT_RANGE = (8, 26)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="directory with one subdirectory per domain")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="domains to summarise, in order; defaults to every subdirectory with block files",
    )
    parser.add_argument(
        "--block-range",
        type=int,
        nargs=2,
        default=list(DEFAULT_RANGE),
        metavar=("LOW", "HIGH"),
        help="candidate block range whose minimum is reported as the winner",
    )
    parser.add_argument("--out", default=None, help="optional file for the markdown report")
    return parser.parse_args(argv)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def _block_number(path: Path, record: Any) -> int:
    match = FILE_PATTERN.search(path.name)
    from_name = int(match.group(1)) if match else None
    if not isinstance(record, dict):
        raise SystemExit(f"{path}: expected a JSON object, got {type(record).__name__}")
    from_record = record.get("block")
    if from_name is None and from_record is None:
        raise SystemExit(f"{path}: no block number in the file name or the record")
    if from_name is not None and from_record is not None and int(from_record) != from_name:
        raise SystemExit(f"{path}: block {from_record} does not match the file name block {from_name}")
    return int(from_name if from_name is not None else from_record)


def load_domain(path: Path, domain: str) -> dict[int, dict[str, Any]]:
    """Read every ``blockNN.json`` of one domain into a block-sorted mapping."""
    records: dict[int, dict[str, Any]] = {}
    for entry in sorted(path.glob("block*.json")):
        try:
            with open(entry, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read {entry}: {exc}") from exc
        number = _block_number(entry, record)
        if number in records:
            raise SystemExit(f"{entry}: block {number} was already read for {domain}")
        if "epe" not in record:
            raise SystemExit(f"{entry}: no epe field")
        epe = record["epe"]
        if not isinstance(epe, (int, float)) or isinstance(epe, bool):
            raise SystemExit(f"{entry}: epe must be a number, got {epe!r}")
        records[number] = {"epe": float(epe), "reliability": record.get("reliability")}
    if not records:
        raise SystemExit(f"{path}: no block*.json files found")
    return dict(sorted(records.items()))


def missing_blocks(blocks: Iterable[int], block_range: Sequence[int]) -> list[int]:
    """Blocks of the closed range that are absent from ``blocks``."""
    low, high = int(block_range[0]), int(block_range[1])
    return [block for block in range(low, high + 1) if block not in set(blocks)]


def _reliability_columns(records: dict[int, dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    scalar = False
    for record in records.values():
        value = record.get("reliability")
        if isinstance(value, dict):
            keys.update(str(key) for key in value)
        elif value is not None:
            scalar = True
    columns = sorted(keys)
    if scalar:
        columns.append("Reliab.")
    return columns


def _table(domain: str, records: dict[int, dict[str, Any]], block_range: Sequence[int]) -> list[str]:
    columns = _reliability_columns(records)
    low, high = int(block_range[0]), int(block_range[1])
    candidates = {block: row["epe"] for block, row in records.items() if low <= block <= high}
    best = min(candidates.values()) if candidates else None
    header = "| " + " | ".join(["Block", "EPE", *columns]) + " |"
    lines = [f"## {domain}", "", header, "|---:|---:|" + "---:|" * len(columns)]
    for block, row in records.items():
        cells = [_fmt(block), _fmt(row["epe"])]
        value = row.get("reliability")
        for key in columns:
            if isinstance(value, dict):
                cells.append(_fmt(value.get(key)))
            elif key == "Reliab.":
                cells.append(_fmt(value))
            else:
                cells.append("n/a")
        winner = best is not None and row["epe"] == best
        text = "| " + " | ".join(cells) + " |"
        if winner:
            text = "| **" + "** | **".join(cells) + "** |"
        lines.append(text)
    absent = missing_blocks(records, block_range)
    lines.append("")
    lines.append(f"winner inside blocks {low}-{high}: " + (f"**{min(candidates, key=candidates.get)}** (EPE {_fmt(best)})" if candidates else "n/a"))
    lines.append("missing blocks in range: " + (", ".join(str(block) for block in absent) if absent else "none"))
    lines.append("")
    return lines


def _selection(domains: Sequence[tuple[str, dict[int, dict[str, Any]]]], block_range: Sequence[int]) -> list[str]:
    low, high = int(block_range[0]), int(block_range[1])
    lines = ["## Selection", "", "| Domain | Winner | EPE | Next best | EPE |", "|---|---:|---:|---:|---:|"]
    for domain, records in domains:
        ranked = sorted(((row["epe"], block) for block, row in records.items() if low <= block <= high))
        if not ranked:
            lines.append(f"| {domain} | n/a | n/a | n/a | n/a |")
            continue
        winner = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                domain,
                winner[1],
                _fmt(winner[0]),
                runner[1] if runner else "n/a",
                _fmt(runner[0]) if runner else "n/a",
            )
        )
    lines.append("")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"{root}: not a directory")
    if args.domains:
        names = list(args.domains)
    else:
        names = sorted(entry.name for entry in root.iterdir() if entry.is_dir() and any(entry.glob("block*.json")))
    if not names:
        raise SystemExit(f"{root}: no domain subdirectories with block*.json files")
    domains = []
    for name in names:
        directory = root / name
        if not directory.is_dir():
            raise SystemExit(f"{directory}: not a directory")
        domains.append((name, load_domain(directory, name)))
    report = ["# TIA layer probe", ""]
    for name, records in domains:
        report.extend(_table(name, records, args.block_range))
    if len(domains) > 1:
        report.extend(_selection(domains, args.block_range))
    text = "\n".join(report).rstrip() + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
