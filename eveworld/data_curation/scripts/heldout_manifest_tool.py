#!/usr/bin/env python3
"""Validate frozen EVE manifests and expose packed training indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def row_id(row: dict) -> str:
    for key in ("file_name", "source_file_name", "request_id", "id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError(f"manifest row has no stable id: {row}")


def validate_splits(rows: list[dict], expected_split: str) -> None:
    ids = [row_id(row) for row in rows]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate manifest ids: {duplicates[:10]}")
    bad = [row_id(row) for row in rows if str(row.get("split", "")) != expected_split]
    if bad:
        raise ValueError(
            f"expected split={expected_split!r}, but {len(bad)} rows differ: {bad[:10]}"
        )


def validate_data(rows: list[dict], data_path: Path) -> None:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{data_path} must contain a JSON list")
    data_ids = {row_id(row) for row in data}
    missing = [row_id(row) for row in rows if row_id(row) not in data_ids]
    if missing:
        raise ValueError(f"manifest ids absent from {data_path}: {missing[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("count", "indices", "validate-eval"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--data-path", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    validate_splits(rows, args.expected_split)

    if args.command == "count":
        print(len(rows))
        return
    if args.command == "indices":
        indices = []
        for row in rows:
            value = row.get("packed_index")
            if value is None:
                raise ValueError(f"training row has no packed_index: {row_id(row)}")
            indices.append(int(value))
        if len(indices) != len(set(indices)):
            raise ValueError("packed_index values are not unique")
        print(",".join(str(value) for value in indices))
        return
    if args.data_path is None:
        parser.error("validate-eval requires --data-path")
    validate_data(rows, args.data_path)
    print(json.dumps({"count": len(rows), "split": args.expected_split, "valid": True}))


if __name__ == "__main__":
    main()
