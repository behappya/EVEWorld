#!/usr/bin/env python3
"""Resolve missing/error judge rows as zero while preserving an audit trail."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    args = parse_args()
    if args.audit.exists():
        raise FileExistsError(f"audit already exists: {args.audit}")

    manifest = load_manifest(args.manifest)
    items = {item["key"]: item for item in manifest}
    if len(items) != len(manifest):
        raise ValueError(f"duplicate keys in {args.manifest}")

    with args.csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"missing CSV header: {args.csv}")
    row_by_key = {row["key"]: row for row in rows}
    if len(row_by_key) != len(rows):
        raise ValueError(f"duplicate keys in {args.csv}")

    missing = sorted(set(items) - set(row_by_key))
    failed = sorted(
        key for key in items if key in row_by_key and row_by_key[key].get("error")
    )
    if not missing and not failed:
        raise ValueError("no unresolved rows found")

    timestamp = datetime.now(timezone.utc).isoformat()
    changes: list[dict[str, Any]] = []
    for key in failed:
        row = row_by_key[key]
        changes.append({"key": key, "kind": "error", "original_row": dict(row)})
        row["prediction"] = "0"
        row["raw_text"] = "FORCED_ZERO_BY_USER"
        row["error"] = ""

    for key in missing:
        item = items[key]
        row = {field: "" for field in fieldnames}
        row.update(
            {
                "key": key,
                "model": str(item["model"]),
                "split": str(item["split"]),
                "index": str(item["index"]),
                "request_id": str(item["request_id"]),
                "video_path": str(item["video_path"]),
                "prompt": str(item["prompt"]),
                "prediction": "0",
                "raw_text": "FORCED_ZERO_BY_USER",
                "error": "",
                "model_retry_count": "",
                "latency_sec": "",
                "created_at": timestamp,
            }
        )
        rows.append(row)
        row_by_key[key] = row
        changes.append({"key": key, "kind": "missing", "manifest_item": item})

    before_sha256 = sha256(args.csv)
    backup = args.csv.with_name(f"{args.csv.stem}.before_forced_zero.csv")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    shutil.copy2(args.csv, backup)

    temporary = args.csv.with_name(f".{args.csv.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.csv)

    audit = {
        "applied_at": timestamp,
        "policy": "Assign prediction=0 to unresolved/error rows by explicit user directive.",
        "reason": args.reason,
        "csv": str(args.csv.resolve()),
        "manifest": str(args.manifest.resolve()),
        "backup": str(backup.resolve()),
        "before_sha256": before_sha256,
        "after_sha256": sha256(args.csv),
        "row_count": len(rows),
        "changes": changes,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
