#!/usr/bin/env python3
"""Freeze the EVE-Frontier development and clean held-out manifests.

The 92-row ``raw_data/manifest.jsonl`` was exposed to the previous LAD-LoRA
feasibility run, so none of those rows may be called a final test set.  The
GR1 download contains 100 metadata rows; the eight rows absent from the old
manifest are retained as a small clean candidate test set.  The output makes
that provenance explicit and fails closed if the old manifest changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "file_name" not in row or "text" not in row:
                raise ValueError(f"{path}:{line_number} must contain file_name and text")
            rows.append(row)
    return rows


def _read_metadata(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "file_name" not in rows[0] or "text" not in rows[0]:
        raise ValueError(f"{path} must contain file_name and text columns")
    return rows


def _stable_key(row: dict, seed: int) -> str:
    payload = f"{seed}:{row['file_name']}:{row['text']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict], split: str, exposure: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            output = dict(row)
            output["split"] = split
            output["prior_training_exposure"] = exposure
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-manifest",
        type=Path,
        default=Path("/data/datasets/gagi/gr1_finetune_data/raw_data/manifest.jsonl"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/data/datasets/gagi/gr1_finetune_data/raw_hf/metadata.csv"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data/datasets/gagi/gr1_finetune_data/raw_hf/gr1"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    old_rows = _read_jsonl(args.old_manifest)
    metadata_rows = _read_metadata(args.metadata)
    packed_files = sorted(args.old_manifest.parent.glob("*.mp4"), key=lambda path: path.name)
    packed_index_by_id = {path.name: index for index, path in enumerate(packed_files)}
    old_ids = {row["file_name"] for row in old_rows}
    if len(old_rows) != 92 or len(old_ids) != len(old_rows):
        raise ValueError(f"expected 92 unique old-manifest rows, got {len(old_rows)} rows/{len(old_ids)} ids")

    metadata_by_id = {row["file_name"]: row for row in metadata_rows}
    if len(metadata_by_id) != len(metadata_rows):
        raise ValueError("metadata contains duplicate file_name values")
    missing_from_metadata = old_ids - metadata_by_id.keys()
    if missing_from_metadata:
        raise ValueError(f"old manifest rows absent from metadata: {sorted(missing_from_metadata)}")

    clean_rows = [row for row in metadata_rows if row["file_name"] not in old_ids]
    if len(clean_rows) != 8:
        raise ValueError(f"expected exactly 8 clean candidate rows, got {len(clean_rows)}")
    for row in clean_rows:
        video_path = args.raw_root / row["file_name"]
        row["video"] = str(video_path)
        row["media_available"] = video_path.is_file()

    development_rows = [metadata_by_id[row["file_name"]] for row in old_rows]
    if set(packed_index_by_id) != old_ids:
        raise ValueError("raw_data mp4 ids do not exactly match the frozen old manifest")
    for row in metadata_rows:
        row["packed_index"] = packed_index_by_id.get(row["file_name"])
    development_rows.sort(key=lambda row: _stable_key(row, args.seed))
    if not 1 <= args.val_count < len(development_rows):
        raise ValueError("--val-count must leave at least one development training row")
    val_rows = development_rows[: args.val_count]
    train_rows = development_rows[args.val_count :]

    # The old 92-row manifest is development-only.  The eight omitted GR1
    # rows are a clean candidate test set, but too small for the final paper
    # claim; the summary records that expansion is still required.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "train.jsonl", train_rows, "train", "old_lad_lora_exposed")
    _write_jsonl(args.out_dir / "val.jsonl", val_rows, "val", "old_lad_lora_exposed")
    _write_jsonl(args.out_dir / "test_candidate.jsonl", clean_rows, "test_candidate", "clean_vs_old_manifest")

    summary = {
        "created": "2026-07-15",
        "seed": args.seed,
        "old_manifest": str(args.old_manifest),
        "metadata": str(args.metadata),
        "old_manifest_count": len(old_rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "clean_test_candidate_count": len(clean_rows),
        "clean_test_candidate_ids": sorted(row["file_name"] for row in clean_rows),
        "clean_test_media_available": sum(bool(row["media_available"]) for row in clean_rows),
        "final_test_ready": False,
        "final_test_blocker": "The eight reserved GR1 rows are not downloaded and are insufficient for the preregistered held-out claim; acquire them and expand with external/new data before final model tuning is frozen.",
        "leakage_policy": "train/val are development-only; test_candidate is never used for tuning or checkpoint selection.",
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
