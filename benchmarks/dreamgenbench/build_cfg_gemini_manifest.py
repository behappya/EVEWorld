#!/usr/bin/env python3
"""Build a complete Gemini-IF manifest for the CFG checkpoint sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(data_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("env", "object", "behavior"):
        path = data_root / f"eval175_gr1_{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected list in {path}")
        for row in payload:
            item = dict(row)
            item["split"] = split
            rows.append(item)
    if len(rows) != 126:
        raise ValueError(f"expected 126 DreamGen rows, got {len(rows)}")
    request_ids = [str(row["request_id"]) for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate request_id in DreamGen input")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cfg-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.data_root)
    output_rows: list[dict[str, object]] = []
    for step in (50, 100, 150, 200, 250, 300):
        for cfg, tag in ((1.0, "cfg_1"), (2.5, "cfg_2p5"),
                         (5.0, "cfg_5"), (7.5, "cfg_7p5")):
            model = f"step_{step:03d}_{tag}"
            directory = args.cfg_root / f"step_{step:03d}" / tag / "generated_only"
            for row in rows:
                request_id = str(row["request_id"])
                video = directory / f"{request_id}.mp4"
                if not video.is_file() or video.stat().st_size <= 0:
                    raise FileNotFoundError(video)
                output_rows.append({
                    "key": f"{model}__{request_id}",
                    "video_path": str(video.resolve()),
                    "prompt": str(row["prompt"]),
                    "model": model,
                    "split": str(row["split"]),
                    "request_id": request_id,
                    "training_step": step,
                    "cfg": cfg,
                    "image": str(row["image"]),
                })

    expected = 6 * 4 * 126
    if len(output_rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(output_rows)}")
    keys = [str(row["key"]) for row in output_rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("manifest keys are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "rows": len(output_rows),
        "cells": 24,
        "rows_per_cell": 126,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
