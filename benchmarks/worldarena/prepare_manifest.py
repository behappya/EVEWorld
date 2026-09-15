#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit the isolated WorldArena 1.0 manifest"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def normalize_prompt(value: Any) -> str:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid prompt: {value!r}")
    return value.strip()


def main() -> None:
    args = parse_args()
    source = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(source, list) or len(source) != args.expected_count:
        raise ValueError(
            f"Summary coverage mismatch: {len(source) if isinstance(source, list) else 'invalid'}"
            f"/{args.expected_count}"
        )

    rows = []
    for index, item in enumerate(source, start=1):
        request_id = f"fixed_scene_task_episode{index}"
        image = item.get("image")
        if not isinstance(image, str) or not Path(image).is_file():
            raise FileNotFoundError(f"Missing first frame for {request_id}: {image}")
        rows.append(
            {
                "request_id": request_id,
                "prompt": normalize_prompt(item.get("prompt")),
                "image": image,
                "gt_path": item.get("gt_path"),
            }
        )

    expected_names = {f"{row['request_id']}.mp4" for row in rows}
    model_audits = {}
    complete = True
    for model in args.models:
        directory = args.video_root / f"{model}_test"
        files = {path.name: path for path in directory.glob("*.mp4")}
        missing = sorted(expected_names - set(files))
        extra = sorted(set(files) - expected_names)
        empty = sorted(name for name, path in files.items() if path.stat().st_size == 0)
        valid_count = len(expected_names) - len(missing) - len(empty)
        model_complete = not missing and not extra and not empty
        complete = complete and model_complete
        model_audits[model] = {
            "directory": str(directory),
            "expected_count": args.expected_count,
            "file_count": len(files),
            "valid_count": valid_count,
            "missing": missing,
            "extra": extra,
            "empty": empty,
            "complete": model_complete,
        }

    audit = {
        "summary": str(args.summary),
        "video_root": str(args.video_root),
        "expected_count": args.expected_count,
        "models": model_audits,
        "complete": complete,
    }
    if not complete:
        atomic_write_json(args.audit_output, audit)
        raise SystemExit("WorldArena 1.0 video audit failed")

    output_rows = rows[: args.limit] if args.limit is not None else rows
    if not output_rows:
        raise ValueError("Manifest must contain at least one row")
    atomic_write_json(args.output, output_rows)
    atomic_write_json(args.audit_output, audit)
    print(
        f"manifest={args.output} rows={len(output_rows)} "
        f"audit={args.audit_output} complete={complete}"
    )


if __name__ == "__main__":
    main()
