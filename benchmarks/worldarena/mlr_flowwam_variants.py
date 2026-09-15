#!/usr/bin/env python3
"""Evaluate the frozen GDINO MLR protocol on arbitrary FlowWAM variants."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlr_dispatch import extract_mover, wilson  # noqa: E402


def parse_variants(specs: list[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"variant must be NAME=VIDEO_DIR, got: {spec}")
        name, directory = spec.split("=", 1)
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid variant name: {name!r}")
        if name in variants:
            raise ValueError(f"duplicate variant name: {name}")
        variants[name] = Path(directory)
    if not variants:
        raise ValueError("at least one --variant is required")
    return variants


def build_jobs(manifest: Path, variants: dict[str, Path]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = json.loads(manifest.read_text())
    if len(rows) != 250:
        raise ValueError(f"expected 250 FlowWAM rows, got {len(rows)}")
    selected: list[tuple[dict[str, Any], str]] = []
    excluded: Counter[str] = Counter()
    for row in rows:
        mover, reason = extract_mover(str(row["prompt"]))
        if mover is None:
            excluded[reason] += 1
        else:
            selected.append((row, mover))
    jobs: list[dict[str, str]] = []
    for name, directory in variants.items():
        for row, mover in selected:
            video = directory / f"{row['request_id']}.mp4"
            image = Path(row["image"])
            if not video.is_file():
                raise FileNotFoundError(video)
            if not image.is_file():
                raise FileNotFoundError(image)
            jobs.append({
                "model": name,
                "request_id": str(row["request_id"]),
                "prompt": str(row["prompt"]),
                "mover": mover,
                "video": str(video.resolve()),
                "condition_image": str(image.resolve()),
            })
    audit = {
        "protocol": "flowwam_mlr_gdino_v2",
        "flowwam_rows": len(rows),
        "flowwam_selected": len(selected),
        "flowwam_excluded": dict(sorted(excluded.items())),
        "models": list(variants),
        "job_count": len(jobs),
    }
    return jobs, audit


def merge(output_dir: Path, jobs: list[dict[str, str]], audit: dict[str, Any],
          num_shards: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for shard in range(num_shards):
        payload = json.loads((output_dir / f"part{shard}.json").read_text())
        records.extend(payload["records"])
    expected = {(job["model"], job["request_id"]) for job in jobs}
    actual = {(row.get("model"), row.get("request_id")) for row in records}
    if len(records) != len(jobs) or actual != expected:
        raise RuntimeError(
            f"record coverage mismatch: records={len(records)} "
            f"actual={len(actual)} expected={len(expected)}"
        )
    summaries: dict[str, Any] = {}
    for model in audit["models"]:
        model_rows = [row for row in records if row.get("model") == model]
        errors = [row for row in model_rows if row.get("error")]
        eligible = [row for row in model_rows if row.get("eligible") and not row.get("error")]
        events = sum(bool(row.get("mlr_event")) for row in eligible)
        summaries[model] = {
            "records": len(model_rows),
            "errors": len(errors),
            "eligible": len(eligible),
            "coverage": len(eligible) / len(model_rows) if model_rows else 0.0,
            "events": events,
            "mlr": events / len(eligible) if eligible else None,
            "wilson_95": wilson(events, len(eligible)),
        }
    report = {"metadata": audit, "summary": summaries, "records": records}
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=VIDEO_DIR")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--frame-count", type=int, default=24)
    args = ap.parse_args()
    if args.num_shards != 8:
        raise ValueError("this runner requires eight GPUs")
    variants = parse_variants(args.variant)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs, audit = build_jobs(args.manifest, variants)
    audit["num_shards"] = args.num_shards
    audit["frame_count"] = args.frame_count
    (args.output_dir / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    (args.output_dir / "parser_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    script = HERE / "mlr_eval.py"
    processes: list[subprocess.Popen[bytes]] = []
    for shard in range(args.num_shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), PYTHONUNBUFFERED="1")
        processes.append(subprocess.Popen([
            args.python, str(script), "--jobs", str(args.output_dir / "jobs.json"),
            "--shard-index", str(shard), "--num-shards", str(args.num_shards),
            "--output", str(args.output_dir / f"part{shard}.json"),
            "--frame-count", str(args.frame_count),
        ], env=env))
    rc = 0
    for process in processes:
        rc |= process.wait()
    if rc:
        return rc
    report = merge(args.output_dir, jobs, audit, args.num_shards)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
