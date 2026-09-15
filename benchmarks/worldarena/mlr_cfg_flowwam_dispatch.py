#!/usr/bin/env python3
"""Run the frozen GDino MLR protocol over CFG and FlowWAM outputs."""
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


def load_dreamgen_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("env", "object", "behavior"):
        rows.extend(json.loads((root / f"eval175_gr1_{split}.json").read_text()))
    if len(rows) != 126:
        raise ValueError(f"expected 126 DreamGen rows, got {len(rows)}")
    return rows


def select_movers(rows: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], str]], dict[str, int]]:
    selected: list[tuple[dict[str, Any], str]] = []
    excluded: Counter[str] = Counter()
    for row in rows:
        mover, reason = extract_mover(str(row["prompt"]))
        if mover is None:
            excluded[reason] += 1
        else:
            selected.append((row, mover))
    return selected, dict(sorted(excluded.items()))


def add_job(jobs: list[dict[str, str]], *, model: str, request_id: str,
            prompt: str, mover: str, video: Path, image: Path) -> None:
    if not video.is_file():
        raise FileNotFoundError(video)
    if not image.is_file():
        raise FileNotFoundError(image)
    jobs.append({
        "model": model,
        "request_id": request_id,
        "prompt": prompt,
        "mover": mover,
        "video": str(video.resolve()),
        "condition_image": str(image.resolve()),
    })


def build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    dream_rows = load_dreamgen_rows(args.dreamgen_data_root)
    selected, excluded = select_movers(dream_rows)
    flow_rows = json.loads(args.flow_manifest.read_text())
    if len(flow_rows) != 250:
        raise ValueError(f"expected 250 FlowWAM rows, got {len(flow_rows)}")
    flow_selected, flow_excluded = select_movers(flow_rows)
    jobs: list[dict[str, str]] = []

    for step in (50, 100, 150, 200, 250, 300):
        for cfg, tag in ((1.0, "cfg_1"), (2.5, "cfg_2p5"),
                         (5.0, "cfg_5"), (7.5, "cfg_7p5")):
            model = f"step_{step:03d}_{tag}"
            directory = args.cfg_root / f"step_{step:03d}" / tag / "generated_only"
            for row, mover in selected:
                add_job(
                    jobs, model=model, request_id=str(row["request_id"]),
                    prompt=str(row["prompt"]), mover=mover,
                    video=directory / f"{row['request_id']}.mp4",
                    image=Path(row["image"]),
                )

    for arm in ("control", "eve"):
        model = f"flowwam_{arm}_heldout_r250"
        directory = args.flow_root / f"arm_{arm}_final_robot_only"
        for row, mover in flow_selected:
            add_job(
                jobs, model=model, request_id=str(row["request_id"]),
                prompt=str(row["prompt"]), mover=mover,
                video=directory / f"{row['request_id']}.mp4",
                image=Path(row["image"]),
            )
    audit = {
        "protocol": "dreamgen_cfg_and_flowwam_mlr_gdino_v2",
        "dreamgen_rows": len(dream_rows),
        "dreamgen_selected": len(selected),
        "dreamgen_excluded": excluded,
        "flowwam_rows": len(flow_rows),
        "flowwam_selected": len(flow_selected),
        "flowwam_excluded": flow_excluded,
        "models": sorted({job["model"] for job in jobs}),
        "job_count": len(jobs),
        "num_shards": args.num_shards,
        "frame_count": args.frame_count,
    }
    return jobs, audit


def merge(args: argparse.Namespace, jobs: list[dict[str, str]], audit: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for shard in range(args.num_shards):
        payload = json.loads((args.output_dir / f"part{shard}.json").read_text())
        records.extend(payload["records"])
    expected = {(job["model"], job["request_id"]) for job in jobs}
    actual = {(row.get("model"), row.get("request_id")) for row in records}
    if len(records) != len(jobs) or actual != expected:
        raise RuntimeError(f"record coverage mismatch: actual={len(actual)} expected={len(expected)}")
    summaries: dict[str, Any] = {}
    for model in sorted({job["model"] for job in jobs}):
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
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dreamgen-data-root", type=Path, required=True)
    ap.add_argument("--cfg-root", type=Path, required=True)
    ap.add_argument("--flow-manifest", type=Path, required=True)
    ap.add_argument("--flow-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--frame-count", type=int, default=24)
    args = ap.parse_args()
    if args.num_shards != 8:
        raise ValueError("this runner requires eight GPUs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs, audit = build_jobs(args)
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
    report = merge(args, jobs, audit)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
