#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


TRANSPORT_VERBS = re.compile(
    r"\b(pick(?:ing)?(?:\s+up)?|grab(?:bing)?|lift(?:ing)?|tak(?:e|ing)|"
    r"mov(?:e|ing)|plac(?:e|ing)|set(?:ting)?|hold(?:ing)?|position(?:ing)?|"
    r"put(?:ting)?|transfer(?:ring)?|bring(?:ing)?|carr(?:y|ying)|rais(?:e|ing)|"
    r"grasp(?:ing)?|shift(?:ing)?|relocat(?:e|ing)|slid(?:e|ing)|catch(?:ing)?|"
    r"secur(?:e|ing)|grip(?:ping)?|stack(?:ing)?|seiz(?:e|ing)|transport(?:ing)?|"
    r"lay(?:ing)?|drop(?:ping)?|remov(?:e|ing))\b",
    re.IGNORECASE,
)

# Canonical open-vocabulary queries. The first class mentioned after the first
# transport verb is used; fixture-only actions such as pressing/opening are excluded.
OBJECT_PATTERNS = {
    "tea box": r"\btea\s+box\b",
    "playing cards": r"\b(?:playing\s*cards?|playingcards|card box)\b",
    "toy car": r"\b(?:toy\s*car|toycar|car)\b",
    "payment sign": r"\b(?:payment\s*sign|paymentsign|qr\s*(?:code\s*)?sign)\b",
    "kitchen pot": r"\b(?:kitchen\s*pot|kitchenpot|cooking\s+pot)\b",
    "electronic scale": r"\b(?:electronic\s*scale|electronicscale)\b",
    "display stand": r"\b(?:display\s*stand|displaystand|phone\s+stand|phonestand)\b",
    "bread basket": r"\b(?:bread\s*basket|breadbasket)\b",
    "trash bin": r"\b(?:trash\s*bin|trashbin|dustbin)\b",
    "hamburger": r"\b(?:hamburger|hamburg)\b",
    "beverage can": r"\b(?:soda|beverage|drink)?\s*can\b",
    "bottle": r"\bbottle\b",
    "block": r"\bblock\b",
    "bowl": r"\bbowl\b",
    "hammer": r"\bhammer\b",
    "fries": r"\bfries\b",
    "microphone": r"\bmicrophone\b",
    "bread": r"\b(?:bread|loaf)\b",
    "fan": r"\bfan\b",
    "mouse": r"\bmouse\b",
    "basket": r"\bbasket\b",
    "shoe": r"\b(?:shoe|footwear)\b",
    "laptop": r"\b(?:laptop|computer)\b",
    "roller": r"\broller\b",
    "cup": r"\b(?:cup|mug)\b",
    "phone": r"\b(?:phone|smartphone)\b",
    "scanner": r"\bscanner\b",
    "kettle": r"\b(?:kettle|teapot)\b",
    "stapler": r"\bstapler\b",
    "clock": r"\b(?:clock|alarm-clock)\b",
    "soap": r"\bsoap\b",
    "mat": r"\bmat\b",
    "tray": r"\btray\b",
    "plate": r"\bplate\b",
    "skillet": r"\b(?:skillet|pan)\b",
    "box": r"\bbox\b",
    "container": r"\bcontainer\b",
    "rack": r"\brack\b",
    "lid": r"\blid\b",
    "bell": r"\bbell\b",
}
COMPILED_OBJECTS = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in OBJECT_PATTERNS.items()
}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def task_clause(prompt: str) -> str:
    parts = re.split(r"enters the frame to\s*", prompt, maxsplit=1, flags=re.IGNORECASE)
    return parts[-1].strip()


def extract_mover(prompt: str) -> tuple[str | None, str]:
    clause = task_clause(prompt)
    verb = TRANSPORT_VERBS.search(clause)
    if verb is None:
        return None, "no_transport_verb"
    tail = clause[verb.end() :]
    matches = []
    for name, pattern in COMPILED_OBJECTS.items():
        match = pattern.search(tail)
        if match is not None:
            matches.append((match.start(), -len(match.group(0)), name))
    if not matches:
        return None, "no_supported_object"
    matches.sort()
    return matches[0][2], "eligible"


def wilson(events: int, count: int, z: float = 1.96) -> list[float] | None:
    if count == 0:
        return None
    rate = events / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = []
    excluded = Counter()
    movers = Counter()
    for row in manifest:
        mover, reason = extract_mover(row["prompt"])
        if mover is None:
            excluded[reason] += 1
            continue
        selected.append((row, mover))
        movers[mover] += 1
    if args.limit is not None:
        selected = selected[: args.limit]
    jobs = []
    missing = []
    for model in args.models:
        video_dir = args.video_root / f"{model}_test"
        for row, mover in selected:
            video = video_dir / f"{row['request_id']}.mp4"
            condition_image = Path(row["image"])
            if not video.is_file() or not condition_image.is_file():
                missing.append(str(video if not video.is_file() else condition_image))
                continue
            jobs.append(
                {
                    "model": model,
                    "request_id": row["request_id"],
                    "prompt": row["prompt"],
                    "mover": mover,
                    "video": str(video.resolve()),
                    "condition_image": str(condition_image.resolve()),
                }
            )
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} inputs; examples: {missing[:5]}")
    audit = {
        "protocol": "worldarena1_mlr_gdino_v2",
        "manifest": str(args.manifest.resolve()),
        "manifest_count": len(manifest),
        "selected_prompt_count": len(selected),
        "models": args.models,
        "job_count": len(jobs),
        "excluded": dict(sorted(excluded.items())),
        "mover_counts_before_limit": dict(sorted(movers.items())),
        "limit": args.limit,
    }
    return jobs, audit


def merge(args: argparse.Namespace, jobs: list[dict[str, str]], audit: dict[str, Any]) -> int:
    records = []
    for shard in range(args.num_shards):
        path = args.output_dir / f"part{shard}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["records"])
    expected_keys = {(job["model"], job["request_id"]) for job in jobs}
    actual_keys = {(row["model"], row["request_id"]) for row in records}
    if actual_keys != expected_keys or len(records) != len(jobs):
        raise RuntimeError(
            f"Record coverage mismatch: actual={len(actual_keys)} expected={len(expected_keys)}"
        )

    summaries = {}
    selected_count = audit["selected_prompt_count"]
    for model in args.models:
        model_rows = [row for row in records if row["model"] == model]
        errors = [row for row in model_rows if row["error"] is not None]
        eligible = [row for row in model_rows if row["eligible"] and row["error"] is None]
        events = sum(bool(row["mlr_event"]) for row in eligible)
        summaries[model] = {
            "selected_prompts": selected_count,
            "records": len(model_rows),
            "eligible": len(eligible),
            "coverage": len(eligible) / selected_count if selected_count else 0.0,
            "events": events,
            "mlr": events / len(eligible) if eligible else None,
            "wilson_95": wilson(events, len(eligible)),
            "errors": len(errors),
        }

    paper_gate = None
    if len(args.models) == 3:
        pretrain, sft, eve = (summaries[model] for model in args.models)
        comparable = (
            pretrain["errors"] == sft["errors"] == eve["errors"] == 0
            and pretrain["eligible"] == sft["eligible"] == eve["eligible"]
            and eve["eligible"] > 0
        )
        relative_reduction = None
        if comparable and sft["mlr"] is not None and sft["mlr"] > 0:
            relative_reduction = 1.0 - eve["mlr"] / sft["mlr"]
        paper_gate = {
            "comparable_coverage": comparable,
            "eve_better_than_sft": bool(
                comparable
                and eve["mlr"] is not None
                and sft["mlr"] is not None
                and eve["mlr"] < sft["mlr"]
            ),
            "eve_vs_sft_relative_reduction": relative_reduction,
        }
    payload = {
        "metadata": audit,
        "summary": summaries,
        "paper_gate": paper_gate,
        "records": records,
    }
    atomic_write_json(args.output_dir / "summary.json", payload)
    print(json.dumps({"summary": summaries, "paper_gate": paper_gate}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch and aggregate WorldArena 1.0 MLR")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs, audit = build_jobs(args)
    atomic_write_json(args.output_dir / "jobs.json", jobs)
    atomic_write_json(args.output_dir / "parser_audit.json", audit)
    script = Path(__file__).with_name("mlr_eval.py")
    processes = []
    for shard in range(args.num_shards):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), PYTHONUNBUFFERED="1")
        processes.append(
            subprocess.Popen(
                [
                    args.python,
                    str(script),
                    "--jobs",
                    str(args.output_dir / "jobs.json"),
                    "--shard-index",
                    str(shard),
                    "--num-shards",
                    str(args.num_shards),
                    "--output",
                    str(args.output_dir / f"part{shard}.json"),
                ],
                env=environment,
            )
        )
    return_code = 0
    for process in processes:
        return_code |= process.wait()
    if return_code:
        raise SystemExit(return_code)
    raise SystemExit(merge(args, jobs, audit))


if __name__ == "__main__":
    main()
