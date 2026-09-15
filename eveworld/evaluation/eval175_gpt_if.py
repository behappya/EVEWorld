#!/usr/bin/env python3
"""Run DreamGenBench GPT-IF on audited EVAL-175 manifests."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image


FIELDS = [
    "key",
    "model",
    "split",
    "index",
    "request_id",
    "video_path",
    "prompt",
    "prediction",
    "raw_text",
    "response_model",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-errors", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_prediction(text: str) -> int:
    value = (text or "").strip()
    if value.startswith("1"):
        return 1
    if value.startswith("0"):
        return 0
    match = re.search(r"\b([01])\b", value)
    return int(match.group(1)) if match else 0


def prompt_text(instruction: str) -> str:
    return (
        "The video shows a robot arm completing a specific task. "
        f"Please evaluate: if the video follows the instruction to finish the task '{instruction}', give a positive score. "
        "Reply only '0' for No or '1' for Yes."
    )


def frame_uri(frame: Any) -> str:
    buffer = BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def evaluate(row: dict[str, Any], args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    from dreamgenbench.utils import sample_video_frames

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    frames = sample_video_frames(row["video_path"], num_frames=8, scale_factor=0.5)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text(row["prompt"])}]
    content.extend({"type": "image_url", "image_url": {"url": frame_uri(frame)}} for frame in frames)
    raw_text = ""
    response_model = ""
    error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content}],
                seed=args.seed,
                temperature=0.0,
                top_p=1.0,
                max_tokens=4,
            )
            raw_text = (response.choices[0].message.content or "").strip()
            response_model = str(getattr(response, "model", "") or "")
            error = ""
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(2 ** (attempt - 1))
    return {
        **row,
        "prediction": parse_prediction(raw_text) if not error else 0,
        "raw_text": raw_text,
        "response_model": response_model,
        "error": error,
    }


def read_done(path: Path, rerun_errors: bool) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["key"]
            for row in csv.DictReader(handle)
            if row.get("key") and not (rerun_errors and row.get("error"))
        }


def prune_error_rows(path: Path) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if not row.get("error")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable {args.api_key_env}")
    manifest = args.manifest.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    rows = read_jsonl(manifest)
    if args.start_offset:
        rows = rows[args.start_offset :]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not args.resume and output_csv.exists():
        output_csv.unlink()
    elif args.rerun_errors:
        prune_error_rows(output_csv)
    done = read_done(output_csv, args.rerun_errors) if args.resume else set()
    pending = [row for row in rows if row["key"] not in done]
    if not pending:
        print(f"Nothing to do; {len(done)} keys already present in {output_csv}")
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(evaluate, row, args, api_key): row for row in pending}
        for future in as_completed(futures):
            row = future.result()
            append_rows(output_csv, [row])
            completed += 1
            print(
                f"[gpt_if] {completed}/{len(pending)} {row['key']} pred={row['prediction']} error={bool(row['error'])}",
                flush=True,
            )

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    positive = sum(int(row["prediction"]) for row in output_rows)
    summary = {
        "manifest": str(manifest),
        "output_csv": str(output_csv),
        "model": args.model,
        "seed": args.seed,
        "count": len(output_rows),
        "positive": positive,
        "errors": sum(bool(row.get("error")) for row in output_rows),
        "score": positive / len(output_rows) if output_rows else None,
        "response_models": sorted({row.get("response_model", "") for row in output_rows if row.get("response_model")}),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
