#!/usr/bin/env python3
"""Run official DreamGenBench Qwen-IF and PA-I from an audited manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
FIELDS = ["key", "model", "split", "index", "request_id", "video_path", "prompt", "prediction", "raw_text", "error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_QWEN)
    parser.add_argument("--metrics", default="qwen_if,pa_i", help="Comma-separated qwen_if,pa_i")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-errors", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_prediction(text: str) -> int:
    value = (text or "").strip()
    if value.startswith("1"):
        return 1
    if value.startswith("0"):
        return 0
    match = re.search(r"\b([01])\b", value)
    return int(match.group(1)) if match else 0


def qwen_if_prompt(instruction: str) -> str:
    return (
        "The video shows a robot arm completing a specific task. "
        f"Does the video follow the instruction to finish the task: '{instruction}'? "
        "If it fails to follow the instruction (e.g. miss the object, action or do some other actions), please answer 0. "
        "Answer 0 for No or 1 for Yes. Reply only 0 or 1."
    )


def pa_i_prompt() -> str:
    return (
        "The video shows a robot arm completing a specific task. "
        "Does the video show good physics dynamics and showcase a good alignment with the physical world? "
        "Please be a strict judge. If it breaks the laws of physics, please answer 0. "
        "Answer 0 for No or 1 for Yes. Reply only 0 or 1."
    )


def decode_reply(model: Any, processor: Any, inputs: Any) -> str:
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=4)
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated_ids)]
    replies = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return replies[0].strip()


def evaluate_qwen_if(row: dict[str, Any], model: Any, processor: Any, device: str) -> str:
    from dreamgenbench.utils import sample_video_frames

    frames = sample_video_frames(row["video_path"], num_frames=49, scale_factor=0.3)
    content = [{"type": "text", "text": qwen_if_prompt(row["prompt"])}]
    content.extend({"type": "image", "image": frame} for frame in frames)
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(device)
    return decode_reply(model, processor, inputs)


def evaluate_pa_i(row: dict[str, Any], model: Any, processor: Any, device: str) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": row["video_path"]},
                {"type": "text", "text": pa_i_prompt()},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return decode_reply(model, processor, inputs)


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


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in FIELDS})


def summarize(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "count": 0, "positive": 0, "errors": 0, "score": None}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    positive = sum(int(row["prediction"]) for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    return {
        "path": str(path),
        "count": len(rows),
        "positive": positive,
        "errors": errors,
        "score": positive / len(rows) if rows else None,
    }


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    unknown = sorted(set(metrics) - {"qwen_if", "pa_i"})
    if unknown:
        raise SystemExit(f"Unknown metrics: {unknown}")
    rows = read_jsonl(manifest)
    if args.start_offset:
        rows = rows[args.start_offset :]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows selected from {manifest}")

    set_seed(args.seed)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype="auto",
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    evaluators = {"qwen_if": evaluate_qwen_if, "pa_i": evaluate_pa_i}

    for metric in metrics:
        output_csv = output_dir / f"{metric}.csv"
        if not args.resume and output_csv.exists():
            output_csv.unlink()
        elif args.rerun_errors:
            prune_error_rows(output_csv)
        done = read_done(output_csv, args.rerun_errors) if args.resume else set()
        for position, row in enumerate(rows, start=1):
            if row["key"] in done:
                continue
            raw_text = ""
            error = ""
            try:
                raw_text = evaluators[metric](row, model, processor, args.device)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            record = {
                **row,
                "prediction": parse_prediction(raw_text) if not error else 0,
                "raw_text": raw_text,
                "error": error,
            }
            append_row(output_csv, record)
            print(
                f"[{metric}] {position}/{len(rows)} {row['key']} pred={record['prediction']} error={bool(error)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "manifest": str(manifest),
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "metrics": {metric: summarize(output_dir / f"{metric}.csv") for metric in metrics},
    }
    (output_dir / "qwen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
