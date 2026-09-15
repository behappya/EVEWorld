#!/usr/bin/env python3
"""Evaluate DreamGenBench generated videos with a Gemini physical-process judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google.genai import types as gt

import gemini_consensus_judge as gemini_ref  # noqa: E402


DEFAULT_ROUND0_MANIFEST = Path(
    "/data/datasets/gagi/eve_v2_outputs/eval175_eval/manifests/round0.jsonl"
)
DEFAULT_EVE_MANIFEST = Path(
    "/data/datasets/gagi/eve_v2_outputs/eval175_eval_extended_s300/manifests/"
    "t4g_wmapA_pre_seed42_s250.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/datasets/gagi/eve_v2_outputs/gemini_eval/dreamgen_process_v1"
)
DEFAULT_MODEL = os.getenv("DIFROST_MODEL", gemini_ref.DEFAULT_MODEL)

SYSTEM_PROMPT = """You are a strict evaluator of embodied robot video rollouts.
Use only the provided video and instruction. Do not infer success from the final
frame alone: inspect the full temporal process. Return only JSON matching the
requested schema. Use integer 0 or 1 for all binary fields.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "instruction_completion": {"type": "integer"},
        "physical_validity": {"type": "integer"},
        "target_conservation": {"type": "integer"},
        "duplicate_shortcut": {"type": "integer"},
        "confidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "instruction_completion",
        "physical_validity",
        "target_conservation",
        "duplicate_shortcut",
        "confidence",
        "reason",
    ],
}


def build_prompt(instruction: str) -> str:
    return f"""Evaluate this embodied video rollout against the instruction below.

Instruction:
{instruction}

Use these definitions:
- instruction_completion: 1 only if the instructed final state/action is visibly achieved.
- physical_validity: 1 only if the interaction is temporally and physically plausible,
  with no obvious teleportation, clipping, impossible contact, or unsupported state jump.
- target_conservation: 1 if the manipulated target remains the same visible instance and
  its count/identity is consistent throughout the interaction. For non-transport tasks,
  judge the relevant manipulated object or state consistently.
- duplicate_shortcut: 1 if the video reaches the goal by creating or revealing an extra
  target instance while the original remains, rather than completing the interaction.

Return JSON only. The reason should be one or two concise sentences grounded in visible
events. Do not mark a video invalid merely because the task fails; separate failure from
physical validity and target conservation.
"""


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_json(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates.extend(fenced)
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_result(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in (
        "instruction_completion",
        "physical_validity",
        "target_conservation",
        "duplicate_shortcut",
    ):
        raw = value.get(key)
        try:
            result[key] = int(raw)
        except (TypeError, ValueError):
            return None
        if result[key] not in (0, 1):
            return None
    result["confidence"] = str(value.get("confidence") or "unknown")
    result["reason"] = str(value.get("reason") or "")
    return result


def evaluate_one(
    item: dict[str, Any],
    *,
    model: str,
    timeout: float,
    retries: int,
    thinking_level: str,
    include_thoughts: bool,
) -> dict[str, Any]:
    video_path = Path(item["video_path"])
    base = {
        "key": item["key"],
        "model": item["model"],
        "split": item["split"],
        "index": item["index"],
        "request_id": item["request_id"],
        "prompt": item["prompt"],
        "video_path": str(video_path),
        "gemini_model": model,
    }
    if not video_path.is_file():
        return {**base, "error": f"missing_video: {video_path}"}

    config = gt.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=3000,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=SCHEMA,
        thinking_config=gt.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        ),
    )
    contents = [
        gt.Content(
            role="user",
            parts=[
                gt.Part.from_bytes(data=video_path.read_bytes(), mime_type="video/mp4"),
                gt.Part.from_text(text=build_prompt(item["prompt"])),
            ],
        )
    ]
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            client = gemini_ref.get_gemini_client(timeout)
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            texts, thoughts = gemini_ref.response_text_parts(response)
            raw_text = "\n\n".join(texts) or getattr(response, "text", "") or ""
            parsed = normalize_result(parse_json(raw_text))
            if parsed is None:
                raise ValueError(f"invalid structured response: {raw_text[:500]}")
            usage = getattr(response, "usage_metadata", None)
            return {
                **base,
                **parsed,
                "raw_response": raw_text,
                "raw_thoughts": "\n\n".join(thoughts),
                "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "candidates_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
                "elapsed_s": round(time.time() - started, 3),
                "retry_count": attempt,
                "error": "",
            }
        except Exception as exc:  # retry transient API and parsing failures
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(3.0 * attempt, 15.0))
    return {**base, "error": last_error, "retry_count": retries}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = (
        "instruction_completion",
        "physical_validity",
        "target_conservation",
        "duplicate_shortcut",
    )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        return {
            "n": n,
            **{
                key: sum(row[key] for row in rows) / n if n else None
                for key in metric_keys
            },
        }

    summary: dict[str, Any] = {"records": len(records), "errors": 0, "models": {}}
    for model in sorted({str(r.get("model")) for r in records}):
        rows = [r for r in records if r.get("model") == model]
        good = [r for r in rows if not r.get("error")]
        model_summary: dict[str, Any] = {
            "records": len(rows),
            "successful": len(good),
            "errors": len(rows) - len(good),
            "overall": aggregate(good),
            "splits": {},
        }
        for split in ("gr1_env", "gr1_object", "gr1_behavior"):
            subset = [r for r in good if r.get("split") == split]
            model_summary["splits"][split] = aggregate(subset)
        summary["models"][model] = model_summary
    summary["errors"] = sum(1 for r in records if r.get("error"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round0-manifest", type=Path, default=DEFAULT_ROUND0_MANIFEST)
    parser.add_argument("--eve-manifest", type=Path, default=DEFAULT_EVE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--thinking-level", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--include-thoughts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    items = read_manifest(args.round0_manifest) + read_manifest(args.eve_manifest)
    if args.limit:
        items = items[: args.limit]
    if len(items) != 252 and not args.limit:
        raise RuntimeError(f"expected 252 manifest rows, found {len(items)}")
    for item in items:
        if not Path(item["video_path"]).is_file():
            raise FileNotFoundError(item["video_path"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"
    config_path = args.output_dir / "run_config.json"
    existing: dict[str, dict[str, Any]] = {}
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not row.get("error"):
                        existing[row["key"]] = row
    pending = [item for item in items if item["key"] not in existing]
    config_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "round0_manifest": str(args.round0_manifest),
                "eve_manifest": str(args.eve_manifest),
                "records": len(items),
                "concurrency": args.concurrency,
                "retries": args.retries,
                "timeout": args.timeout,
                "thinking_level": args.thinking_level,
                "include_thoughts": args.include_thoughts,
                "judge": "Gemini-Process v1; independent of VideoPhy PA-II",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_lock = threading.Lock()
    with records_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    evaluate_one,
                    item,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                    thinking_level=args.thinking_level,
                    include_thoughts=args.include_thoughts,
                ): item
                for item in pending
            }
            for position, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                with write_lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                if not row.get("error"):
                    existing[row["key"]] = row
                print(f"[{position}/{len(pending)}] {row['key']} error={bool(row.get('error'))}", flush=True)

    all_rows = list(existing.values())
    summary_path.write_text(json.dumps(summarize(all_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summarize(all_rows), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
