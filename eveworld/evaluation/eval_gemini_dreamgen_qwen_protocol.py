#!/usr/bin/env python3
"""Evaluate DreamGenBench with Gemini under the existing Qwen protocol."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google.genai import types as gt

QWEN_SCRIPT_DIR = Path("benchmarks/dreamgenbench")
if str(QWEN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_SCRIPT_DIR))

import gemini_consensus_judge as gemini_ref  # noqa: E402
import eval_dreamgenbench_qwen_api as qwen_protocol  # noqa: E402


DEFAULT_MANIFESTS = (
    Path(
        "/data/datasets/gagi/eve_v2_outputs/eval175_eval/manifests/"
        "round0.jsonl"
    ),
    Path(
        "/data/datasets/gagi/eve_v2_outputs/eval175_eval_extended_s300/"
        "manifests/t4g_wmapA_pre_seed42_s250.jsonl"
    ),
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/datasets/gagi/eve_v2_outputs/gemini_eval/"
    "dreamgen_qwen_protocol_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini with the exact DreamGenBench Qwen-IF/PA-I protocol."
    )
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--metrics", default="qwen_if,pa_i")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--frame-count", type=int, default=49)
    parser.add_argument("--max-image-side", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--model-retries", type=int, default=4)
    parser.add_argument("--model-timeout", type=float, default=1200.0)
    parser.add_argument("--model-max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking-level", choices=["low", "medium", "high"], default="low"
    )
    parser.add_argument(
        "--include-thoughts", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_items(manifests: list[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for manifest in manifests:
        loader_args = argparse.Namespace(manifest=manifest, video_dir=Path())
        items.extend(qwen_protocol.load_eval_items(loader_args))
    return items


def jpeg_frames(video_path: Path, args: argparse.Namespace) -> list[bytes]:
    urls = qwen_protocol.video_frame_data_urls(video_path, args)
    return [base64.b64decode(url.split(",", 1)[1]) for url in urls]


def evaluate_one(
    *,
    metric: str,
    item: dict[str, Any],
    frames: list[bytes],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    video_path = Path(item["video_path"])
    prompt = str(item["prompt"])
    parts = [
        gt.Part.from_text(text=qwen_protocol.metric_prompt(metric, prompt)),
        *(gt.Part.from_bytes(data=frame, mime_type="image/jpeg") for frame in frames),
    ]
    contents = [gt.Content(role="user", parts=parts)]
    config = gt.GenerateContentConfig(
        max_output_tokens=args.model_max_tokens,
        temperature=args.temperature,
        thinking_config=gt.ThinkingConfig(
            thinking_level=args.thinking_level,
            include_thoughts=args.include_thoughts,
        ),
    )
    raw_text = ""
    last_error = ""
    for retry in range(1, args.model_retries + 1):
        try:
            client = gemini_ref.get_gemini_client(args.model_timeout)
            response = client.models.generate_content(
                model=args.model,
                contents=contents,
                config=config,
            )
            texts, _ = gemini_ref.response_text_parts(response)
            raw_text = "\n".join(texts) or getattr(response, "text", "") or ""
            return {
                "key": item.get("key", ""),
                "model": item.get("model", ""),
                "split": item.get("split", ""),
                "index": item.get("index", ""),
                "request_id": item.get("request_id", ""),
                "video_path": str(video_path),
                "prompt": prompt,
                "prediction": qwen_protocol.parse_prediction(raw_text),
                "raw_text": raw_text,
                "error": None,
                "model_retry_count": retry,
                "latency_sec": round(time.time() - started, 4),
                "created_at": qwen_protocol.now_iso(),
            }
        except Exception as exc:  # retry transient API failures
            last_error = str(exc)
            if retry < args.model_retries:
                time.sleep(min(2.0 * retry, 8.0))
    return {
        "key": item.get("key", ""),
        "model": item.get("model", ""),
        "split": item.get("split", ""),
        "index": item.get("index", ""),
        "request_id": item.get("request_id", ""),
        "video_path": str(video_path),
        "prompt": prompt,
        "prediction": 0,
        "raw_text": raw_text,
        "error": f"model failed after {args.model_retries} retries: {last_error}",
        "model_retry_count": args.model_retries,
        "latency_sec": round(time.time() - started, 4),
        "created_at": qwen_protocol.now_iso(),
    }


def summarize_by_model(path: Path) -> dict[str, Any]:
    import csv

    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    result: dict[str, Any] = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        values = [int(float(row["prediction"])) for row in subset]
        result[model] = {
            "count": len(subset),
            "positive": sum(values),
            "errors": sum(bool(row.get("error")) for row in subset),
            "score": sum(values) / len(values) if values else None,
        }
    return result


def main() -> None:
    args = parse_args()
    using_default_manifests = not args.manifest
    manifests = args.manifest or list(DEFAULT_MANIFESTS)
    items = load_items(manifests)
    if args.limit:
        items = items[: args.limit]
    if not args.limit:
        if using_default_manifests and len(items) != 252:
            raise RuntimeError(f"expected 252 videos, found {len(items)}")
        if not using_default_manifests and (not items or len(items) % 126 != 0):
            raise RuntimeError(
                f"custom manifests must contain complete 126-video groups; found {len(items)}"
            )

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    unknown = sorted(set(metrics) - {"qwen_if", "pa_i"})
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "manifests": [str(path.resolve()) for path in manifests],
        "records": len(items),
        "metrics": metrics,
        "concurrency": args.concurrency,
        "frame_count": args.frame_count,
        "max_image_side": args.max_image_side,
        "jpeg_quality": args.jpeg_quality,
        "model_retries": args.model_retries,
        "model_timeout": args.model_timeout,
        "model_max_tokens": args.model_max_tokens,
        "temperature": args.temperature,
        "thinking_level": args.thinking_level,
        "include_thoughts": args.include_thoughts,
        "resume": args.resume,
        "rerun_errors": args.rerun_errors,
        "limit": args.limit,
        "protocol_source": str(QWEN_SCRIPT_DIR / "eval_dreamgenbench_qwen_api.py"),
        "protocol_note": "Exact Qwen metric prompts, frame sampling, and binary parser; Gemini backend only.",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Retaining 49 JPEGs per video is useful for the two-model run but too large
    # for multi-seed sweeps containing thousands of videos.
    use_frame_cache = len(items) <= 512
    frame_cache: dict[Path, list[bytes]] = {}
    for metric in metrics:
        csv_path = args.output_dir / f"{metric}.csv"
        if not args.resume and csv_path.exists():
            csv_path.unlink()
        elif args.rerun_errors:
            qwen_protocol.prune_error_rows(csv_path)
        done = qwen_protocol.read_done(csv_path, args.rerun_errors) if args.resume else set()
        pending = [item for item in items if item["key"] not in done]
        print(f"[{metric}] pending {len(pending)}/{len(items)}", flush=True)

        def run(item: dict[str, Any]) -> dict[str, Any]:
            path = Path(item["video_path"])
            if use_frame_cache:
                frames = frame_cache.get(path)
                if frames is None:
                    frames = jpeg_frames(path, args)
                    frame_cache[path] = frames
            else:
                frames = jpeg_frames(path, args)
            return evaluate_one(metric=metric, item=item, frames=frames, args=args)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run, item): item for item in pending}
            completed = len(items) - len(pending)
            for future in as_completed(futures):
                row = future.result()
                qwen_protocol.append_csv(csv_path, [row])
                completed += 1
                print(
                    f"[{metric}] {completed}/{len(items)} {row['key']} "
                    f"pred={row['prediction']} error={bool(row['error'])}",
                    flush=True,
                )

    summary = {
        metric: summarize_by_model(args.output_dir / f"{metric}.csv")
        for metric in metrics
    }
    summary["protocol"] = "DreamGenBench Qwen-IF/PA-I with Gemini backend"
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
