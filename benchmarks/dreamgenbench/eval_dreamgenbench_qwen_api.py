#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2


DEFAULT_VIDEO_DIR = Path("/data/datasets/gagi/gr1_dreamgen_eval/dreamgenbench_video_dirs/gr1_dreamgen_8gpu_full_20260625_212933")
DEFAULT_OUTPUT_ROOT = Path("/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs")
DEFAULT_QWEN_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B"

_TLS = threading.local()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DreamGenBench videos with an OpenAI-compatible Qwen VL endpoint.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest with video_path and full prompt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=f"dreamgen_qwen_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--qwen-base", type=str, default=DEFAULT_QWEN_BASE)
    parser.add_argument("--qwen-model", type=str, default="auto")
    parser.add_argument("--metrics", type=str, default="qwen_if,pa_i", help="Comma-separated metrics: qwen_if,pa_i")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--frame-count", type=int, default=49)
    parser.add_argument("--max-image-side", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--parallel-frame-decode",
        action="store_true",
        help="Decode frames inside evaluation workers instead of the submit thread.",
    )
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--model-timeout", type=float, default=600.0)
    parser.add_argument("--model-max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-errors", action="store_true")
    return parser.parse_args()


def normalize_qwen_base(base: str) -> str:
    value = str(base or "").strip()
    if not value:
        raise ValueError("empty qwen base")
    if value.startswith("http://") or value.startswith("https://"):
        out = value.rstrip("/")
        parsed = urlparse(out)
        if not parsed.path or parsed.path == "/":
            out = out + "/v1"
        return out
    host_port = value if ":" in value else f"{value}:8000"
    return f"http://{host_port}/v1"


def ensure_no_proxy(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [item.strip() for item in current.split(",") if item.strip()]
        if host not in parts:
            parts.append(host)
        os.environ[key] = ",".join(parts)


def get_openai_client(base_url: str):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing Python package: openai. Install it in the EVEWorld conda env.") from exc
    if not hasattr(_TLS, "openai_clients"):
        _TLS.openai_clients = {}
    if base_url not in _TLS.openai_clients:
        _TLS.openai_clients[base_url] = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    return _TLS.openai_clients[base_url]


def resolve_model(base_url: str, requested: str, timeout: float) -> str:
    if requested and requested != "auto":
        return requested
    try:
        client = get_openai_client(base_url)
        models = client.models.list(timeout=min(timeout, 20.0))
        for item in getattr(models, "data", []) or []:
            model_id = getattr(item, "id", None)
            if model_id:
                return str(model_id)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not auto-detect qwen model id, using fallback {DEFAULT_QWEN_MODEL}: {exc}", flush=True)
    return DEFAULT_QWEN_MODEL


def extract_text_from_message(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def parse_prediction(text: str) -> int:
    value = "" if text is None else str(text).strip()
    if value.startswith("1"):
        return 1
    if value.startswith("0"):
        return 0
    match = re.search(r"\b([01])\b", value)
    if match:
        return int(match.group(1))
    lowered = value.lower()
    if re.search(r"\byes\b", lowered) and not re.search(r"\bno\b", lowered):
        return 1
    return 0


def prompt_from_video_path(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return stem.replace("_", " ")


def load_eval_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest is None:
        videos = sorted(args.video_dir.expanduser().resolve().glob("**/*.mp4"))
        return [{"video_path": str(path), "prompt": prompt_from_video_path(path)} for path in videos]
    manifest = args.manifest.expanduser().resolve()
    items: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = [field for field in ("key", "video_path", "prompt") if field not in item]
            if missing:
                raise ValueError(f"{manifest}:{line_number}: missing fields {missing}")
            video_path = Path(item["video_path"]).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"{manifest}:{line_number}: missing video {video_path}")
            items.append({**item, "video_path": str(video_path)})
    return items


def resize_frame(frame: Any, max_side: int) -> Any:
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def frame_indices(total_frames: int, frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if total_frames <= 0:
        return list(range(frame_count))
    if frame_count == 1:
        return [total_frames // 2]
    return sorted({round(i * (total_frames - 1) / (frame_count - 1)) for i in range(frame_count)})


def video_frame_data_urls(video_path: Path, args: argparse.Namespace) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        urls: list[str] = []
        for idx in frame_indices(total, args.frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = resize_frame(frame, args.max_image_side)
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            if not ok:
                continue
            b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            urls.append(f"data:image/jpeg;base64,{b64}")
        if not urls:
            raise ValueError(f"no frames extracted from video: {video_path}")
        return urls
    finally:
        cap.release()


def metric_prompt(metric: str, prompt: str) -> str:
    if metric == "qwen_if":
        return (
            "The video shows a robot arm completing a specific task. "
            f"Does the video follow the instruction to finish the task: '{prompt}'? "
            "If it fails to follow the instruction (e.g. miss the object, action or do some other actions), please answer 0. "
            "Answer 0 for No or 1 for Yes. Reply only 0 or 1."
        )
    if metric == "pa_i":
        return (
            "The video shows a robot arm completing a specific task. "
            "Does the video show good physics dynamics and showcase a good alignment with the physical world? "
            "Please be a strict judge. If it breaks the laws of physics, please answer 0. "
            "Answer 0 for No or 1 for Yes. Reply only 0 or 1."
        )
    raise ValueError(f"unknown metric: {metric}")


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    disable_thinking: bool,
) -> Any:
    client = get_openai_client(base_url)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    kwargs["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": not disable_thinking}
    }
    try:
        return client.chat.completions.create(**kwargs)
    except Exception:
        if disable_thinking:
            kwargs.pop("extra_body", None)
            return client.chat.completions.create(**kwargs)
        raise


def evaluate_one(
    *,
    metric: str,
    item: dict[str, Any],
    frame_urls: list[str],
    base_url: str,
    model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    video_path = Path(item["video_path"])
    prompt = str(item["prompt"])
    content: list[dict[str, Any]] = [{"type": "text", "text": metric_prompt(metric, prompt)}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in frame_urls)
    messages = [{"role": "user", "content": content}]
    raw_text = ""
    last_error = ""
    for retry in range(1, args.model_retries + 1):
        try:
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.model_max_tokens,
                timeout=args.model_timeout,
                disable_thinking=args.disable_thinking,
            )
            raw_text = extract_text_from_message(response.choices[0].message.content)
            if not raw_text:
                raise ValueError("empty model response")
            return {
                "key": item.get("key", ""),
                "model": item.get("model", ""),
                "split": item.get("split", ""),
                "index": item.get("index", ""),
                "request_id": item.get("request_id", ""),
                "video_path": str(video_path),
                "prompt": prompt,
                "prediction": parse_prediction(raw_text),
                "raw_text": raw_text,
                "error": None,
                "model_retry_count": retry,
                "latency_sec": round(time.time() - started, 4),
                "created_at": now_iso(),
            }
        except Exception as exc:  # noqa: BLE001
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
        "created_at": now_iso(),
    }


def evaluate_one_with_frame_decode(
    *,
    metric: str,
    item: dict[str, Any],
    base_url: str,
    model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_urls = video_frame_data_urls(Path(item["video_path"]), args)
    return evaluate_one(
        metric=metric,
        item=item,
        frame_urls=frame_urls,
        base_url=base_url,
        model=model,
        args=args,
    )


def read_done(path: Path, rerun_errors: bool) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if rerun_errors and row.get("error"):
                continue
            identity = row.get("key") or row.get("video_path")
            if identity:
                done.add(identity)
    return done


def prune_error_rows(path: Path) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if not row.get("error") and row.get("raw_text", "").strip()
        ]
    path.unlink()
    append_csv(path, rows)


def append_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = [
        "key",
        "model",
        "split",
        "index",
        "request_id",
        "video_path",
        "prompt",
        "prediction",
        "raw_text",
        "error",
        "model_retry_count",
        "latency_sec",
        "created_at",
    ]
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(records)


def summarize_csv(path: Path) -> dict[str, Any]:
    values: list[int] = []
    errors = 0
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    values.append(int(float(row.get("prediction", "0"))))
                except ValueError:
                    values.append(0)
                if row.get("error"):
                    errors += 1
    return {
        "path": str(path),
        "exists": path.exists(),
        "count": len(values),
        "positive": sum(values),
        "error_count": errors,
        "score": (sum(values) / len(values)) if values else None,
    }


def write_config(args: argparse.Namespace, base_url: str, model: str, metrics: list[str], output_root: Path) -> None:
    payload = {
        "video_dir": str(args.video_dir),
        "manifest": str(args.manifest.expanduser().resolve()) if args.manifest is not None else None,
        "output_root": str(output_root),
        "run_name": args.run_name,
        "qwen_base": base_url,
        "qwen_model": model,
        "metrics": metrics,
        "start_offset": args.start_offset,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "max_inflight": args.max_inflight,
        "frame_count": args.frame_count,
        "max_image_side": args.max_image_side,
        "jpeg_quality": args.jpeg_quality,
        "parallel_frame_decode": args.parallel_frame_decode,
        "model_timeout": args.model_timeout,
        "model_max_tokens": args.model_max_tokens,
        "temperature": args.temperature,
        "disable_thinking": args.disable_thinking,
        "created_at": now_iso(),
    }
    (output_root / f"{args.run_name}_run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_url = normalize_qwen_base(args.qwen_base)
    ensure_no_proxy(base_url)
    model = resolve_model(base_url, args.qwen_model, args.model_timeout)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_config(args, base_url, model, metrics, output_root)

    items = load_eval_items(args)
    if args.start_offset:
        items = items[args.start_offset :]
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        source = args.manifest if args.manifest is not None else args.video_dir
        raise SystemExit(f"No evaluation items found in {source}")

    print("DreamGenBench Qwen API eval", flush=True)
    print(f"Video dir:      {args.video_dir}", flush=True)
    print(f"Manifest:       {args.manifest or '-'}", flush=True)
    print(f"Video count:    {len(items)}", flush=True)
    print(f"Output root:    {output_root}", flush=True)
    print(f"Run name:       {args.run_name}", flush=True)
    print(f"Qwen base:      {base_url}", flush=True)
    print(f"Qwen model:     {model}", flush=True)
    print(f"Max tokens:     {args.model_max_tokens}", flush=True)
    print(f"Metrics:        {', '.join(metrics)}", flush=True)

    frame_cache: dict[Path, list[str]] = {}
    for metric in metrics:
        csv_path = output_root / f"{args.run_name}_{metric}.csv"
        if not args.resume and csv_path.exists():
            csv_path.unlink()
        elif args.rerun_errors:
            prune_error_rows(csv_path)
        done = read_done(csv_path, rerun_errors=args.rerun_errors) if args.resume else set()
        pending = [item for item in items if (item.get("key") or item["video_path"]) not in done]
        print(f"[{metric}] pending {len(pending)}/{len(items)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {}
            pending_iter = iter(pending)

            def submit_next() -> bool:
                try:
                    item = next(pending_iter)
                except StopIteration:
                    return False
                path = Path(item["video_path"])
                if args.parallel_frame_decode:
                    future = pool.submit(
                        evaluate_one_with_frame_decode,
                        metric=metric,
                        item=item,
                        base_url=base_url,
                        model=model,
                        args=args,
                    )
                else:
                    if path not in frame_cache:
                        frame_cache[path] = video_frame_data_urls(path, args)
                    future = pool.submit(
                        evaluate_one,
                        metric=metric,
                        item=item,
                        frame_urls=frame_cache[path],
                        base_url=base_url,
                        model=model,
                        args=args,
                    )
                futures[future] = item
                return True

            target_inflight = args.max_inflight if args.max_inflight > 0 else args.concurrency
            for _ in range(target_inflight):
                if not submit_next():
                    break
            completed = len(items) - len(pending)
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                records = []
                for future in finished:
                    item = futures.pop(future)
                    path = Path(item["video_path"])
                    try:
                        records.append(future.result())
                    except Exception as exc:  # noqa: BLE001
                        records.append({
                            "key": item.get("key", ""),
                            "model": item.get("model", ""),
                            "split": item.get("split", ""),
                            "index": item.get("index", ""),
                            "request_id": item.get("request_id", ""),
                            "video_path": str(path),
                            "prompt": item["prompt"],
                            "prediction": 0,
                            "raw_text": "",
                            "error": str(exc),
                            "model_retry_count": 0,
                            "latency_sec": None,
                            "created_at": now_iso(),
                        })
                    completed += 1
                    if completed % 5 == 0 or completed == len(items):
                        print(f"[{metric}] completed {completed}/{len(items)}", flush=True)
                    submit_next()
                append_csv(csv_path, records)

    summary = {metric: summarize_csv(output_root / f"{args.run_name}_{metric}.csv") for metric in metrics}
    summary["note"] = "This uses an OpenAI-compatible Qwen3.6-VL endpoint, not the official local Qwen2.5-VL-7B judge. PA here is PA-I only unless a PA-II CSV is added separately."
    summary_path = output_root / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
