#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import re
import ssl
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
from google import genai
from google.genai import types as gt


DEFAULT_VIDEO_DIR = Path("/data/datasets/gagi/gr1_dreamgen_eval/dreamgenbench_video_dirs/gr1_dreamgen_8gpu_full_20260625_212933")
DEFAULT_OUTPUT_ROOT = Path("/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs")

DIFROST_API_TOKEN = os.getenv(
    "DIFROST_API_TOKEN",
    None,
)
DIFROST_GENAI_BASE_URL = os.getenv("DIFROST_GENAI_BASE_URL", "https://api-gateway.example.com/v1")
DIFROST_HOST = os.getenv("DIFROST_HOST", "api-gateway.example.com")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DreamGenBench GPT-IF with Difrost GenAI.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=f"dreamgen_gpt_if_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--base-url", type=str, default=DIFROST_GENAI_BASE_URL)
    parser.add_argument("--host", type=str, default=DIFROST_HOST)
    parser.add_argument("--api-token", type=str, default=DIFROST_API_TOKEN)
    parser.add_argument("--model", type=str, default=os.environ.get("DIFROST_MODEL", "gpt-5.5"))
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--scale-factor", type=float, default=0.5)
    parser.add_argument("--model-retries", type=int, default=6)
    parser.add_argument("--model-max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking-level", type=str, default=os.getenv("DIFROST_THINKING_LEVEL", "low"), choices=["low", "medium", "high"])
    parser.add_argument("--include-thoughts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-errors", action="store_true")
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    value = str(base_url or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    host_port = value if ":" in value else f"{value}:8000"
    return f"http://{host_port}/api/v1"


def ensure_no_proxy(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        parts = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        if host not in parts:
            parts.append(host)
        os.environ[key] = ",".join(parts)


def make_client(base_url: str, api_token: str, host: str) -> genai.Client:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    http_opts = gt.HttpOptions(
        base_url=base_url,
        api_version="genai",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Host": host,
            "X-Difrost-Session-Affinity": uuid.uuid4().hex,
        },
        async_client_args={"ssl": ssl_ctx},
        client_args={"verify": False},
    )
    return genai.Client(vertexai=False, api_key="placeholder", http_options=http_opts)


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
    vid_path = str(path)
    if "cogvideo" in vid_path:
        if "zeroshot" in vid_path:
            prompt = vid_path.split("/", 8)[-1].split("_", 1)[1].replace(".mp4", "")
        else:
            prompt = vid_path.split("/")[-3].split("_", 1)[1]
    elif "hunyuan" in vid_path:
        prompt = path.name.split("_", 1)[1]
        prompt = prompt.split("_", 4)[-1].replace(".mp4", "")
    elif "wan" in vid_path or "cosmos" in vid_path:
        prompt = path.stem.split("_", 1)[1]
    else:
        stem = path.stem
        prompt = stem.split("_", 1)[1] if "_" in stem else stem
    return prompt.replace("_", " ")


def frame_indices(total_frames: int, frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if total_frames <= 0:
        return list(range(frame_count))
    if frame_count == 1:
        return [total_frames // 2]
    return sorted({round(i * (total_frames - 1) / (frame_count - 1)) for i in range(frame_count)})


def encode_frame_png(frame_bgr: Any, scale_factor: float) -> bytes:
    if scale_factor > 0 and scale_factor != 1.0:
        height, width = frame_bgr.shape[:2]
        frame_bgr = cv2.resize(
            frame_bgr,
            (max(1, round(width * scale_factor)), max(1, round(height * scale_factor))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise ValueError("failed to encode frame as png")
    return encoded.tobytes()


def video_frame_pngs(video_path: Path, args: argparse.Namespace) -> list[bytes]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[bytes] = []
        for idx in frame_indices(total, args.frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(encode_frame_png(frame, args.scale_factor))
        if not frames:
            raise ValueError(f"no frames extracted from video: {video_path}")
        return frames
    finally:
        cap.release()


def gpt_if_prompt(prompt: str) -> str:
    return (
        "The video shows a robot arm completing a specific task. "
        f"Please evaluate: if the video follows the instruction to finish the task '{prompt}', give a positive score. "
        "Reply only '0' for No or '1' for Yes."
    )


def build_contents(prompt: str, frame_pngs: list[bytes]) -> list[gt.Content]:
    parts = [
        gt.Part.from_uri(
            file_uri=f"data:image/png;base64,{base64.b64encode(frame).decode('ascii')}",
            mime_type="image/png",
        )
        for frame in frame_pngs
    ]
    parts.append(gt.Part.from_text(text=gpt_if_prompt(prompt)))
    return [gt.Content(role="user", parts=parts)]


def response_text_parts(response: Any) -> tuple[list[str], list[str]]:
    candidate_texts: list[str] = []
    thought_texts: list[str] = []
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                thought_texts.append(text)
            else:
                candidate_texts.append(text)
    return candidate_texts, thought_texts


async def evaluate_one(
    *,
    client: genai.Client,
    sem: asyncio.Semaphore,
    video_path: Path,
    frame_pngs: list[bytes],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    prompt = prompt_from_video_path(video_path)
    contents = build_contents(prompt, frame_pngs)
    config = gt.GenerateContentConfig(
        max_output_tokens=args.model_max_tokens,
        temperature=args.temperature,
        thinking_config=gt.ThinkingConfig(
            thinking_level=args.thinking_level,
            include_thoughts=args.include_thoughts,
        ),
    )
    raw_text = ""
    thinking_text = ""
    last_error = ""
    resp = None
    async with sem:
        for retry in range(1, args.model_retries + 1):
            try:
                resp = await client.aio.models.generate_content(model=args.model, contents=contents, config=config)
                candidate_texts, thought_texts = response_text_parts(resp)
                raw_text = "\n\n".join(candidate_texts) or getattr(resp, "text", "") or ""
                thinking_text = "\n\n".join(thought_texts)
                usage = getattr(resp, "usage_metadata", None)
                return {
                    "video_path": str(video_path),
                    "prompt": prompt,
                    "prediction": parse_prediction(raw_text),
                    "raw_text": raw_text,
                    "thinking_text": thinking_text,
                    "error": None,
                    "model_retry_count": retry,
                    "latency_sec": round(time.time() - started, 4),
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                    "candidates_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
                    "thoughts_tokens": getattr(usage, "thoughts_token_count", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_token_count", 0) if usage else 0,
                    "created_at": now_iso(),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if retry < args.model_retries:
                    await asyncio.sleep(min(5.0 * retry, 30.0))
    return {
        "video_path": str(video_path),
        "prompt": prompt,
        "prediction": 0,
        "raw_text": raw_text,
        "thinking_text": thinking_text,
        "error": f"model failed after {args.model_retries} retries: {last_error}",
        "model_retry_count": args.model_retries,
        "latency_sec": round(time.time() - started, 4),
        "prompt_tokens": 0,
        "candidates_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0,
        "created_at": now_iso(),
    }


def read_done(path: Path, rerun_errors: bool) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if rerun_errors and row.get("error"):
                continue
            video_path = row.get("video_path")
            if video_path:
                done.add(video_path)
    return done


def append_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = [
        "video_path",
        "prompt",
        "prediction",
        "raw_text",
        "thinking_text",
        "error",
        "model_retry_count",
        "latency_sec",
        "prompt_tokens",
        "candidates_tokens",
        "thoughts_tokens",
        "total_tokens",
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


def write_config(args: argparse.Namespace, base_url: str, output_root: Path) -> None:
    payload = {
        "video_dir": str(args.video_dir),
        "output_root": str(output_root),
        "run_name": args.run_name,
        "base_url": base_url,
        "host": args.host,
        "model": args.model,
        "start_offset": args.start_offset,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "frame_count": args.frame_count,
        "scale_factor": args.scale_factor,
        "model_max_tokens": args.model_max_tokens,
        "temperature": args.temperature,
        "thinking_level": args.thinking_level,
        "include_thoughts": args.include_thoughts,
        "created_at": now_iso(),
    }
    (output_root / f"{args.run_name}_gpt_if_run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_batch(args: argparse.Namespace) -> None:
    base_url = normalize_base_url(args.base_url)
    ensure_no_proxy(base_url)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_config(args, base_url, output_root)

    videos = sorted(args.video_dir.expanduser().resolve().glob("**/*.mp4"))
    if args.start_offset:
        videos = videos[args.start_offset :]
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No mp4 files found in {args.video_dir}")

    csv_path = output_root / f"{args.run_name}_gpt_if.csv"
    done = read_done(csv_path, rerun_errors=args.rerun_errors) if args.resume else set()
    pending = [path for path in videos if str(path) not in done]

    print("DreamGenBench GPT-IF Difrost GenAI eval", flush=True)
    print(f"Video dir:      {args.video_dir}", flush=True)
    print(f"Video count:    {len(videos)}", flush=True)
    print(f"Pending:        {len(pending)}", flush=True)
    print(f"Output root:    {output_root}", flush=True)
    print(f"Run name:       {args.run_name}", flush=True)
    print(f"Base URL:       {base_url}", flush=True)
    print(f"Host:           {args.host}", flush=True)
    print(f"Model:          {args.model}", flush=True)
    print(f"Concurrency:    {args.concurrency}", flush=True)

    client = make_client(base_url, args.api_token, args.host)
    sem = asyncio.Semaphore(args.concurrency)
    completed = len(done)
    records_buffer: list[dict[str, Any]] = []

    async def process(path: Path) -> None:
        nonlocal completed, records_buffer
        try:
            frame_pngs = await asyncio.to_thread(video_frame_pngs, path, args)
            rec = await evaluate_one(client=client, sem=sem, video_path=path, frame_pngs=frame_pngs, args=args)
        except Exception as exc:  # noqa: BLE001
            rec = {
                "video_path": str(path),
                "prompt": prompt_from_video_path(path),
                "prediction": 0,
                "raw_text": "",
                "thinking_text": "",
                "error": str(exc),
                "model_retry_count": 0,
                "latency_sec": None,
                "prompt_tokens": 0,
                "candidates_tokens": 0,
                "thoughts_tokens": 0,
                "total_tokens": 0,
                "created_at": now_iso(),
            }
        records_buffer.append(rec)
        completed += 1
        if len(records_buffer) >= 5 or completed == len(videos):
            append_csv(csv_path, records_buffer)
            records_buffer = []
        if completed % 5 == 0 or completed == len(videos):
            print(f"[gpt_if] completed {completed}/{len(videos)}", flush=True)

    await asyncio.gather(*(process(path) for path in pending))
    if records_buffer:
        append_csv(csv_path, records_buffer)

    summary = {"gpt_if": summarize_csv(csv_path)}
    summary_path = output_root / f"{args.run_name}_gpt_if_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    asyncio.run(run_batch(parse_args()))


if __name__ == "__main__":
    main()
