#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2


DEFAULT_METADATA_JSONL = Path("/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl")
DEFAULT_VIDEO_DIR = Path("/data/datasets/gagi/giga_world_0_outputs/pbench_robot_serving_full_20260612_145541")
DEFAULT_QWEN_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B"

RESULT_JSONL = "qwen_vqa_results.jsonl"
SUMMARY_JSON = "qwen_vqa_summary.json"
RUN_CONFIG_JSON = "run_config.json"

_TLS = threading.local()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PBench Robot generated videos with a Qwen VL judge.")
    parser.add_argument("--metadata-jsonl", type=Path, default=DEFAULT_METADATA_JSONL)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-base", type=str, default=DEFAULT_QWEN_BASE)
    parser.add_argument("--qwen-model", type=str, default="auto")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Number of PBench samples to evaluate. 0 means all.")
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--crop-mode", choices=("right-half", "left-half", "none"), default="right-half")
    parser.add_argument("--model-retries", type=int, default=3)
    parser.add_argument("--model-timeout", type=float, default=300.0)
    parser.add_argument("--model-max-tokens", type=int, default=256)
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
        raise SystemExit(
            "Missing Python package: openai\n"
            "Install it in the conda env with:\n"
            "  source ~/miniconda/etc/profile.d/conda.sh\n"
            "  conda activate giga_models\n"
            "  python -m pip install openai"
        ) from exc

    if not hasattr(_TLS, "openai_clients"):
        _TLS.openai_clients = {}
    if base_url not in _TLS.openai_clients:
        _TLS.openai_clients[base_url] = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    return _TLS.openai_clients[base_url]


def extract_text_from_message(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def parse_json_object_text(text: str) -> dict[str, Any]:
    value = "" if text is None else str(text).strip()
    block = re.search(r"```(?:json)?\s*([\s\S]*?)```", value, flags=re.IGNORECASE)
    if block:
        value = block.group(1).strip()
    if not value.startswith("{"):
        left = value.find("{")
        right = value.rfind("}")
        if left != -1 and right != -1 and right > left:
            value = value[left : right + 1]
    for candidate in (value, re.sub(r",(\s*[}\]])", r"\1", value)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    lower = text.strip().lower()
    if re.search(r"\byes\b", lower) and not re.search(r"\bno\b", lower):
        return {"answer": "yes"}
    if re.search(r"\bno\b", lower) and not re.search(r"\byes\b", lower):
        return {"answer": "no"}
    raise ValueError(f"model output is not a valid JSON object: {text}")


def normalize_yes_no(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return ""


def answer_schema() -> dict[str, Any]:
    return {
        "name": "pbench_yes_no_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }


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
    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    base_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": not disable_thinking}}

    response_formats: list[dict[str, Any] | None] = [
        {"type": "json_schema", "json_schema": answer_schema()},
        {"type": "json_object"},
        None,
    ]
    last_error = ""
    for response_format in response_formats:
        kwargs = dict(base_kwargs)
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if response_format is None:
                break
    if "extra_body" in base_kwargs:
        kwargs = dict(base_kwargs)
        kwargs.pop("extra_body", None)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    raise RuntimeError(last_error)


def crop_frame(frame: Any, crop_mode: str) -> Any:
    if crop_mode == "none":
        return frame
    height, width = frame.shape[:2]
    mid = width // 2
    if crop_mode == "left-half":
        return frame[:, :mid]
    return frame[:, mid:]


def resize_frame(frame: Any, max_side: int) -> Any:
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


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
            frame = crop_frame(frame, args.crop_mode)
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


def build_prompt(question: str) -> str:
    return (
        "You are evaluating a generated robot video for PBench-style physical AI VQA. "
        "The images are uniformly sampled frames from the generated video in chronological order. "
        "Answer the yes/no question using only the visual evidence in these frames. "
        "If the visual evidence contradicts the question, answer no. "
        "Return only JSON in this exact format: {\"answer\":\"yes\"} or {\"answer\":\"no\"}.\n\n"
        f"Question: {question}"
    )


def infer_question(
    *,
    base_url: str,
    model: str,
    frame_urls: list[str],
    sample: dict[str, Any],
    sample_index: int,
    question_index: int,
    qa: dict[str, Any],
    video_path: Path,
    attempt: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    question = str(qa.get("question") or "").strip()
    gold = normalize_yes_no(qa.get("answer"))
    content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(question)}]
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
            message = response.choices[0].message
            raw_text = extract_text_from_message(message.content)
            parsed = parse_json_object_text(raw_text)
            pred = normalize_yes_no(parsed.get("answer"))
            if not pred:
                raise ValueError(f"invalid answer: {parsed.get('answer')!r}")
            return {
                "sample_index": sample_index,
                "pbench_id": sample.get("pbench_id"),
                "question_index": question_index,
                "attempt": attempt,
                "video_path": str(video_path),
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "is_correct": bool(pred and pred == gold),
                "category": qa.get("category"),
                "subcategory": qa.get("subcategory"),
                "raw_text": raw_text,
                "model_retry_count": retry,
                "error": None,
                "latency_sec": round(time.time() - started, 4),
                "created_at": now_iso(),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if retry < args.model_retries:
                time.sleep(min(2.0 * retry, 8.0))
    return {
        "sample_index": sample_index,
        "pbench_id": sample.get("pbench_id"),
        "question_index": question_index,
        "attempt": attempt,
        "video_path": str(video_path),
        "question": question,
        "gold_answer": gold,
        "pred_answer": None,
        "is_correct": False,
        "category": qa.get("category"),
        "subcategory": qa.get("subcategory"),
        "raw_text": raw_text,
        "model_retry_count": args.model_retries,
        "error": f"model failed after {args.model_retries} retries: {last_error}",
        "latency_sec": round(time.time() - started, 4),
        "created_at": now_iso(),
    }


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("qa_pairs"), list):
                raise ValueError(f"{path}:{line_no} qa_pairs is not a list")
            samples.append(row)
    return samples


def candidate_video_paths(video_dir: Path, sample: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    pbench_id = str(sample.get("pbench_id") or "").strip()
    if pbench_id:
        out.append(video_dir / f"{pbench_id}.mp4")
    output_mp4 = str(sample.get("output_mp4") or "").strip()
    if output_mp4:
        out.append(video_dir / output_mp4)
    return out


def find_video(video_dir: Path, sample: dict[str, Any]) -> Path | None:
    for path in candidate_video_paths(video_dir, sample):
        if path.exists():
            return path
    return None


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done(path: Path, rerun_errors: bool) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if rerun_errors and record.get("error"):
                continue
            pbench_id = str(record.get("pbench_id") or "")
            question_index = int(record.get("question_index") or 0)
            attempt = int(record.get("attempt") or 0)
            if pbench_id and question_index and attempt:
                done.add((pbench_id, question_index, attempt))
    return done


def summarize(result_path: Path, output_dir: Path) -> dict[str, Any]:
    total = 0
    correct = 0
    errors = 0
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_subcategory: dict[str, Counter[str]] = defaultdict(Counter)
    by_sample: dict[str, Counter[str]] = defaultdict(Counter)

    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                total += 1
                is_correct = bool(record.get("is_correct"))
                if is_correct:
                    correct += 1
                if record.get("error"):
                    errors += 1
                cat = str(record.get("category") or "unknown")
                sub = f"{cat}/{record.get('subcategory') or 'unknown'}"
                sid = str(record.get("pbench_id") or "unknown")
                by_category[cat]["total"] += 1
                by_category[cat]["correct"] += int(is_correct)
                by_subcategory[sub]["total"] += 1
                by_subcategory[sub]["correct"] += int(is_correct)
                by_sample[sid]["total"] += 1
                by_sample[sid]["correct"] += int(is_correct)

    def table(counter_map: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, counts in sorted(counter_map.items()):
            denom = int(counts["total"])
            num = int(counts["correct"])
            out[key] = {"correct": num, "total": denom, "accuracy": round(num / denom, 6) if denom else None}
        return out

    sample_accuracies = [
        counts["correct"] / counts["total"] for counts in by_sample.values() if int(counts["total"])
    ]
    question_micro_accuracy = correct / total if total else None
    sample_macro_accuracy = sum(sample_accuracies) / len(sample_accuracies) if sample_accuracies else None

    payload = {
        "attempt_count": total,
        "correct_count": correct,
        "error_count": errors,
        "sample_count": len(by_sample),
        "accuracy": round(question_micro_accuracy, 6) if question_micro_accuracy is not None else None,
        "question_micro_accuracy": round(question_micro_accuracy, 6) if question_micro_accuracy is not None else None,
        "sample_macro_accuracy": round(sample_macro_accuracy, 6) if sample_macro_accuracy is not None else None,
        "domain_score_like": round(sample_macro_accuracy * 100, 4) if sample_macro_accuracy is not None else None,
        "by_category": table(by_category),
        "by_subcategory": table(by_subcategory),
        "by_sample": table(by_sample),
        "result_jsonl": str(result_path),
        "created_at": now_iso(),
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_config(args: argparse.Namespace, base_url: str, model: str, output_dir: Path) -> None:
    payload = {
        "metadata_jsonl": str(args.metadata_jsonl),
        "video_dir": str(args.video_dir),
        "output_dir": str(output_dir),
        "qwen_base": base_url,
        "qwen_model": model,
        "start_offset": args.start_offset,
        "limit": args.limit,
        "n_repeats": args.n_repeats,
        "concurrency": args.concurrency,
        "max_inflight": args.max_inflight,
        "frame_count": args.frame_count,
        "max_image_side": args.max_image_side,
        "jpeg_quality": args.jpeg_quality,
        "crop_mode": args.crop_mode,
        "model_retries": args.model_retries,
        "model_timeout": args.model_timeout,
        "model_max_tokens": args.model_max_tokens,
        "temperature": args.temperature,
        "disable_thinking": args.disable_thinking,
        "thinking_enabled": not args.disable_thinking,
        "created_at": now_iso(),
    }
    path = output_dir / RUN_CONFIG_JSON
    if args.resume and path.exists():
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_url = normalize_qwen_base(args.qwen_base)
    ensure_no_proxy(base_url)
    model = resolve_model(base_url, args.qwen_model, args.model_timeout)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / RESULT_JSONL
    result_path.touch(exist_ok=True)
    write_config(args, base_url, model, output_dir)

    samples = load_samples(args.metadata_jsonl.expanduser().resolve())
    selected = samples[args.start_offset :]
    if args.limit > 0:
        selected = selected[: args.limit]

    done = load_done(result_path, args.rerun_errors if args.resume else True)
    max_inflight = args.max_inflight or max(1, args.concurrency * 2)
    submitted = 0
    completed = 0
    skipped = 0
    build_errors: list[dict[str, Any]] = []
    inflight = {}

    print("============================================", flush=True)
    print("PBench Robot Qwen VQA evaluation", flush=True)
    print(f"metadata:       {args.metadata_jsonl}", flush=True)
    print(f"video_dir:      {args.video_dir}", flush=True)
    print(f"output_dir:     {output_dir}", flush=True)
    print(f"qwen_base:      {base_url}", flush=True)
    print(f"qwen_model:     {model}", flush=True)
    print(f"samples:        {len(selected)}", flush=True)
    print(f"existing done:  {len(done)}", flush=True)
    print(f"frame_count:    {args.frame_count}", flush=True)
    print(f"crop_mode:      {args.crop_mode}", flush=True)
    print("============================================", flush=True)

    def drain_one_or_more(block: bool) -> None:
        nonlocal completed
        if not inflight:
            return
        done_futures, _ = wait(inflight, return_when=FIRST_COMPLETED if block else FIRST_COMPLETED, timeout=None if block else 0)
        records = []
        for fut in done_futures:
            inflight.pop(fut)
            records.append(fut.result())
            completed += 1
        append_jsonl(result_path, records)
        if records:
            print(f"completed={completed} submitted={submitted} inflight={len(inflight)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for local_idx, sample in enumerate(selected):
            sample_index = args.start_offset + local_idx
            pbench_id = str(sample.get("pbench_id") or f"sample_{sample_index:06d}")
            video_path = find_video(args.video_dir, sample)
            qa_pairs = list(sample.get("qa_pairs") or [])
            if video_path is None:
                error_record = {
                    "sample_index": sample_index,
                    "pbench_id": pbench_id,
                    "error": "missing generated video",
                    "candidate_video_paths": [str(p) for p in candidate_video_paths(args.video_dir, sample)],
                    "created_at": now_iso(),
                }
                build_errors.append(error_record)
                print(f"missing video for {pbench_id}", flush=True)
                continue
            try:
                frame_urls = video_frame_data_urls(video_path, args)
            except Exception as exc:  # noqa: BLE001
                build_errors.append(
                    {
                        "sample_index": sample_index,
                        "pbench_id": pbench_id,
                        "video_path": str(video_path),
                        "error": f"frame extraction failed: {exc}",
                        "created_at": now_iso(),
                    }
                )
                print(f"frame extraction failed for {pbench_id}: {exc}", flush=True)
                continue

            for question_index, qa in enumerate(qa_pairs, start=1):
                for attempt in range(1, args.n_repeats + 1):
                    key = (pbench_id, question_index, attempt)
                    if key in done:
                        skipped += 1
                        continue
                    while len(inflight) >= max_inflight:
                        drain_one_or_more(block=True)
                    inflight[
                        executor.submit(
                            infer_question,
                            base_url=base_url,
                            model=model,
                            frame_urls=frame_urls,
                            sample=sample,
                            sample_index=sample_index,
                            question_index=question_index,
                            qa=qa,
                            video_path=video_path,
                            attempt=attempt,
                            args=args,
                        )
                    ] = key
                    submitted += 1
            drain_one_or_more(block=False)

        while inflight:
            drain_one_or_more(block=True)

    if build_errors:
        append_jsonl(output_dir / "qwen_vqa_build_errors.jsonl", build_errors)

    summary = summarize(result_path, output_dir)
    summary.update({"submitted": submitted, "completed": completed, "skipped_done": skipped, "build_error_count": len(build_errors)})
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
