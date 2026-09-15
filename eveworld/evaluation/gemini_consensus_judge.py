#!/usr/bin/env python3
"""Evaluate consensus v2 Lance questions with Gemini and Qwen LLM judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import ssl
import threading
import time
import urllib.request
import uuid
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import lance
import pyarrow as pa
from google import genai
from google.genai import types as gt
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


DEFAULT_SOURCE_LANCE = Path("/data/datasets/er_gemini_gpt/consensus_v2.lance")
DEFAULT_OUTPUT_DIR = Path("/data/datasets/er_gemini_gpt/gemini35_high4_eval_consensus_v2_qwenjudge")

DIFROST_API_TOKEN = os.getenv(
    "DIFROST_API_TOKEN",
    None,
)
DIFROST_GENAI_BASE_URL = os.getenv("DIFROST_GENAI_BASE_URL", "https://llm-api.cc/api/v1")
DIFROST_HOST = os.getenv("DIFROST_HOST", "llm-api.cc")

DEFAULT_MODEL = os.getenv("DIFROST_MODEL", "gemini-3.5-flash")
DEFAULT_N_REPEATS = 4
DEFAULT_CONCURRENCY = 300
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_RETRIES = 6
DEFAULT_MODEL_TIMEOUT = 1200.0
DEFAULT_MODEL_MAX_TOKENS = 32000
DEFAULT_THINKING_LEVEL = "high"
DEFAULT_INCLUDE_THOUGHTS = True
DEFAULT_TEMPERATURE = 0.5

DEFAULT_JUDGE_BASE = "http://10.60.32.16:8000/v1"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_JUDGE_TIMEOUT = 1200.0
DEFAULT_JUDGE_RETRIES = 2
DEFAULT_JUDGE_MAX_TOKENS = 12000

RESULT_JSONL = "gemini35_high4_consensus_v2_qwenjudge_attempts.jsonl"
BUILD_ERRORS_JSONL = "gemini35_high4_consensus_v2_qwenjudge_build_errors.jsonl"
QUESTION_SUMMARY_JSONL = "gemini35_high4_consensus_v2_qwenjudge_question_summary.jsonl"
QUESTION_SUMMARY_LANCE = "gemini35_high4_consensus_v2_qwenjudge_question_summary_lance"
SUMMARY_JSON = "gemini35_high4_consensus_v2_qwenjudge_summary.json"
RUN_CONFIG_JSON = "run_config.json"

VALID_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SOURCE_COLUMNS = [
    "question",
    "options",
    "answer",
    "question_images",
    "source_kind",
    "uuid",
    "source_erqa_id",
    "source_status",
    "source_question_type",
    "source_image_set",
    "candidate_key",
    "candidate_row_index",
    "candidate_image_mime_types",
    "candidate_image_count",
    "candidate_image_paths",
    "candidate_question",
    "candidate_options",
    "candidate_answer",
    "v2_raw_reason",
    "audit_run_count",
    "audit_clear_count",
    "audit_ambiguous_count",
    "audit_error_count",
    "audit_answer_agree_count",
    "audit_majority_answer",
    "audit_answer_counts_json",
]
RESUME_CONFIG_KEYS = (
    "source_lance",
    "row_count",
    "start_offset",
    "limit",
    "gemini_model",
    "gemini_thinking_level",
    "gemini_include_thoughts",
    "gemini_temperature",
    "judge_base",
    "judge_model",
)

GEMINI_SYSTEM_PROMPT = (
    "You are a careful embodied robot-vision multiple-choice QA assistant. "
    "Use only the provided image or image sequence and choose exactly one option. "
    "Make your final selected option letter clear."
)

JUDGE_SYSTEM_PROMPT = """You judge multiple-choice visual QA answers.

You are given the question, parsed options, the correct answer, and a model's
answer content. Do not use images. Decide whether the model answer content
ultimately selects a correct option. Return only JSON matching the schema.
"""

QUESTION_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("record_key", pa.string()),
        pa.field("uuid", pa.string()),
        pa.field("source_index", pa.int64()),
        pa.field("source_kind", pa.string()),
        pa.field("source_erqa_id", pa.string()),
        pa.field("source_status", pa.string()),
        pa.field("source_question_type", pa.string()),
        pa.field("source_image_set", pa.string()),
        pa.field("v2_raw_reason", pa.string()),
        pa.field("audit_clear_count", pa.int64()),
        pa.field("image_count", pa.int64()),
        pa.field("question", pa.string()),
        pa.field("option_labels", pa.list_(pa.string())),
        pa.field("option_texts", pa.list_(pa.string())),
        pa.field("gold_answer", pa.string()),
        pa.field("correct_count", pa.int64()),
        pa.field("attempt_count", pa.int64()),
        pa.field("model_error_count", pa.int64()),
        pa.field("judge_error_count", pa.int64()),
        pa.field("pred_choice_labels", pa.list_(pa.string())),
        pa.field(
            "attempt_statuses",
            pa.list_(
                pa.struct(
                    [
                        pa.field("attempt", pa.int64()),
                        pa.field("pred_choice_label", pa.string()),
                        pa.field("direct_is_correct", pa.bool_()),
                        pa.field("judge_is_correct", pa.bool_()),
                        pa.field("is_correct", pa.bool_()),
                        pa.field("model_error", pa.string()),
                        pa.field("judge_error", pa.string()),
                    ]
                )
            ),
        ),
    ]
)

_TLS = threading.local()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if block:
        text = block.group(1).strip()
    try:
        value = json.loads(text)
    except Exception:
        left = text.find("{")
        right = text.rfind("}")
        if left < 0 or right <= left:
            return None
        try:
            value = json.loads(text[left : right + 1])
        except Exception:
            return None
    return value if isinstance(value, dict) else None


def parse_options_text(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?:^|\s)([A-Z])[\.\)]\s*", str(text or "")))
    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = re.sub(r"^[\s:;\-]+", "", text[start:end].strip())
        if label and value:
            options[label] = value
    return options


def parse_options(value: Any) -> tuple[list[str], list[str], dict[str, str], str]:
    obj = parse_json_object(value)
    if obj is None:
        obj = parse_options_text(str(value or ""))
    cleaned: dict[str, str] = {}
    for raw_label, raw_text in obj.items():
        label = str(raw_label or "").strip().upper()
        if len(label) != 1 or label not in VALID_LETTERS:
            continue
        text = str(raw_text or "").strip()
        if text:
            cleaned[label] = text
    labels = [label for label in VALID_LETTERS if label in cleaned]
    labels.extend(sorted(label for label in cleaned if label not in labels))
    options = {label: cleaned[label] for label in labels}
    return labels, [options[label] for label in labels], options, json.dumps(options, ensure_ascii=False, sort_keys=True)


def normalize_answer(value: Any, option_labels: list[str]) -> str:
    allowed = {str(label).strip().upper() for label in option_labels if str(label).strip()}
    text = str(value or "").strip().upper()
    if not text:
        return ""
    boxed = re.search(r"\\BOXED\{\s*([A-Z])\s*\}", text)
    if boxed and boxed.group(1) in allowed:
        return boxed.group(1)
    if text in allowed:
        return text
    answer = re.search(r'["\']?(?:answer|choice_label|normalized_answer)["\']?\s*[:：]\s*["\']?([A-Z])["\']?', text)
    if answer and answer.group(1) in allowed:
        return answer.group(1)
    option = re.search(r"\bOPTION\s*([A-Z])\b", text)
    if option and option.group(1) in allowed:
        return option.group(1)
    leading = re.match(r"^\s*([A-Z])(?:[\.\):,\s]|$)", text)
    if leading and leading.group(1) in allowed:
        return leading.group(1)
    bare = re.search(r"\b([A-Z])\b", text)
    if bare and bare.group(1) in allowed:
        return bare.group(1)
    return ""


def extract_pred_choice_label(value: Any, option_labels: list[str]) -> str:
    allowed = {str(label).strip().upper() for label in option_labels if str(label).strip()}
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = parse_json_object(text)
    if parsed:
        for key in ("normalized_answer", "choice_label", "answer"):
            label = normalize_answer(parsed.get(key), option_labels)
            if label:
                return label

    upper = text.upper()
    patterns = (
        r"(?:FINAL|CORRECT|SELECTED)\s+(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[^A-Z0-9]{0,20}([A-Z])\b",
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)\s*[^A-Z0-9]{0,20}([A-Z])\b",
        r"\bOPTION\s*([A-Z])\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, upper):
            label = match.group(1)
            if label in allowed:
                return label

    tail = upper[-500:]
    for match in reversed(list(re.finditer(r"(?<![A-Z])([A-Z])(?![A-Z])", tail))):
        label = match.group(1)
        if label in allowed:
            return label
    return ""


def guess_mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def verify_image_bytes(image_bytes: bytes) -> None:
    Image.open(BytesIO(image_bytes)).verify()


def extract_image_bytes(image_obj: Any) -> bytes | None:
    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)
    if isinstance(image_obj, dict):
        data = image_obj.get("bytes")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    return None


def normalize_qwen_base(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty judge base")
    if not text.startswith(("http://", "https://")):
        text = f"http://{text}"
    text = text.rstrip("/")
    return text if text.endswith("/v1") else f"{text}/v1"


def ensure_no_proxy_for_base_url(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if not host:
        return
    for env_key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(env_key, "")
        parts = [item.strip() for item in current.split(",") if item.strip()]
        if host not in set(parts):
            parts.append(host)
        os.environ[env_key] = ",".join(parts)


def resolve_openai_model(base_url: str, requested: str) -> str:
    model = str(requested or "").strip()
    if model and model.lower() != "auto":
        return model
    models_url = f"{base_url.rstrip('/')}/models"
    with urllib.request.urlopen(models_url, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError(f"cannot auto-resolve judge model from {models_url}: {payload!r}")
    return str(data[0]["id"])


def make_gemini_client(timeout_seconds: float) -> genai.Client:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout_ms = int(timeout_seconds * 1000) if timeout_seconds and timeout_seconds > 0 else None
    http_opts = gt.HttpOptions(
        base_url=DIFROST_GENAI_BASE_URL,
        api_version="genai",
        timeout=timeout_ms,
        headers={
            "Authorization": f"Bearer {DIFROST_API_TOKEN}",
            "Host": DIFROST_HOST,
            "X-Difrost-Session-Affinity": uuid.uuid4().hex,
        },
        async_client_args={"ssl": ssl_ctx},
        client_args={"verify": False},
    )
    return genai.Client(vertexai=False, api_key="placeholder", http_options=http_opts)


def get_gemini_client(timeout_seconds: float) -> genai.Client:
    cached = getattr(_TLS, "gemini_client_cache", None)
    key = float(timeout_seconds or 0)
    if not cached or cached[0] != key:
        cached = (key, make_gemini_client(timeout_seconds))
        _TLS.gemini_client_cache = cached
    return cached[1]


def get_judge_client(base_url: str) -> OpenAI:
    clients = getattr(_TLS, "judge_clients", None)
    if clients is None:
        clients = {}
        _TLS.judge_clients = clients
    if base_url not in clients:
        clients[base_url] = OpenAI(base_url=base_url, api_key="EMPTY")
    return clients[base_url]


def response_text_parts(response: Any) -> tuple[list[str], list[str]]:
    candidate_texts: list[str] = []
    thought_texts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                thought_texts.append(text)
            else:
                candidate_texts.append(text)
    return candidate_texts, thought_texts


def extract_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if hasattr(content, "text"):
        text = getattr(content, "text", None)
        if isinstance(text, str):
            return text.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("value")
            elif hasattr(item, "text"):
                text = getattr(item, "text", None)
            else:
                text = item
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def judge_schema() -> dict[str, Any]:
    return {
        "name": "consensus_v2_answer_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"is_correct": {"type": "boolean"}},
            "required": ["is_correct"],
            "additionalProperties": False,
        },
    }


def parse_judge_bool(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        raise ValueError("empty judge response")
    block = re.search(r"```(?:json)?\s*([\s\S]*?)```", value, flags=re.IGNORECASE)
    if block:
        value = block.group(1).strip()
    if not value.startswith("{"):
        left = value.find("{")
        right = value.rfind("}")
        if left != -1 and right != -1 and right > left:
            value = value[left : right + 1]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict) and isinstance(parsed.get("is_correct"), bool):
            return bool(parsed["is_correct"])
    except Exception:
        pass
    lowered = value.lower()
    if lowered in {"true", "yes", "correct"}:
        return True
    if lowered in {"false", "no", "incorrect", "wrong"}:
        return False
    raise ValueError(f"cannot parse judge boolean: {text!r}")


def build_judge_prompt(item: dict[str, Any], content: str) -> str:
    correct_text = item["options"].get(item["gold_answer"], "")
    return f"""Question:
{item["question"]}

Parsed options:
{json.dumps(item["options"], ensure_ascii=False, sort_keys=True)}

Correct answer:
{item["gold_answer"]}. {correct_text}

Model answer content:
{content}

Is the model answer content ultimately correct? Return true if it clearly selects the correct option or an equivalent final answer. Otherwise return false.
""".strip()


def create_judge_completion(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> Any:
    last_error: Exception | None = None
    for response_format in (
        {"type": "json_schema", "json_schema": judge_schema()},
        {"type": "json_object"},
        None,
    ):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            if response_format is None:
                raise
    raise RuntimeError(f"judge completion failed: {last_error}")


def judge_content(item: dict[str, Any], content: str, args: argparse.Namespace, judge_base: str) -> dict[str, Any]:
    if not str(content or "").strip():
        return {
            "judge_is_correct": False,
            "judge_raw_text": "",
            "judge_retry_count": 0,
            "judge_error": "empty gemini content",
        }

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(item, content)},
    ]
    last_error = ""
    last_raw = ""
    for retry_idx in range(1, max(int(args.judge_retries), 1) + 1):
        try:
            response = create_judge_completion(
                client=get_judge_client(judge_base),
                model=args.judge_model,
                messages=messages,
                temperature=0.0,
                max_tokens=args.judge_max_tokens,
                timeout=args.judge_timeout,
            )
            raw = extract_message_text(response.choices[0].message.content)
            return {
                "judge_is_correct": parse_judge_bool(raw),
                "judge_raw_text": raw,
                "judge_retry_count": retry_idx,
                "judge_error": "",
            }
        except Exception as exc:
            last_error = str(exc)
            last_raw = last_raw or ""
            if retry_idx < int(args.judge_retries):
                time.sleep(min(2.0 * retry_idx, 8.0))
    return {
        "judge_is_correct": False,
        "judge_raw_text": last_raw,
        "judge_retry_count": int(args.judge_retries),
        "judge_error": f"judge failed after {args.judge_retries} retries: {last_error}",
    }


def stable_uid(row: dict[str, Any], source_index: int) -> str:
    row_uuid = str(row.get("uuid") or "").strip()
    return row_uuid or f"row_{source_index}"


def record_key_for(row: dict[str, Any], source_index: int) -> str:
    source_kind = str(row.get("source_kind") or "unknown").strip() or "unknown"
    return f"{source_kind}:{stable_uid(row, source_index)}"


def build_images(row: dict[str, Any], args: argparse.Namespace) -> list[tuple[bytes, str]]:
    raw_values = normalize_list(row.get("question_images"))
    mimes = [str(x or "").strip() for x in normalize_list(row.get("candidate_image_mime_types"))]
    images: list[tuple[bytes, str]] = []
    for idx, image_obj in enumerate(raw_values):
        data = extract_image_bytes(image_obj)
        if not data:
            raise ValueError(f"image {idx} is empty or not bytes")
        if args.verify_images:
            verify_image_bytes(data)
        mime = mimes[idx] if idx < len(mimes) and mimes[idx] else guess_mime(data)
        images.append((data, mime))
    return images


def build_item(row: dict[str, Any], source_index: int, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    uid = stable_uid(row, source_index)
    source_kind = str(row.get("source_kind") or "").strip()
    record_key = f"{source_kind or 'unknown'}:{uid}"
    labels, texts, options, options_json = parse_options(row.get("options") or row.get("candidate_options"))
    gold = normalize_answer(row.get("answer") or row.get("candidate_answer"), labels)
    question = str(row.get("question") or row.get("candidate_question") or "").strip()

    base_error = {
        "record_key": record_key,
        "uuid": uid,
        "source_index": int(source_index),
        "source_kind": source_kind,
        "source_erqa_id": str(row.get("source_erqa_id") or ""),
        "created_at": now_iso(),
    }
    if not question or len(labels) < 2 or not gold:
        return None, {
            **base_error,
            "error": "invalid question/options/answer fields",
            "question": question,
            "answer": row.get("answer"),
            "options": options_json,
            "option_labels": labels,
        }
    try:
        images = build_images(row, args)
    except Exception as exc:
        return None, {
            **base_error,
            "error": f"image build failed: {exc}",
            "image_paths": row.get("candidate_image_paths"),
        }
    if not images:
        return None, {**base_error, "error": "no image bytes"}

    return {
        "record_key": record_key,
        "uuid": uid,
        "source_index": int(source_index),
        "source_kind": source_kind,
        "source_erqa_id": str(row.get("source_erqa_id") or ""),
        "source_status": str(row.get("source_status") or ""),
        "source_question_type": str(row.get("source_question_type") or ""),
        "source_image_set": str(row.get("source_image_set") or ""),
        "candidate_key": str(row.get("candidate_key") or ""),
        "candidate_row_index": row.get("candidate_row_index"),
        "v2_raw_reason": str(row.get("v2_raw_reason") or ""),
        "audit_run_count": int(row.get("audit_run_count") or 0),
        "audit_clear_count": int(row.get("audit_clear_count") or 0),
        "audit_ambiguous_count": int(row.get("audit_ambiguous_count") or 0),
        "audit_error_count": int(row.get("audit_error_count") or 0),
        "audit_answer_agree_count": int(row.get("audit_answer_agree_count") or 0),
        "audit_majority_answer": str(row.get("audit_majority_answer") or ""),
        "audit_answer_counts_json": str(row.get("audit_answer_counts_json") or ""),
        "question": question,
        "option_labels": labels,
        "option_texts": texts,
        "options": options,
        "options_json": options_json,
        "gold_answer": gold,
        "gold_answer_text": options.get(gold, ""),
        "image_count": len(images),
        "images": images,
    }, None


def build_user_prompt(item: dict[str, Any]) -> str:
    options = [f"{label}. {text}" for label, text in zip(item["option_labels"], item["option_texts"], strict=True)]
    return (
        "Answer the embodied visual multiple-choice question using the provided image sequence. "
        "If there are multiple images, treat them as ordered displayed frames unless the question says otherwise.\n\n"
        f"Question:\n{item['question']}\n\n"
        "Options:\n"
        + "\n".join(options)
        + "\n\n"
        "Answer naturally, but make the final selected option letter clear."
    )


async def generate_content_async(
    client: genai.Client,
    *,
    model: str,
    contents: list[gt.Content],
    config: gt.GenerateContentConfig,
) -> Any:
    return await client.aio.models.generate_content(model=model, contents=contents, config=config)


def generate_content_sync(
    client: genai.Client,
    *,
    model: str,
    contents: list[gt.Content],
    config: gt.GenerateContentConfig,
) -> Any:
    return asyncio.run(generate_content_async(client, model=model, contents=contents, config=config))


def infer_gemini_once(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    parts: list[gt.Part] = []
    for data, mime in item["images"]:
        parts.append(gt.Part.from_bytes(data=data, mime_type=mime or guess_mime(data)))
    parts.append(gt.Part.from_text(text=build_user_prompt(item)))
    contents = [gt.Content(role="user", parts=parts)]
    config = gt.GenerateContentConfig(
        system_instruction=GEMINI_SYSTEM_PROMPT,
        max_output_tokens=args.model_max_tokens,
        temperature=args.temperature,
        thinking_config=gt.ThinkingConfig(
            thinking_level=args.thinking_level,
            include_thoughts=args.include_thoughts,
        ),
    )

    last_error = ""
    last_raw = ""
    last_reasoning = ""
    resp = None
    for retry_idx in range(1, max(int(args.model_retries), 1) + 1):
        raw_text = ""
        raw_reasoning = ""
        try:
            client = get_gemini_client(args.model_timeout)
            resp = generate_content_sync(client, model=args.model, contents=contents, config=config)
            candidate_texts, thought_texts = response_text_parts(resp)
            raw_text = "\n\n".join(candidate_texts) or getattr(resp, "text", "") or ""
            raw_reasoning = "\n\n".join(thought_texts)
            usage = getattr(resp, "usage_metadata", None)
            return {
                "raw_text": raw_text,
                "raw_reasoning": raw_reasoning,
                "model_retry_count": retry_idx,
                "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "candidates_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
                "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
                "model_error": "",
            }
        except Exception as exc:
            last_error = str(exc)
            last_raw = raw_text or last_raw
            last_reasoning = raw_reasoning or last_reasoning
            if retry_idx < int(args.model_retries):
                time.sleep(min(5.0 * retry_idx, 30.0))

    usage = getattr(resp, "usage_metadata", None) if resp is not None else None
    return {
        "raw_text": last_raw,
        "raw_reasoning": last_reasoning,
        "model_retry_count": int(args.model_retries),
        "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "candidates_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        "model_error": f"model failed after {args.model_retries} retries: {last_error}",
    }


def run_attempt(item: dict[str, Any], attempt: int, args: argparse.Namespace, judge_base: str) -> dict[str, Any]:
    started = time.time()
    gemini = infer_gemini_once(item, args)
    raw_text = str(gemini.get("raw_text") or "")
    pred_choice_label = extract_pred_choice_label(raw_text, item["option_labels"])
    direct_is_correct = bool(pred_choice_label and pred_choice_label == item["gold_answer"])

    if gemini.get("model_error"):
        judge = {
            "judge_is_correct": False,
            "judge_raw_text": "",
            "judge_retry_count": 0,
            "judge_error": "skipped because gemini failed",
        }
    else:
        judge = judge_content(item, raw_text, args, judge_base)

    model_error = str(gemini.get("model_error") or "")
    judge_error = str(judge.get("judge_error") or "")
    is_correct = bool(judge.get("judge_is_correct")) and not model_error and not judge_error
    return {
        "run_id": args.run_id,
        "record_key": item["record_key"],
        "uuid": item["uuid"],
        "source_index": item["source_index"],
        "attempt": int(attempt),
        "source_kind": item["source_kind"],
        "source_erqa_id": item["source_erqa_id"],
        "source_status": item["source_status"],
        "source_question_type": item["source_question_type"],
        "source_image_set": item["source_image_set"],
        "candidate_key": item.get("candidate_key"),
        "candidate_row_index": item.get("candidate_row_index"),
        "v2_raw_reason": item["v2_raw_reason"],
        "audit_run_count": item["audit_run_count"],
        "audit_clear_count": item["audit_clear_count"],
        "audit_ambiguous_count": item["audit_ambiguous_count"],
        "audit_error_count": item["audit_error_count"],
        "audit_answer_agree_count": item["audit_answer_agree_count"],
        "audit_majority_answer": item["audit_majority_answer"],
        "audit_answer_counts_json": item["audit_answer_counts_json"],
        "image_count": item["image_count"],
        "question": item["question"],
        "option_labels": item["option_labels"],
        "option_texts": item["option_texts"],
        "options": item["options"],
        "gold_answer": item["gold_answer"],
        "gold_answer_text": item["gold_answer_text"],
        "gemini_model": args.model,
        "gemini_thinking_level": args.thinking_level,
        "gemini_include_thoughts": bool(args.include_thoughts),
        "gemini_temperature": float(args.temperature),
        "raw_text": raw_text,
        "raw_reasoning": gemini.get("raw_reasoning") or "",
        "pred_choice_label": pred_choice_label or "",
        "direct_is_correct": direct_is_correct,
        "judge_model": args.judge_model,
        "judge_base": judge_base,
        "judge_is_correct": bool(judge.get("judge_is_correct")),
        "judge_raw_text": judge.get("judge_raw_text") or "",
        "judge_retry_count": int(judge.get("judge_retry_count") or 0),
        "is_correct": is_correct,
        "model_error": model_error,
        "judge_error": judge_error,
        "error": model_error or judge_error,
        "model_retry_count": int(gemini.get("model_retry_count") or 0),
        "prompt_tokens": int(gemini.get("prompt_tokens") or 0),
        "candidates_tokens": int(gemini.get("candidates_tokens") or 0),
        "thoughts_tokens": int(gemini.get("thoughts_tokens") or 0),
        "total_tokens": int(gemini.get("total_tokens") or 0),
        "latency_sec": round(time.time() - started, 4),
        "created_at": now_iso(),
    }


def run_config(args: argparse.Namespace, ds: lance.LanceDataset, judge_base: str) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "source_lance": str(args.source_lance.resolve()),
        "schema": str(ds.schema),
        "row_count": ds.count_rows(),
        "start_offset": args.start_offset,
        "limit": args.limit,
        "n_repeats": args.n_repeats,
        "concurrency": args.concurrency,
        "max_inflight": args.max_inflight,
        "batch_size": args.batch_size,
        "gemini_model": args.model,
        "gemini_model_timeout": args.model_timeout,
        "gemini_model_max_tokens": args.model_max_tokens,
        "gemini_model_retries": args.model_retries,
        "gemini_thinking_level": args.thinking_level,
        "gemini_include_thoughts": bool(args.include_thoughts),
        "gemini_temperature": args.temperature,
        "difrost_base_url": DIFROST_GENAI_BASE_URL,
        "difrost_host": DIFROST_HOST,
        "judge_base": judge_base,
        "judge_model": args.judge_model,
        "judge_timeout": args.judge_timeout,
        "judge_retries": args.judge_retries,
        "judge_max_tokens": args.judge_max_tokens,
        "evaluation_method": "gemini_freeform_answer_qwen_text_llm_judge",
        "created_at": now_iso(),
    }


def prepare_output(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.reset_output:
        if args.resume:
            raise ValueError("--reset-output is incompatible with --resume; pass --no-resume")
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / RUN_CONFIG_JSON
    if args.resume:
        if config_path.exists():
            old = json.loads(config_path.read_text(encoding="utf-8"))
            for key in RESUME_CONFIG_KEYS:
                if old.get(key) != config.get(key):
                    raise ValueError(f"resume refused; config mismatch for {key}")
            old_repeats = int(old.get("n_repeats") or 0)
            new_repeats = int(config.get("n_repeats") or 0)
            if old_repeats and new_repeats < old_repeats:
                raise ValueError(f"resume refused; n_repeats cannot shrink from {old_repeats} to {new_repeats}")
            if new_repeats > old_repeats:
                history = list(old.get("resume_history") or [])
                history.append(
                    {
                        "previous_n_repeats": old_repeats,
                        "new_n_repeats": new_repeats,
                        "updated_at": now_iso(),
                    }
                )
                updated = dict(old)
                updated.update(
                    {
                        "n_repeats": new_repeats,
                        "resume_history": history,
                        "updated_at": now_iso(),
                    }
                )
                config_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        if any(args.output_dir.iterdir()):
            raise FileExistsError(f"{args.output_dir} is not empty but has no {RUN_CONFIG_JSON}; choose another output dir")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in (RESULT_JSONL, BUILD_ERRORS_JSONL):
            (args.output_dir / name).write_text("", encoding="utf-8")
        return
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; pass --resume, --reset-output, or choose another output dir")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in (RESULT_JSONL, BUILD_ERRORS_JSONL):
        (args.output_dir / name).write_text("", encoding="utf-8")


def load_done_attempts(path: Path, rerun_errors: bool) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            key = (str(record.get("record_key") or ""), int(record.get("attempt") or 0))
            if not key[0] or not key[1]:
                continue
            if rerun_errors and record.get("error"):
                continue
            done.add(key)
    return done


def latest_attempt_records(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            key = (str(record.get("record_key") or ""), int(record.get("attempt") or 0))
            if key[0] and key[1]:
                latest[key] = record
    return list(latest.values())


def question_metrics(rows_by_question: dict[str, list[dict[str, Any]]], repeats: int) -> dict[str, Any]:
    question_rows = [rows for rows in rows_by_question.values() if rows]
    question_count = len(question_rows)
    if not question_count:
        return {
            "question_count": 0,
            "attempts": 0,
            "correct": 0,
            f"avg@{repeats}": 0.0,
            f"pass@{repeats}": 0.0,
            f"major@{repeats}": 0.0,
            "completed_questions": 0,
        }
    attempts = sum(len(rows) for rows in question_rows)
    correct = sum(1 for rows in question_rows for rec in rows if bool(rec.get("is_correct")))
    pass_questions = 0
    major_questions = 0
    completed_questions = 0
    for rows in question_rows:
        correct_count = sum(1 for rec in rows if bool(rec.get("is_correct")))
        pass_questions += int(correct_count > 0)
        major_questions += int(correct_count > (len(rows) / 2.0))
        completed_questions += int(len(rows) >= repeats)
    return {
        "question_count": question_count,
        "attempts": attempts,
        "correct": correct,
        f"avg@{repeats}": round(correct / attempts, 6) if attempts else 0.0,
        f"pass@{repeats}": round(pass_questions / question_count, 6),
        f"major@{repeats}": round(major_questions / question_count, 6),
        "completed_questions": completed_questions,
    }


def group_stats(records: list[dict[str, Any]], field: str, repeats: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get(field) or "unknown")].append(rec)
    out: dict[str, Any] = {}
    for value, rows in sorted(grouped.items()):
        by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in rows:
            by_question[str(rec.get("record_key") or "")].append(rec)
        metrics = question_metrics(by_question, repeats)
        attempts = len(rows)
        correct = sum(1 for rec in rows if bool(rec.get("is_correct")))
        model_errors = sum(1 for rec in rows if rec.get("model_error"))
        judge_errors = sum(1 for rec in rows if rec.get("judge_error"))
        out[value] = {
            **metrics,
            "attempts": attempts,
            "correct": correct,
            "model_errors": model_errors,
            "judge_errors": judge_errors,
            "attempt_accuracy": round(correct / attempts, 6) if attempts else 0.0,
        }
    return out


def write_summary_lance_batch(rows: list[dict[str, Any]], output_dir: Path, first_write: bool) -> None:
    if not rows:
        return
    normalized = [{field.name: row.get(field.name) for field in QUESTION_SUMMARY_SCHEMA} for row in rows]
    table = pa.Table.from_pylist(normalized, schema=QUESTION_SUMMARY_SCHEMA)
    lance.write_dataset(table, str(output_dir / QUESTION_SUMMARY_LANCE), mode="overwrite" if first_write else "append")


def summarize(output_dir: Path, *, repeats: int, write_lance: bool, write_batch_size: int = 8192) -> dict[str, Any]:
    result_path = output_dir / RESULT_JSONL
    records = latest_attempt_records(result_path)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_question[str(rec.get("record_key") or "")].append(rec)

    question_summaries: list[dict[str, Any]] = []
    correct_distribution: Counter[int] = Counter()
    for record_key, rows in sorted(by_question.items()):
        rows = sorted(rows, key=lambda rec: int(rec.get("attempt") or 0))
        first = rows[0]
        correct_count = sum(1 for rec in rows if bool(rec.get("is_correct")))
        model_error_count = sum(1 for rec in rows if rec.get("model_error"))
        judge_error_count = sum(1 for rec in rows if rec.get("judge_error"))
        correct_distribution[correct_count] += 1
        question_summaries.append(
            {
                "record_key": record_key,
                "uuid": str(first.get("uuid") or ""),
                "source_index": int(first.get("source_index") or 0),
                "source_kind": str(first.get("source_kind") or ""),
                "source_erqa_id": str(first.get("source_erqa_id") or ""),
                "source_status": str(first.get("source_status") or ""),
                "source_question_type": str(first.get("source_question_type") or ""),
                "source_image_set": str(first.get("source_image_set") or ""),
                "v2_raw_reason": str(first.get("v2_raw_reason") or ""),
                "audit_clear_count": int(first.get("audit_clear_count") or 0),
                "image_count": int(first.get("image_count") or 0),
                "question": str(first.get("question") or ""),
                "option_labels": [str(x) for x in first.get("option_labels") or []],
                "option_texts": [str(x) for x in first.get("option_texts") or []],
                "gold_answer": str(first.get("gold_answer") or ""),
                "correct_count": correct_count,
                "attempt_count": len(rows),
                "model_error_count": model_error_count,
                "judge_error_count": judge_error_count,
                "pred_choice_labels": [str(rec.get("pred_choice_label") or "") for rec in rows],
                "attempt_statuses": [
                    {
                        "attempt": int(rec.get("attempt") or 0),
                        "pred_choice_label": str(rec.get("pred_choice_label") or ""),
                        "direct_is_correct": bool(rec.get("direct_is_correct")),
                        "judge_is_correct": bool(rec.get("judge_is_correct")),
                        "is_correct": bool(rec.get("is_correct")),
                        "model_error": str(rec.get("model_error") or ""),
                        "judge_error": str(rec.get("judge_error") or ""),
                    }
                    for rec in rows
                ],
            }
        )

    summary_path = output_dir / QUESTION_SUMMARY_JSONL
    summary_path.write_text("", encoding="utf-8")
    first_write = True
    batch: list[dict[str, Any]] = []
    for summary in question_summaries:
        batch.append(summary)
        if len(batch) >= write_batch_size:
            append_jsonl(summary_path, batch)
            if write_lance:
                write_summary_lance_batch(batch, output_dir, first_write)
                first_write = False
            batch.clear()
    if batch:
        append_jsonl(summary_path, batch)
        if write_lance:
            write_summary_lance_batch(batch, output_dir, first_write)
            first_write = False

    if write_lance and not question_summaries:
        stale = output_dir / QUESTION_SUMMARY_LANCE
        if stale.exists():
            shutil.rmtree(stale)

    attempts = len(records)
    correct = sum(1 for rec in records if bool(rec.get("is_correct")))
    model_errors = sum(1 for rec in records if rec.get("model_error"))
    judge_errors = sum(1 for rec in records if rec.get("judge_error"))
    metrics = question_metrics(by_question, repeats)
    payload = {
        "question_count": len(by_question),
        "attempts": attempts,
        "correct": correct,
        "model_errors": model_errors,
        "judge_errors": judge_errors,
        "attempt_accuracy": round(correct / attempts, 6) if attempts else 0.0,
        **{k: v for k, v in metrics.items() if k.startswith(("avg@", "pass@", "major@")) or k == "completed_questions"},
        "correct_count_distribution": {str(k): v for k, v in sorted(correct_distribution.items())},
        "by_source_kind": group_stats(records, "source_kind", repeats),
        "by_v2_raw_reason": group_stats(records, "v2_raw_reason", repeats),
        "by_source_status": group_stats(records, "source_status", repeats),
        "by_source_image_set": group_stats(records, "source_image_set", repeats),
        "by_source_erqa_id": group_stats(records, "source_erqa_id", repeats),
        "attempts_jsonl": str(result_path),
        "question_summary_jsonl": str(summary_path),
        "question_summary_lance": str(output_dir / QUESTION_SUMMARY_LANCE) if write_lance else "",
        "updated_at": now_iso(),
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def print_attempt_error(record: dict[str, Any]) -> None:
    model_error = str(record.get("model_error") or "")
    judge_error = str(record.get("judge_error") or "")
    if not model_error and not judge_error:
        return
    tqdm.write(
        "[attempt-error] "
        f"record_key={record.get('record_key')} attempt={record.get('attempt')} "
        f"model_error={model_error[:300]} judge_error={judge_error[:300]}"
    )


def dry_run_preview(ds: lance.LanceDataset, args: argparse.Namespace, judge_base: str) -> None:
    preview: list[dict[str, Any]] = []
    build_errors: list[dict[str, Any]] = []
    row_count = ds.count_rows()
    seen_rows = 0
    selected_rows = 0
    columns = [name for name in SOURCE_COLUMNS if name in ds.schema.names]
    preview_limit = min(args.limit or 5, 5)
    for batch in ds.to_batches(columns=columns, batch_size=args.batch_size, scan_in_order=True):
        for row in batch.to_pylist():
            source_index = seen_rows
            seen_rows += 1
            if source_index < args.start_offset:
                continue
            if args.limit and selected_rows >= args.limit:
                break
            selected_rows += 1
            item, error = build_item(row, source_index, args)
            if error:
                build_errors.append(error)
            elif item is not None and len(preview) < preview_limit:
                preview.append(
                    {
                        "record_key": item["record_key"],
                        "source_index": item["source_index"],
                        "source_kind": item["source_kind"],
                        "source_erqa_id": item["source_erqa_id"],
                        "v2_raw_reason": item["v2_raw_reason"],
                        "audit_clear_count": item["audit_clear_count"],
                        "image_count": item["image_count"],
                        "option_labels": item["option_labels"],
                        "gold_answer": item["gold_answer"],
                        "question": item["question"][:500],
                    }
                )
            if len(preview) >= preview_limit and (not args.limit or selected_rows >= args.limit):
                break
        if len(preview) >= preview_limit and (not args.limit or selected_rows >= args.limit):
            break
    payload = {
        "source_lance": str(args.source_lance.resolve()),
        "row_count": row_count,
        "start_offset": args.start_offset,
        "limit": args.limit,
        "n_repeats": args.n_repeats,
        "total_attempts_if_full_selection": (args.limit or max(row_count - args.start_offset, 0)) * args.n_repeats,
        "gemini_model": args.model,
        "judge_base": judge_base,
        "judge_model": args.judge_model,
        "preview": preview,
        "build_errors_seen": build_errors[:5],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print("Dry run only; no model calls and no output files written.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lance", type=Path, default=DEFAULT_SOURCE_LANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default="consensus_v2_gemini35_high4_qwenjudge")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-inflight", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--write-summary-lance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-images", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model-retries", type=int, default=DEFAULT_MODEL_RETRIES)
    parser.add_argument("--model-timeout", type=float, default=DEFAULT_MODEL_TIMEOUT)
    parser.add_argument("--model-max-tokens", type=int, default=DEFAULT_MODEL_MAX_TOKENS)
    parser.add_argument("--thinking-level", choices=["low", "medium", "high", "xhigh"], default=DEFAULT_THINKING_LEVEL)
    parser.add_argument("--include-thoughts", action=argparse.BooleanOptionalAction, default=DEFAULT_INCLUDE_THOUGHTS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)

    parser.add_argument("--judge-base", type=str, default=DEFAULT_JUDGE_BASE)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-timeout", type=float, default=DEFAULT_JUDGE_TIMEOUT)
    parser.add_argument("--judge-retries", type=int, default=DEFAULT_JUDGE_RETRIES)
    parser.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source_lance.exists():
        raise FileNotFoundError(args.source_lance)
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_inflight < 0:
        raise ValueError("--max-inflight must be >= 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.start_offset < 0:
        raise ValueError("--start-offset must be >= 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.source_lance = args.source_lance.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    judge_base = normalize_qwen_base(args.judge_base)
    ensure_no_proxy_for_base_url(judge_base)
    args.judge_model = resolve_openai_model(judge_base, args.judge_model)

    ds = lance.dataset(str(args.source_lance))
    if args.dry_run:
        dry_run_preview(ds, args, judge_base)
        return
    if args.summarize_only:
        payload = summarize(args.output_dir, repeats=args.n_repeats, write_lance=args.write_summary_lance)
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:8000], flush=True)
        return

    max_inflight = args.max_inflight or args.concurrency
    prepare_output(args, run_config(args, ds, judge_base))
    result_path = args.output_dir / RESULT_JSONL
    build_error_path = args.output_dir / BUILD_ERRORS_JSONL
    done_attempts = load_done_attempts(result_path, args.rerun_errors)
    columns = [name for name in SOURCE_COLUMNS if name in ds.schema.names]

    print(f"Source Lance: {args.source_lance}", flush=True)
    print(f"Rows: {ds.count_rows()}", flush=True)
    print(f"Repeats: {args.n_repeats}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Gemini: model={args.model} thinking={args.thinking_level} include_thoughts={args.include_thoughts}", flush=True)
    print(f"Qwen judge: base={judge_base} model={args.judge_model}", flush=True)
    print(f"Concurrency: {args.concurrency}; max_inflight: {max_inflight}", flush=True)
    print(f"Already done attempts: {len(done_attempts)}", flush=True)

    submitted = 0
    completed = 0
    seen_rows = 0
    selected_rows = 0
    skipped_done_rows = 0
    build_errors: list[dict[str, Any]] = []
    inflight = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        with tqdm(desc="consensus_v2_gemini_qwenjudge", unit="attempt") as pbar:
            for batch in ds.to_batches(columns=columns, batch_size=args.batch_size, scan_in_order=True):
                for row in batch.to_pylist():
                    source_index = seen_rows
                    seen_rows += 1
                    if source_index < args.start_offset:
                        continue
                    if args.limit and selected_rows >= args.limit:
                        break
                    selected_rows += 1

                    record_key = record_key_for(row, source_index)
                    required = [(record_key, attempt) for attempt in range(1, args.n_repeats + 1)]
                    if all(key in done_attempts for key in required):
                        skipped_done_rows += 1
                        continue

                    item, error = build_item(row, source_index, args)
                    if error:
                        build_errors.append(error)
                        if len(build_errors) >= 100:
                            append_jsonl(build_error_path, build_errors)
                            build_errors.clear()
                        continue
                    assert item is not None

                    for attempt in range(1, args.n_repeats + 1):
                        key = (item["record_key"], attempt)
                        if key in done_attempts:
                            continue
                        while len(inflight) >= max_inflight:
                            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                            records = []
                            for fut in done:
                                inflight.pop(fut)
                                record = fut.result()
                                print_attempt_error(record)
                                records.append(record)
                                completed += 1
                                pbar.update(1)
                            append_jsonl(result_path, records)
                        inflight[executor.submit(run_attempt, item, attempt, args, judge_base)] = key
                        submitted += 1
                if args.limit and selected_rows >= args.limit:
                    break

            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                records = []
                for fut in done:
                    inflight.pop(fut)
                    record = fut.result()
                    print_attempt_error(record)
                    records.append(record)
                    completed += 1
                    pbar.update(1)
                append_jsonl(result_path, records)

    if build_errors:
        append_jsonl(build_error_path, build_errors)

    payload = summarize(args.output_dir, repeats=args.n_repeats, write_lance=args.write_summary_lance)
    payload.update(
        {
            "submitted_attempts": submitted,
            "completed_attempts": completed,
            "selected_rows": selected_rows,
            "skipped_done_rows": skipped_done_rows,
            "build_error_jsonl": str(build_error_path),
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:12000], flush=True)
    print(f"Wrote attempts: {result_path}", flush=True)
    print(f"Wrote summary: {args.output_dir / SUMMARY_JSON}", flush=True)


if __name__ == "__main__":
    main()
