"""Qwen instruction-following judge for Table 1 and Table 6.

The judge is shown a handful of frames spread over the clip and asked one binary question: does
the clip follow the instruction? EVEWorld answers yes on 80.16% of the DreamGenBench prompts,
against 73.81% for Standard SFT and 79.37% for the GigaWorld-0 baseline, and the same protocol is
used on the held-out RoboTwin instructions of Table 6. Cost is dominated by the image tokens, so
the clips are reduced to :data:`DEFAULT_FRAME_COUNT` frames whose longest side is at most
:data:`DEFAULT_MAX_SIDE` pixels before they are sent as JPEG data URLs.

The default rubric is the one used for the published numbers and lives in the module as
:data:`DEFAULT_RUBRIC`; a checkout that ships ``data/prompts/qwen_if.txt`` overrides it, and the
``template`` argument of :func:`build_judge_prompt` overrides both. The endpoint is reached
through the OpenAI-compatible client, which is imported inside
:meth:`QwenIFJudge._get_client` so that this module imports without the SDK and without network
access. The same module runs from the shell::

    python -m eveworld.evaluation.instruction_following.qwen_if \\
        --manifest data/metadata/dreamgenbench/manifest.jsonl --output results/qwen_if.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse

import numpy as np

from eveworld.utils.io import list_files, read_jsonl, repo_root, write_json, write_jsonl
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CATEGORIES",
    "DEFAULT_MODEL",
    "DEFAULT_RUBRIC",
    "PROMPT_PATH",
    "QwenIFJudge",
    "build_judge_prompt",
    "frames_for_judge",
    "main",
    "parse_verdict",
    "render_judge_prompt",
    "score_records",
    "summarize",
]

DEFAULT_MODEL = "qwen-plus"
PROMPT_PATH = Path("data/prompts") / "qwen_if.txt"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_FRAME_COUNT = 8
DEFAULT_MAX_SIDE = 512
DEFAULT_JPEG_QUALITY = 85
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 600.0
DEFAULT_RETRIES = 3
CATEGORIES = ("syntax", "spatial", "attributes", "count", "environment", "object", "behavior")

DEFAULT_RUBRIC = (
    "The video shows a robot arm completing a specific task. "
    "Does the video follow the instruction to finish the task: '{question}'? "
    "If it fails to follow the instruction (e.g. miss the object, action or do some other "
    "actions), please answer 0. Answer 0 for No or 1 for Yes. Reply only 0 or 1."
)

_JSON_KEYS = ("verdict", "answer", "result", "prediction", "label", "response")
_QUESTION_KEYS = ("question", "instruction", "prompt")
_VIDEO_KEYS = ("video", "video_path", "path")
_CATEGORY_KEYS = ("category", "metric", "task_type", "split")
_TRUE_WORDS = r"\b(yes|true|correct|follows)\b"
_FALSE_WORDS = r"\b(no|false|incorrect|fails)\b"


class JudgeLike(Protocol):
    """The part of a judge that :func:`score_records` needs."""

    def judge_batch(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return one scored record per input record."""


def build_judge_prompt(
    question: str,
    answer: str | None = None,
    requirement: str | None = None,
    template: str | Path | None = None,
) -> str:
    """Render the judge prompt for one instruction.

    Args:
        question: The instruction the clip is supposed to follow.
        answer: Optional expected answer, for records that carry one.
        requirement: Optional extra constraint the clip has to satisfy.
        template: Template text, or a :class:`Path` to read it from. Defaults to
            :data:`PROMPT_PATH` when that file exists and to :data:`DEFAULT_RUBRIC` otherwise.

    Returns:
        The prompt, with ``{question}``, ``{answer}`` and ``{requirement}`` filled in. A clause
        whose slot the template does not contain is appended at the end, so nothing the caller
        passed is dropped silently.
    """
    return render_judge_prompt(question, load_judge_template(template), answer, requirement)


def load_judge_template(
    template: str | Path | None = None,
    *,
    path: str | Path = PROMPT_PATH,
    fallback: str = DEFAULT_RUBRIC,
) -> str:
    """Return the template to use: an explicit one, the file at ``path``, or ``fallback``.

    A string is taken as template text and a :class:`Path` is read from disk, which keeps the two
    uses apart without a guessing game about file extensions. The default ``path`` is resolved
    against :func:`~eveworld.utils.io.repo_root`, so the module finds its prompt file no matter
    which directory the evaluation was started from.
    """
    if template is not None:
        if isinstance(template, Path):
            return _read_template_text(template)
        return str(template)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    if candidate.is_file():
        return _read_template_text(candidate)
    logger.debug("no template at %s, using the built-in rubric", candidate)
    return fallback


def render_judge_prompt(
    question: str,
    template: str,
    answer: str | None = None,
    requirement: str | None = None,
) -> str:
    """Fill a judge template with the instruction and the optional extra constraints."""
    text = str(question or "").strip()
    if not text:
        raise ValueError("the judge question must not be empty")
    values = {"question": text, "answer": _text(answer), "requirement": _text(requirement)}
    rendered = _substitute(str(template), values)
    missing = [
        clause
        for name, clause in (
            ("answer", f"The expected answer is: '{values['answer']}'."),
            ("requirement", f"The task requirement is: '{values['requirement']}'."),
        )
        if values[name] and f"{{{name}}}" not in str(template)
    ]
    if missing:
        rendered = f"{rendered.rstrip()} {' '.join(missing)}"
    return rendered


def parse_verdict(text: Any) -> bool | None:
    """Read a yes/no verdict out of a judge response.

    Handles the replies the judge models actually produce: a bare ``0`` or ``1``, ``yes`` or
    ``no``, ``true`` or ``false``, a JSON object such as ``{"verdict": false}``, and the
    ``A``/``B`` labels of a two-way choice, where ``A`` is the affirmative option.

    Args:
        text: Raw judge response, or an already parsed value.

    Returns:
        ``True`` when the clip follows the instruction, ``False`` when it does not, and ``None``
        when the response says neither, so that the caller can record it as an undecided judge
        call instead of scoring it as a failure.
    """
    if isinstance(text, bool):
        return text
    if isinstance(text, int):
        return bool(text) if text in (0, 1) else None
    value = "" if text is None else str(text).strip()
    if not value:
        return None
    payload = _json_verdict(value)
    if payload is not None:
        return payload
    if value.startswith("1"):
        return True
    if value.startswith("0"):
        return False
    match = re.search(r"\b([01])\b", value)
    if match:
        return match.group(1) == "1"
    lowered = value.lower()
    has_true = re.search(_TRUE_WORDS, lowered) is not None
    has_false = re.search(_FALSE_WORDS, lowered) is not None
    if has_true != has_false:
        return has_true
    label = lowered.strip(" \t.\"'()[]*")
    if label == "a":
        return True
    if label == "b":
        return False
    return None


def frames_for_judge(
    frames: np.ndarray,
    count: int = DEFAULT_FRAME_COUNT,
    max_side: int = DEFAULT_MAX_SIDE,
) -> list[np.ndarray]:
    """Select the frames a judge is shown, downscaled to a bounded token cost.

    Args:
        frames: Clip ``(T, H, W, 3)`` uint8 RGB, or a single ``(H, W, 3)`` frame.
        count: Frames to select, spread evenly over the clip and deduplicated, so a clip shorter
            than ``count`` contributes every frame once.
        max_side: Longest side of the returned frames in pixels; ``0`` keeps the original size.

    Returns:
        The selected frames in temporal order, each ``(H', W', 3)`` uint8 RGB with
        ``max(H', W') <= max_side``.
    """
    clip = _as_frames(frames)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    return [_resize_longest_side(clip[index], max_side) for index in _sample_indices(len(clip), count)]


def frame_data_urls(
    frames: Sequence[np.ndarray],
    *,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> list[str]:
    """Encode frames as JPEG data URLs, the image format both judge APIs accept.

    The frames are RGB while OpenCV writes BGR, so every frame is converted before encoding;
    without that step the judge would see red and blue swapped, which matters for questions about
    object colors.
    """
    import cv2

    urls = []
    for frame in frames:
        bgr = cv2.cvtColor(_as_frame(frame), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("could not encode a frame as JPEG")
        urls.append("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"))
    return urls


def normalize_base_url(base_url: str | None) -> str:
    """Normalize an endpoint into an OpenAI-compatible base URL.

    ``host`` and ``host:port`` become ``http://host:port/v1``, a full URL is kept as it is, and a
    URL without a path gets the ``/v1`` suffix the OpenAI client expects.
    """
    value = str(base_url or "").strip()
    if not value:
        return DEFAULT_BASE_URL
    if value.startswith("http://") or value.startswith("https://"):
        out = value.rstrip("/")
        if not urlparse(out).path or urlparse(out).path == "/":
            out = f"{out}/v1"
        return out
    host_port = value if ":" in value else f"{value}:8000"
    return f"http://{host_port}/v1"


def ensure_no_proxy(base_url: str) -> None:
    """Add the endpoint host to ``NO_PROXY`` so that a local server is not routed through a proxy."""
    host = urlparse(base_url).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [item.strip() for item in current.split(",") if item.strip()]
        if host not in parts:
            parts.append(host)
        os.environ[key] = ",".join(parts)


class QwenIFJudge:
    """Instruction-following judge backed by an OpenAI-compatible Qwen endpoint.

    Args:
        model: Model name as the endpoint knows it.
        api_key: API key; defaults to ``DASHSCOPE_API_KEY``, then ``OPENAI_API_KEY``, then the
            dummy key the OpenAI client accepts for a local server that does not check keys.
        base_url: Endpoint; defaults to ``QWEN_BASE_URL``, then the DashScope compatible mode.
        temperature: Sampling temperature of the judge, kept at zero for reproducible verdicts.
        max_tokens: Response budget of one judgement.
        timeout: Per-request timeout in seconds.
        disable_thinking: Turn off the reasoning mode of hybrid models; if the endpoint rejects
            the flag the request is repeated without it.
        retries: Attempts per clip; a request error or an unparsable answer both count.
        frame_count: Frames sent to the judge.
        max_side: Longest side of those frames in pixels.
        prompt_template: Template text or :class:`Path` for the judgement prompt.
        client: OpenAI-compatible client to use instead of building one, for offline runs.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
        disable_thinking: bool = True,
        retries: int = DEFAULT_RETRIES,
        frame_count: int = DEFAULT_FRAME_COUNT,
        max_side: int = DEFAULT_MAX_SIDE,
        prompt_template: str | Path | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = normalize_base_url(base_url or os.environ.get("QWEN_BASE_URL"))
        self.api_key = (
            api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.disable_thinking = bool(disable_thinking)
        self.retries = int(retries)
        self.frame_count = int(frame_count)
        self.max_side = int(max_side)
        self.prompt_template = prompt_template
        self.client = client
        ensure_no_proxy(self.base_url)

    def judge(
        self,
        frames: np.ndarray | str | Path,
        question: str,
        *,
        answer: str | None = None,
        requirement: str | None = None,
    ) -> bool | None:
        """Judge one clip; returns the verdict, or ``None`` when the model declined to decide."""
        verdict, _ = self._ask(frames, question, answer=answer, requirement=requirement)
        return verdict

    def judge_batch(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Judge a batch of records, keeping failures in the result instead of raising.

        Each record needs a question under ``question``, ``instruction`` or ``prompt``, and either
        an ``(T, H, W, 3)`` uint8 clip under ``frames`` or a path under ``video``,
        ``video_path`` or ``path``. Frame arrays are dropped from the returned records, so the
        result can be written out as JSONL.

        Returns:
            One record per input, carrying the input fields plus ``verdict``, ``raw_text`` and
            ``error``; ``verdict`` is ``None`` for a record whose judgement failed.
        """
        scored: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            item: dict[str, Any] = {"index": index, "verdict": None, "raw_text": "", "error": None}
            try:
                item.update(_public_record(record, index))
                source = record.get("frames")
                if source is None:
                    source = item.get("video")
                if source is None:
                    raise ValueError("record has neither a 'frames' array nor a video path")
                verdict, raw_text = self._ask(
                    source,
                    str(item["question"]),
                    answer=item.get("answer"),
                    requirement=item.get("requirement"),
                )
                item["verdict"] = verdict
                item["raw_text"] = raw_text
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("judge failed on record %d: %s", index, exc)
            scored.append(item)
        return scored

    def _ask(
        self,
        frames: np.ndarray | str | Path,
        question: str,
        *,
        answer: str | None = None,
        requirement: str | None = None,
    ) -> tuple[bool, str]:
        """Run the judgement with retries and return the verdict together with the raw response."""
        prompt = build_judge_prompt(question, answer, requirement, self.prompt_template)
        clip = frames if isinstance(frames, np.ndarray) else self._load_clip(frames)
        selected = frames_for_judge(clip, self.frame_count, self.max_side)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": url}} for url in frame_data_urls(selected))
        messages = [{"role": "user", "content": content}]
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.retries) + 1):
            try:
                raw_text = _message_text(self._chat(messages))
                verdict = parse_verdict(raw_text)
                if verdict is None:
                    raise ValueError(f"unparsable judge verdict: {raw_text.strip()[:120]!r}")
                return verdict, raw_text
            except Exception as exc:
                last_error = exc
                logger.debug("judge attempt %d failed: %s", attempt, exc)
                if attempt < max(1, self.retries):
                    time.sleep(min(2.0 * attempt, 8.0))
        raise RuntimeError(f"judge failed after {self.retries} attempts: {last_error}") from last_error

    def _chat(self, messages: list[dict[str, Any]]) -> Any:
        """Send one chat completion, dropping the thinking flag when the endpoint rejects it."""
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if not self.disable_thinking:
            return client.chat.completions.create(**kwargs)
        try:
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
            return client.chat.completions.create(**kwargs, extra_body=extra)
        except Exception as exc:
            logger.debug("endpoint rejected the thinking flag (%s), retrying without it", exc)
            return client.chat.completions.create(**kwargs)

    def _get_client(self) -> Any:
        """Build the OpenAI-compatible client on first use."""
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            logger.debug("judging with %s at %s", self.model, self.base_url)
        return self.client

    def _load_clip(self, path: str | Path) -> np.ndarray:
        """Decode a clip with the repository video reader."""
        from eveworld.data.transforms.video import load_video

        return _as_frames(load_video(Path(path)))


def score_records(records: Sequence[Mapping[str, Any]], judge: JudgeLike) -> dict[str, Any]:
    """Judge a batch of records and count how many produced a usable verdict.

    Args:
        records: Records in the form :meth:`QwenIFJudge.judge_batch` accepts.
        judge: Judge to run them through, for example :class:`QwenIFJudge`.

    Returns:
        ``{"records", "n", "decided", "errors"}``, where ``decided`` counts the records that came
        back with a verdict and is the denominator of :func:`summarize`.
    """
    scored = list(judge.judge_batch(records))
    errors = sum(1 for item in scored if item.get("error"))
    return {"records": scored, "n": len(scored), "decided": len(scored) - errors, "errors": errors}


def summarize(scores: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Instruction-following accuracy overall and per category, in percent.

    Args:
        scores: The mapping returned by :func:`score_records`, or the list of scored records.

    Returns:
        ``{"overall", "n", "decided", "errors"}`` plus one rounded percentage per category that
        occurs in the data. A category the records do not carry is left out rather than reported
        as zero, so an evaluation that only covers part of the taxonomy does not show empty rows
        in the table.

    The denominator is the set of decided records: a judge call that errored or answered
    ambiguously is reported under ``errors`` instead of counting as a failure to follow the
    instruction.
    """
    records = _score_list(scores)
    decided = [item for item in records if item.get("verdict") is not None]
    summary: dict[str, Any] = {
        "overall": _percent(decided),
        "n": len(records),
        "decided": len(decided),
        "errors": sum(1 for item in records if item.get("error")),
    }
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for item in decided:
        category = _record_category(item)
        if category:
            buckets.setdefault(category, []).append(item)
    ordered = [name for name in CATEGORIES if name in buckets]
    ordered.extend(name for name in buckets if name not in CATEGORIES)
    for name in ordered:
        summary[name] = _percent(buckets[name])
    return summary


def main(argv: list[str] | None = None) -> int:
    """Judge a directory of clips or a manifest and write the summary."""
    parser = argparse.ArgumentParser(
        prog="eveworld.evaluation.instruction_following.qwen_if",
        description="Score clips with the Qwen instruction-following judge.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="JSONL with a prompt and a video per line"
    )
    parser.add_argument(
        "--video-dir", type=Path, default=None, help="clips whose prompt is read from the file name"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="summary JSON and, next to it, a JSONL of records"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name at the endpoint")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--limit", type=int, default=None, help="judge at most this many records")
    args = parser.parse_args(argv)

    try:
        records = _load_cli_records(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not records:
        print("error: no clips to judge; pass --manifest or --video-dir with clips in it", file=sys.stderr)
        return 2

    judge = QwenIFJudge(
        model=args.model,
        base_url=args.base_url,
        frame_count=args.frame_count,
        max_side=args.max_side,
    )
    scores = score_records(records, judge)
    summary = summarize(scores)
    if args.output is not None:
        write_jsonl(scores["records"], args.output.with_suffix(".jsonl"))
        write_json(summary, args.output)
        logger.info("wrote %s and %s", args.output.with_suffix(".jsonl"), args.output)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _load_cli_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read the records to judge from a manifest or from a directory of clips."""
    if args.manifest is not None:
        records = list(read_jsonl(args.manifest))
    elif args.video_dir is not None:
        records = [
            {"video": str(path), "prompt": _prompt_from_path(path)}
            for path in list_files(args.video_dir, suffix=".mp4")
        ]
    else:
        raise ValueError("pass --manifest or --video-dir")
    if args.limit is not None:
        records = records[: args.limit]
    return records


def _prompt_from_path(path: Path) -> str:
    """Recover the instruction from a clip name such as ``0007_pick_up_the_cup.mp4``."""
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return stem.replace("_", " ")


def _public_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Copy a record for the results file, dropping the frame array and normalizing the paths."""
    item: dict[str, Any] = {}
    for key, value in record.items():
        if key == "frames" or key in ("verdict", "raw_text", "error"):
            continue
        item[key] = str(value) if isinstance(value, Path) else value
    item["index"] = index
    item["question"] = _record_text(record, _QUESTION_KEYS, "question")
    video = _record_text(record, _VIDEO_KEYS, "video", required=False)
    if video:
        item["video"] = video
    return item


def _record_text(
    record: Mapping[str, Any],
    keys: Sequence[str],
    name: str,
    *,
    required: bool = True,
) -> str | None:
    """Read the first non-empty string field of ``record`` among ``keys``."""
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    if required:
        raise ValueError(f"record has no {name}; expected one of {list(keys)}")
    return None


def _record_category(record: Mapping[str, Any]) -> str:
    """Lowercase category of a scored record, matched against :data:`CATEGORIES`."""
    value = _record_text(record, _CATEGORY_KEYS, "category", required=False)
    if not value:
        return ""
    lowered = value.strip().lower()
    for name in CATEGORIES:
        if lowered == name:
            return name
    return lowered


def _score_list(scores: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Accept either the mapping of :func:`score_records` or a bare list of scored records."""
    if isinstance(scores, Mapping):
        records = scores.get("records") or []
        return list(records)
    return list(scores)


def _percent(items: Sequence[Mapping[str, Any]]) -> float | None:
    """Share of the records that followed their instruction, in percent with two decimals."""
    if not items:
        return None
    followed = sum(1 for item in items if item.get("verdict"))
    return round(100.0 * followed / len(items), 2)


def _json_verdict(value: str) -> bool | None:
    """Read a verdict out of a JSON response such as ``{"verdict": 0}``."""
    if not value.startswith(("{", "[")):
        return None
    try:
        payload = json.loads(value)
    except ValueError:
        return None
    return _payload_verdict(payload)


def _payload_verdict(payload: Any) -> bool | None:
    """Read a verdict out of a decoded JSON payload."""
    if isinstance(payload, dict):
        for key, item in payload.items():
            if str(key).strip().lower() in _JSON_KEYS:
                verdict = _scalar_verdict(item)
                if verdict is not None:
                    return verdict
        for item in payload.values():
            verdict = _scalar_verdict(item)
            if verdict is not None:
                return verdict
        return None
    return _scalar_verdict(payload)


def _scalar_verdict(value: Any) -> bool | None:
    """Read a verdict out of a JSON scalar."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return parse_verdict(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _substitute(template: str, values: Mapping[str, str]) -> str:
    """Fill the ``{name}`` slots, falling back to plain replacement on a malformed template."""
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        filled = template
        for name, value in values.items():
            filled = filled.replace(f"{{{name}}}", value)
        return filled


def _text(value: Any) -> str:
    """Normalize an optional field to a stripped string."""
    return "" if value is None else str(value).strip()


def _message_text(response: Any) -> str:
    """Collect the text of a chat completion response, whichever shape the endpoint returns."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("the judge returned no choice")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if isinstance(content, str):
        return content.strip()
    parts = []
    for item in content or []:
        text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _read_template_text(path: Path) -> str:
    """Read a prompt template, resolving a relative path against the repository root."""
    candidate = path if path.is_absolute() else repo_root() / path
    if not candidate.is_file():
        raise FileNotFoundError(f"prompt template not found: {candidate}")
    return candidate.read_text(encoding="utf-8").strip()


def _sample_indices(total: int, count: int) -> list[int]:
    """Evenly spread indices over ``total`` frames, rounded and deduplicated as in the campaign."""
    if total <= 0:
        raise ValueError("cannot sample frames from an empty clip")
    if count == 1:
        return [total // 2]
    return sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})


def _resize_longest_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale a frame so that its longest side is at most ``max_side`` pixels."""
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    import cv2

    scale = max_side / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _as_frames(frames: np.ndarray) -> np.ndarray:
    """View ``frames`` as an ``(T, H, W, 3)`` uint8 clip, promoting a single frame to a clip."""
    clip = np.asarray(frames)
    if clip.ndim == 3:
        clip = clip[np.newaxis]
    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(f"frames must be (H, W, 3) or (T, H, W, 3) RGB, got {clip.shape}")
    if clip.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 RGB, got {clip.dtype}")
    return clip


def _as_frame(frame: np.ndarray) -> np.ndarray:
    """Validate a single ``(H, W, 3)`` uint8 RGB frame."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"frame must be (H, W, 3) uint8 RGB, got {array.shape} {array.dtype}")
    return array


if __name__ == "__main__":
    raise SystemExit(main())
