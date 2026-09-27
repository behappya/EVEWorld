"""Gemini instruction-following judge for Table 1 and Table 6.

The second of the two instruction-following judges. It asks the same binary question as
:mod:`~eveworld.evaluation.instruction_following.qwen_if` -- does the clip follow the instruction?
-- and shares that module's prompt rendering, verdict parsing and scoring, so the two judges differ
only in their transport and their model. EVEWorld answers yes on 60.85% of the DreamGenBench
prompts, against 53.57% for Standard SFT and 60.19% for the GigaWorld-0 baseline.

Gemini does not take JPEG data URLs through an OpenAI-compatible client, so the frames are sent as
PNG ``file_uri`` parts, images first and the instruction last, and the reply is read back from the
non-thought parts of the returned candidates. The published numbers were produced through a
Difrost-style GenAI gateway, which is what ``base_url`` selects: a bearer token, an explicit
``Host`` header and a per-client session affinity header, with certificate verification left to the
gateway. Without ``base_url`` the judge talks to the public Gemini API with :data:`DEFAULT_MODEL`.

``google-genai`` is imported inside :meth:`GeminiIFJudge._get_client`, so this module imports
without the SDK. The same module runs from the shell::

    python -m eveworld.evaluation.instruction_following.gemini_if \\
        --manifest data/metadata/dreamgenbench/manifest.jsonl --output results/gemini_if.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np

from eveworld.evaluation.instruction_following.qwen_if import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_MAX_SIDE,
    ensure_no_proxy,
    frames_for_judge,
    load_judge_template as _load_judge_template,
    parse_verdict,
    render_judge_prompt,
    score_records,
    summarize,
)
from eveworld.utils.io import list_files, read_jsonl, write_json, write_jsonl
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_RUBRIC",
    "PROMPT_PATH",
    "GeminiIFJudge",
    "build_judge_prompt",
    "frame_png_urls",
    "load_judge_template",
    "main",
]

DEFAULT_MODEL = "gemini-2.5-flash"
PROMPT_PATH = Path("data/prompts") / "gemini_if.txt"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 32000
DEFAULT_TIMEOUT = 600.0
DEFAULT_RETRIES = 3
DEFAULT_THINKING_LEVEL = "low"

DEFAULT_RUBRIC = (
    "The video shows a robot arm completing a specific task. "
    "Please evaluate: if the video follows the instruction to finish the task '{question}', "
    "give a positive score. Reply only '0' for No or '1' for Yes."
)

_QUESTION_KEYS = ("question", "instruction", "prompt")
_VIDEO_KEYS = ("video", "video_path", "path")


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
        The prompt, with ``{question}``, ``{answer}`` and ``{requirement}`` filled in.
    """
    return render_judge_prompt(
        question,
        load_judge_template(template, path=PROMPT_PATH, fallback=DEFAULT_RUBRIC),
        answer,
        requirement,
    )


def load_judge_template(
    template: str | Path | None = None,
    *,
    path: str | Path = PROMPT_PATH,
    fallback: str = DEFAULT_RUBRIC,
) -> str:
    """Return the template to use: an explicit one, the file at ``path``, or ``fallback``.

    The default ``path`` is this module's :data:`PROMPT_PATH`, so a caller that passes no argument
    gets the Gemini template rather than the Qwen one.
    """
    return _load_judge_template(template, path=path, fallback=fallback)


def frame_png_urls(frames: Sequence[np.ndarray]) -> list[str]:
    """Encode frames as PNG data URLs, the image format the GenAI parts expect.

    The frames are RGB while OpenCV writes BGR, so every frame is converted before encoding;
    without that step the judge would see red and blue swapped, which matters for questions about
    object colors. PNG is used rather than JPEG because it is what the reference pipeline sends.
    """
    import cv2

    urls = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"a frame must have shape (H, W, 3), got {array.shape}")
        bgr = cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("could not encode a frame as PNG")
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        urls.append(f"data:image/png;base64,{payload}")
    return urls


def normalize_base_url(base_url: str | None) -> str:
    """Normalize an endpoint into a GenAI base URL.

    ``host`` and ``host:port`` become ``http://host:8000/api/v1``, the path the gateway serves, and
    a full URL is kept as it is.
    """
    value = str(base_url or "").strip()
    if not value:
        raise ValueError("a gateway base URL must not be empty")
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    host_port = value if ":" in value else f"{value}:8000"
    return f"http://{host_port}/api/v1"


class GeminiIFJudge:
    """Instruction-following judge backed by the Gemini GenAI API.

    Args:
        model: Model name as the endpoint knows it.
        api_key: API key of the public API; defaults to ``GEMINI_API_KEY``, then
            ``GOOGLE_API_KEY``, and is unused when a gateway ``base_url`` is configured.
        base_url: GenAI gateway endpoint; defaults to ``GEMINI_BASE_URL``, then
            ``DIFROST_GENAI_BASE_URL``. When neither is set the public Gemini API is used.
        temperature: Sampling temperature of the judge, kept at zero for reproducible verdicts.
        max_tokens: Response budget of one judgement, including any thinking tokens.
        timeout: Per-request timeout in seconds.
        thinking_level: Reasoning effort of the model, one of ``"low"``, ``"medium"`` or
            ``"high"``; defaults to ``DIFROST_THINKING_LEVEL`` and then to
            :data:`DEFAULT_THINKING_LEVEL`. ``None`` or an empty string leaves the thinking
            configuration to the model.
        include_thoughts: Ask the gateway to return the reasoning text as well; it is dropped from
            the verdict either way.
        retries: Attempts per clip; a request error or an unparsable answer both count.
        frame_count: Frames sent to the judge.
        max_side: Longest side of those frames in pixels.
        api_token: Bearer token of the gateway; defaults to ``DIFROST_API_TOKEN``.
        host: ``Host`` header of the gateway; defaults to ``DIFROST_HOST`` and then to the host of
            ``base_url``.
        prompt_template: Template text or :class:`Path` for the judgement prompt.
        client: GenAI client to use instead of building one, for offline runs.
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
        thinking_level: str | None = None,
        include_thoughts: bool = False,
        retries: int = DEFAULT_RETRIES,
        frame_count: int = DEFAULT_FRAME_COUNT,
        max_side: int = DEFAULT_MAX_SIDE,
        api_token: str | None = None,
        host: str | None = None,
        prompt_template: str | Path | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        raw_base = str(
            base_url or os.environ.get("GEMINI_BASE_URL") or os.environ.get("DIFROST_GENAI_BASE_URL") or ""
        ).strip()
        self.base_url = normalize_base_url(raw_base) if raw_base else ""
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        self.api_token = api_token or os.environ.get("DIFROST_API_TOKEN") or ""
        self.host = host or os.environ.get("DIFROST_HOST") or (urlparse(self.base_url).hostname or "")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        level = thinking_level if thinking_level is not None else os.environ.get("DIFROST_THINKING_LEVEL")
        self.thinking_level = str(level or DEFAULT_THINKING_LEVEL).strip()
        self.include_thoughts = bool(include_thoughts)
        self.retries = int(retries)
        self.frame_count = int(frame_count)
        self.max_side = int(max_side)
        self.prompt_template = prompt_template
        self.client = client
        if self.base_url:
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
        urls = frame_png_urls(frames_for_judge(clip, self.frame_count, self.max_side))
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.retries) + 1):
            try:
                raw_text = self._generate(prompt, urls)
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

    def _generate(self, prompt: str, urls: Sequence[str]) -> str:
        """Send one judgement request and return the model's text answer."""
        from google.genai import types as gt

        parts = [gt.Part.from_uri(file_uri=url, mime_type="image/png") for url in urls]
        parts.append(gt.Part.from_text(text=prompt))
        contents = [gt.Content(role="user", parts=parts)]
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=contents,
            config=self._config(gt),
        )
        return _response_text(response)

    def _config(self, gt: Any) -> Any:
        """Build the generation config, leaving out thinking when no level was requested."""
        kwargs: dict[str, Any] = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.thinking_level:
            kwargs["thinking_config"] = gt.ThinkingConfig(
                thinking_level=self.thinking_level,
                include_thoughts=self.include_thoughts,
            )
        return gt.GenerateContentConfig(**kwargs)

    def _get_client(self) -> Any:
        """Build the GenAI client on first use."""
        if self.client is None:
            from google import genai
            from google.genai import types as gt

            if self.base_url:
                self.client = _gateway_client(
                    genai, gt, self.base_url, self.api_token, self.host, self.timeout
                )
                logger.debug("judging with %s through %s", self.model, self.base_url)
            else:
                if not self.api_key:
                    raise RuntimeError("set GEMINI_API_KEY or pass api_key to use the public Gemini API")
                options = gt.HttpOptions(timeout=int(self.timeout * 1000))
                self.client = genai.Client(api_key=self.api_key, http_options=options)
                logger.debug("judging with %s through the public Gemini API", self.model)
        return self.client

    def _load_clip(self, path: str | Path) -> np.ndarray:
        """Decode a clip with the repository video reader."""
        from eveworld.data.transforms.video import load_video

        return np.asarray(load_video(Path(path)))


def main(argv: list[str] | None = None) -> int:
    """Judge a directory of clips or a manifest and write the summary."""
    parser = argparse.ArgumentParser(
        prog="eveworld.evaluation.instruction_following.gemini_if",
        description="Score clips with the Gemini instruction-following judge.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="JSONL with prompt and video per line")
    parser.add_argument("--video-dir", type=Path, default=None, help="clips, prompt read from the file name")
    parser.add_argument("--output", type=Path, default=None, help="summary JSON plus a JSONL of records")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name at the endpoint")
    parser.add_argument("--base-url", default=None, help="GenAI gateway endpoint; omit for the public API")
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--thinking-level", default=None, choices=["low", "medium", "high"])
    parser.add_argument("--include-thoughts", action=argparse.BooleanOptionalAction, default=False)
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

    judge = GeminiIFJudge(
        model=args.model,
        base_url=args.base_url,
        frame_count=args.frame_count,
        max_side=args.max_side,
        thinking_level=args.thinking_level,
        include_thoughts=args.include_thoughts,
    )
    scores = score_records(records, judge)
    summary = summarize(scores)
    if args.output is not None:
        write_jsonl(scores["records"], args.output.with_suffix(".jsonl"))
        write_json(summary, args.output)
        logger.info("wrote %s and %s", args.output.with_suffix(".jsonl"), args.output)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _gateway_client(
    genai: Any,
    gt: Any,
    base_url: str,
    api_token: str,
    host: str,
    timeout: float,
) -> Any:
    """Build a client for a Difrost-style gateway, which authenticates by header.

    The gateway terminates TLS with its own certificate, so verification is disabled for this
    client and the ``Host`` header carries the name it routes on.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    headers = {
        "Authorization": f"Bearer {api_token}",
        "X-Difrost-Session-Affinity": uuid.uuid4().hex,
    }
    if host:
        headers["Host"] = host
    options = gt.HttpOptions(
        base_url=base_url,
        api_version="genai",
        headers=headers,
        timeout=int(timeout * 1000),
        async_client_args={"ssl": context},
        client_args={"verify": False},
    )
    return genai.Client(vertexai=False, api_key="unused", http_options=options)


def _response_text(response: Any) -> str:
    """Collect the answer parts of a response, skipping the reasoning parts."""
    texts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                texts.append(str(text))
    if texts:
        return "\n\n".join(texts)
    fallback = getattr(response, "text", "") or ""
    return str(fallback)


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


if __name__ == "__main__":
    raise SystemExit(main())
