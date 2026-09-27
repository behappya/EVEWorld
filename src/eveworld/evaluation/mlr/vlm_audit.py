"""Template-driven vision-language audit of flagged MLR events.

The detector decides a clip's Model Laziness Rate on its own, but a count that drops because two
similar objects were counted as one, because the requested object cannot appear in the scene, or
because the object is plausibly hidden behind the arm, is a property of the request or of the
counting rather than of the generation. The optional audit puts those questions to a
vision-language model and withdraws an event whose flagged timestamps the model rejects, so the
released rate counts only the drops the model could not explain away.

The rubric lives in ``data/prompts/vlm_audit.txt`` and receives four slots -- ``{request_id}``,
``{model}``, ``{instruction}`` and ``{counts}`` -- so a checkout can replace it without touching
this module; :data:`DEFAULT_TEMPLATE` stands in when the file is missing.
:func:`parse_audit_response` reads the verdict back from the JSON object or the ``key: value``
lines the models actually produce. The OpenAI-compatible client is imported inside
:meth:`VLMAuditor._get_client`, so this module imports without the SDK installed and raises only
when the audit first reaches the endpoint; ``OPENAI_API_KEY`` is read from the environment.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eveworld.utils.io import repo_root
from eveworld.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TEMPLATE",
    "PROMPT_PATH",
    "VLMAuditor",
    "build_audit_prompt",
    "load_prompt_template",
    "parse_audit_response",
]

DEFAULT_MODEL = "gpt-4o-mini"
PROMPT_PATH = Path("data/prompts") / "vlm_audit.txt"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

DEFAULT_TEMPLATE = (
    "A detector counted the requested objects on every sampled timestamp of a generated clip\n"
    "and flagged a drop that persisted over consecutive sampled timestamps. Decide, from the\n"
    "frames around the flagged timestamps, whether the objects really are missing.\n"
    "\n"
    "  request_id: {request_id}\n"
    "  model: {model}\n"
    "  instruction: {instruction}\n"
    "  counts: {counts}\n"
    "\n"
    "Return exactly one JSON object, with no text before or after it:\n"
    '{"verdict": "confirm" when the objects really are missing and the event should stand,\n'
    '"reject" when they are visible or plausibly hidden and the event should be withdrawn,\n'
    '"confidence": <number between 0.0 and 1.0>, "reason": "<one sentence naming the evidence '
    'you used>"}\n'
    "\n"
    'When the frames are ambiguous, prefer "reject" with a confidence below 0.5.'
)

_ID_KEYS = ("request_id", "clip_id", "id", "key", "video", "vid")
_INSTRUCTION_KEYS = ("instruction", "prompt", "request", "task")
_MODEL_KEYS = ("model", "model_name", "generator")
_COUNT_KEYS = ("counts", "counts_summary", "detector_counts", "count_table")
_DETECTED_ARRAY_KEYS = ("detected_counts", "found_counts", "predicted_counts")
_EXPECTED_KEYS = ("expected", "required", "target", "requested", "wanted")
_DETECTED_KEYS = ("detected", "found", "count", "measured", "actual")

_FIELDS = ("verdict", "confidence", "reason")
_VERDICT_WORDS = {
    "event": "event",
    "confirm": "event",
    "confirmed": "event",
    "yes": "event",
    "true": "event",
    "keep": "event",
    "stand": "event",
    "no_event": "no_event",
    "reject": "no_event",
    "rejected": "no_event",
    "no": "no_event",
    "false": "no_event",
    "withdraw": "no_event",
    "uncertain": "uncertain",
    "ambiguous": "uncertain",
    "unclear": "uncertain",
}
_CONFIDENCE_WORDS = {"low": 0.25, "medium": 0.5, "high": 0.85}
_LINE_PATTERN = re.compile(
    r"""^[\s>*\-]*["']?(?P<key>\w+)["']?\s*[:=]\s*["']?(?P<value>.*?)["']?\s*,?\s*$""",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def load_prompt_template(path: str | Path | None = None) -> str:
    """Read the audit template from disk.

    Args:
        path: Template file. A relative path is resolved against the repository root returned
            by :func:`~eveworld.utils.io.repo_root`; the default is :data:`PROMPT_PATH`, i.e.
            ``data/prompts/vlm_audit.txt`` under that root.

    Returns:
        The template text, without surrounding whitespace.

    Raises:
        FileNotFoundError: When the resolved path is not a file. The message carries the
            resolved absolute path, so a checkout that lost the template can be fixed in one
            step.
    """
    return _read_template(PROMPT_PATH if path is None else path)


def build_audit_prompt(record: Mapping[str, Any], template: str | Path | None = None) -> str:
    """Fill the audit template for one flagged record.

    Args:
        record: Mapping with the clip id, the instruction, the detector counts and the model
            name. Aliases are accepted for each field (``clip_id`` for ``request_id``,
            ``detected_counts`` for ``counts``, ...); a missing id or instruction renders as the
            empty string and a missing model name falls back to :data:`DEFAULT_MODEL`. When the
            record carries ``expected_counts`` next to a detected array, the two are zipped into
            an expected/detected table.
        template: Template text, or a :class:`Path` to read one from. ``None`` reads
            :data:`PROMPT_PATH` and falls back to :data:`DEFAULT_TEMPLATE` when that file is
            missing, so the audit still runs from a checkout without the prompt directory.

    Returns:
        The prompt with the ``{request_id}``, ``{model}``, ``{instruction}`` and ``{counts}``
        slots filled in.

    Raises:
        TypeError: When ``record`` is not a mapping.
    """
    if not isinstance(record, Mapping):
        raise TypeError(f"record must be a mapping, got {type(record).__name__}")
    values = {
        "request_id": _text(_pick(record, _ID_KEYS)),
        "model": _text(_pick(record, _MODEL_KEYS)) or DEFAULT_MODEL,
        "instruction": _text(_pick(record, _INSTRUCTION_KEYS)),
        "counts": _format_counts(_record_counts(record)),
    }
    return _substitute(_template_text(template), values)


def parse_audit_response(text: Any) -> dict[str, Any]:
    """Read the audit verdict out of a model reply.

    The reply may be a JSON object, a JSON object embedded in prose or a code fence, or plain
    ``key: value`` lines. Parsing tries the whole text first, then the first balanced ``{...}``
    block, then the line pattern, in that order.

    Args:
        text: Raw reply. Anything that is not a string is stringified first; ``None`` reads as
            the empty reply.

    Returns:
        ``{"verdict", "confidence", "reason"}``. ``verdict`` is ``"event"`` for a confirmed
        event (``confirm``, ``yes``, ``true``, ``event``, ...), ``"no_event"`` for a rejected one
        (``reject``, ``no``, ``false``, ``no_event``, ...), ``"uncertain"`` for an ambiguous
        reply, and ``None`` when no verdict could be read. ``confidence`` is clamped to
        ``[0, 1]``, accepting percentages and the words ``low``/``medium``/``high``
        (``0.25``/``0.5``/``0.85``); it is ``None`` when absent. ``reason`` is the stripped
        sentence, ``""`` when absent.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    fields = _json_fields(text)
    if fields is None:
        fields = _line_fields(text)
    return {
        "verdict": _as_verdict(fields.get("verdict")),
        "confidence": _as_confidence(fields.get("confidence")),
        "reason": _text(fields.get("reason")),
    }


class VLMAuditor:
    """Optional vision-language review of flagged MLR events.

    Args:
        model: Chat-completions model name; :data:`DEFAULT_MODEL` by default.
        api_key: API key; defaults to ``OPENAI_API_KEY`` from the environment.
        base_url: OpenAI-compatible endpoint; ``None`` keeps the SDK default.
        template: Template text, or a :class:`Path` to read one from; ``None`` reads
            :data:`PROMPT_PATH` for every record.

    The SDK is imported inside :meth:`_get_client`, so the class is constructible on a machine
    without it; the missing package raises only when :meth:`audit` first reaches the endpoint.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        template: str | Path | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.template = template
        self.client: Any = None

    def audit(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Audit a batch of flagged records, one request each.

        Args:
            records: Mappings in the form :func:`build_audit_prompt` accepts.

        Returns:
            One dict per record, in the input order: ``{"request_id", "verdict", "confidence",
            "reason", "response"}``, where ``response`` is the raw message text and the three
            parsed fields follow :func:`parse_audit_response`. Requests are made sequentially so
            a rate limit is met with back pressure rather than a burst of failures.
        """
        results = []
        for record in records:
            prompt = build_audit_prompt(record, self.template)
            response = self._complete(prompt)
            parsed = parse_audit_response(response)
            results.append(
                {
                    "request_id": _text(_pick(record, _ID_KEYS)),
                    "verdict": parsed["verdict"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "response": response,
                }
            )
        return results

    def audit_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Audit a single record; the result has the shape of one :meth:`audit` element."""
        return self.audit([record])[0]

    def _complete(self, prompt: str) -> str:
        """Send one prompt to the endpoint and return the message text."""
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return _message_text(response)

    def _get_client(self) -> Any:
        """Build the OpenAI client on first use; the SDK import lives here on purpose."""
        if self.client is None:
            from openai import OpenAI

            if not self.api_key:
                logger.warning("OPENAI_API_KEY is not set, the audit request will fail")
            settings: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                settings["base_url"] = self.base_url
            self.client = OpenAI(**settings)
            logger.debug("auditing with %s", self.model)
        return self.client


def _read_template(path: str | Path) -> str:
    """Read a template file, resolving a relative path against the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"prompt template not found: {candidate}")
    return candidate.read_text(encoding="utf-8").strip()


def _template_text(template: str | Path | None) -> str:
    """Template to render: an explicit one, the file at :data:`PROMPT_PATH`, or the built-in one."""
    if isinstance(template, Path):
        return _read_template(template)
    if template is not None:
        return str(template)
    try:
        return load_prompt_template()
    except FileNotFoundError as exc:
        logger.warning("%s, using the built-in audit template", exc)
        return DEFAULT_TEMPLATE


def _record_counts(record: Mapping[str, Any]) -> Any:
    """Count table of a record: a combined field, the expected/detected arrays, or ``None``."""
    counts = _pick(record, _COUNT_KEYS)
    if counts is not None:
        return counts
    expected = record.get("expected_counts")
    detected = _pick(record, _DETECTED_ARRAY_KEYS)
    if expected is None or detected is None:
        return detected
    return [{"expected": pair[0], "detected": pair[1]} for pair in zip(expected, detected)]


def _format_counts(counts: Any) -> str:
    """Render a count table as one prompt line.

    A mapping is ordered by its keys when they are integer-like and a sequence by position;
    every entry becomes ``t<timestamp>: <value>`` and entries are joined with ``"; "``. An entry
    that itself carries an expected and a detected number renders as ``expected=<e>,
    detected=<d>``, so a list of ``{"expected": 2, "detected": 1}`` dictionaries reads the same
    as a list of ``(2, 1)`` pairs. An empty or missing table renders as the empty string.
    """
    items = _count_items(counts)
    return "; ".join(f"t{key}: {_count_text(value)}" for key, value in items)


def _count_items(counts: Any) -> list[tuple[Any, Any]]:
    """``(timestamp, value)`` pairs of a count table, ordered by integer-like key when possible."""
    if isinstance(counts, Mapping):
        try:
            keys = sorted(counts, key=int)
        except (TypeError, ValueError):
            keys = sorted(counts, key=str)
        return [(key, counts[key]) for key in keys]
    if isinstance(counts, (str, bytes)):
        return []
    if isinstance(counts, Sequence):
        return list(enumerate(counts))
    if hasattr(counts, "tolist"):
        return list(enumerate(counts.tolist()))
    return []


def _count_text(value: Any) -> str:
    """One count-table entry: an expected/detected pair when it carries either number."""
    if isinstance(value, Mapping):
        expected = _pick(value, _EXPECTED_KEYS)
        detected = _pick(value, _DETECTED_KEYS)
        if expected is not None or detected is not None:
            return f"expected={_scalar(expected)}, detected={_scalar(detected)}"
        return ", ".join(f"{key}={_scalar(value[key])}" for key in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [_scalar(item) for item in value]
        if len(parts) == 2:
            return f"expected={parts[0]}, detected={parts[1]}"
        return ", ".join(parts)
    return _scalar(value)


def _json_fields(text: str) -> Mapping[str, Any] | None:
    """Fields of the JSON object in ``text``: the whole text first, then the first ``{...}``."""
    stripped = text.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return parsed
    block = _first_object(text)
    if block is None:
        return None
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _first_object(text: str) -> str | None:
    """Substring from the first ``{`` to its matching ``}``, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None


def _line_fields(text: str) -> dict[str, Any]:
    """Fields read from ``key: value`` lines, tolerating bullets, quotes and trailing commas."""
    fields: dict[str, Any] = {}
    for line in text.splitlines():
        match = _LINE_PATTERN.match(line)
        if match is None:
            continue
        key = match.group("key").lower()
        if key not in _FIELDS:
            continue
        value = match.group("value").strip().rstrip(",").strip().strip("\"'").strip()
        if not fields.get(key):
            fields[key] = value
    return fields


def _as_verdict(value: Any) -> str | None:
    """Canonical verdict: ``"event"``, ``"no_event"``, ``"uncertain"`` or ``None``."""
    if isinstance(value, bool):
        return "event" if value else "no_event"
    if value is None:
        return None
    key = str(value).strip().strip("\"'").lower()
    key = re.sub(r"[\s\-]+", "_", key).strip(".,;:!?_")
    return _VERDICT_WORDS.get(key)


def _as_confidence(value: Any) -> float | None:
    """Confidence in ``[0, 1]`` from a number, a percentage or a low/medium/high word."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return _clamp_confidence(float(value))
    text = str(value).strip().strip("\"'").lower()
    if not text:
        return None
    if text in _CONFIDENCE_WORDS:
        return _CONFIDENCE_WORDS[text]
    match = _NUMBER_PATTERN.search(text)
    if match is None:
        return None
    number = float(match.group(0))
    if "%" in text[match.end() :]:
        number /= 100.0
    return _clamp_confidence(number)


def _clamp_confidence(value: float) -> float:
    """Clamp a confidence to the ``[0, 1]`` the template asks for."""
    return float(min(max(value, 0.0), 1.0))


def _pick(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """First present, non-empty value among ``keys``."""
    for key in keys:
        if key in record:
            value = record[key]
            if value is not None and value != "":
                return value
    return None


def _substitute(template: str, values: Mapping[str, str]) -> str:
    """Fill ``{name}`` slots, falling back to plain replacement on a malformed template.

    The rubric embeds a literal JSON example, whose braces :meth:`str.format` cannot parse; the
    fallback replaces the four known slots by name and leaves the example alone.
    """
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        filled = template
        for name, value in values.items():
            filled = filled.replace(f"{{{name}}}", value)
        return filled


def _scalar(value: Any) -> str:
    """Render one count-table cell."""
    return "?" if value is None else str(value)


def _text(value: Any) -> str:
    """Normalize an optional field to a stripped string."""
    return "" if value is None else str(value).strip()


def _message_text(response: Any) -> str:
    """Text of the first choice of a chat-completions response."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("the audit model returned no choice")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    return "" if content is None else str(content).strip()
