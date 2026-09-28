"""Prompt parsing: turn an instruction into target, source and destination.

The parser is deliberately dependency-free and deterministic. Internally it splits the
prompt into clauses, finds the verb of each clause, splits the clause body on the
spatial marker that links two objects (`from`, `to`, `on`, `off`, `in`, `above`, ...),
and records which role that marker assigns:

* a source marker (`from`, `off`, `above`, `along`) names where the object was,
* a destination marker (`to`, `into`, `onto`, `inside`, `on top of`, `next to`) names
  where the object goes,
* an ambiguous marker (`on`, `in`, `at`, `under`, `beneath`, `beside`, `near`) is read
  as the source when it appears in the first clause and from the verb otherwise.

So `"move the cup from the shelf to the tray"` yields target `"cup"`, source `"shelf"`
and destination `"tray"`, while `"pick up the green bottle on the blue tray and place it
on the kitchen island"` yields source `"blue tray"` from the first clause and
destination `"kitchen island"` from the second. Phrases without any recognisable
structure fall back to the whole body as the target (`"red block"`).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

__all__ = [
    "PLACEMENT_VERBS",
    "VERBS",
    "ParsedInstruction",
    "parse_instruction",
    "parse_instructions",
]

VERBS = frozenset(
    {
        "bring",
        "carry",
        "deposit",
        "drop",
        "grab",
        "hold",
        "insert",
        "keep",
        "lay",
        "lift",
        "move",
        "pick",
        "place",
        "push",
        "put",
        "relocate",
        "remove",
        "set",
        "shake",
        "take",
        "transfer",
        "wait",
    }
)

PLACEMENT_VERBS = frozenset(
    {
        "bring",
        "carry",
        "deposit",
        "drop",
        "insert",
        "lay",
        "move",
        "place",
        "put",
        "relocate",
        "set",
        "transfer",
    }
)

_SOURCE_MARKERS = frozenset({"above", "along", "from", "off", "off of"})
_DESTINATION_MARKERS = frozenset({"inside", "into", "next to", "on top of", "onto", "to"})
_AMBIGUOUS_MARKERS = frozenset({"at", "beneath", "beside", "in", "near", "on", "under"})

_PARTICLES = ("up", "down", "back", "out", "over", "apart", "away", "aside")
_PRONOUNS = ("it", "them")
_MANNER_ADVERBS = ("steady", "still", "fixed", "upright")
_DIRECTIONAL_ADVERBS = ("in front of", "towards", "toward", "alongside")
_DETERMINERS = ("the", "a", "an")
_NON_OBJECT_HEADS = frozenset({"position", "steady", "still"})

_CLAUSE_RE = re.compile(r"\s+(?:and|then|while)\s+|[,;]")
_VERB_RE = re.compile(r"\b(" + "|".join(sorted(VERBS)) + r")\b")
_LEADING_RE = re.compile(r"^(?:" + "|".join(_PARTICLES + _PRONOUNS) + r")\s+")
_TRAILING_MANNER_RE = re.compile(r"\s+(?:" + "|".join(_MANNER_ADVERBS) + r")$")
_TRAILING_PARTICLE_RE = re.compile(r"\s+(?:" + "|".join(_PARTICLES) + r")$")
_DIRECTIONAL_RE = re.compile(r"\s+(?:" + "|".join(_DIRECTIONAL_ADVERBS) + r")\s+.*$")
_DETERMINER_RE = re.compile(r"^(?:" + "|".join(_DETERMINERS) + r")\s+")
_WAIT_RE = re.compile(
    r"^(?:until|till|for)\s+(?P<object>.+?)" r"(?:\s+(?:(?:to\s+)?(?:stops?|ceases?|settles?|comes?|remains?|reaches?)|is|are)\b.*)?$"
)
_PUNCTUATION = " \t\r\n.,;:!?\"'"


def _marker_re(markers: Iterable[str]) -> re.Pattern[str]:
    """Alternation over `markers`, longest first so `onto` never matches as `on`."""
    ordered = sorted(markers, key=lambda marker: (-len(marker), marker))
    return re.compile(r"\b(" + "|".join(ordered) + r")\b\s+")


_MARKER_RE = _marker_re(_SOURCE_MARKERS | _DESTINATION_MARKERS | _AMBIGUOUS_MARKERS)
_DESTINATION_RE = _marker_re(_DESTINATION_MARKERS)


@dataclass
class ParsedInstruction:
    """Target object of an instruction together with where it moves between.

    All fields except `raw` are `None` when the prompt does not say. `objects` collects
    the distinct object names the prompt mentions, in the order target, source,
    destination, and is filled in from those fields when left empty.
    """

    raw: str = ""
    verb: str | None = None
    target: str | None = None
    source: str | None = None
    destination: str | None = None
    objects: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.objects:
            for name in (self.target, self.source, self.destination):
                if name and name not in self.objects:
                    self.objects.append(name)

    def to_dict(self) -> dict[str, Any]:
        """Plain-`dict` view, the form written to the annotation JSON files."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedInstruction":
        """Rebuild an instruction from :meth:`to_dict` output.

        Missing keys fall back to their defaults, so records written before a field was
        added still load.
        """
        return cls(
            raw=str(data.get("raw") or ""),
            verb=data.get("verb"),
            target=data.get("target"),
            source=data.get("source"),
            destination=data.get("destination"),
            objects=list(data.get("objects") or []),
        )


def parse_instruction(text: str) -> ParsedInstruction:
    """Parse a single instruction prompt.

    Args:
        text: instruction as written in the dataset, e.g. `"Pick up the red block from
            the table and put it in the box."`.

    Returns:
        A :class:`ParsedInstruction`; `raw` is the prompt verbatim and the other fields
        are lowercased, free of determiners and trailing punctuation.
    """
    raw = "" if text is None else str(text)
    normalized = _normalize_text(raw)
    if not normalized:
        return ParsedInstruction(raw=raw)
    clauses = [clause.strip() for clause in _CLAUSE_RE.split(normalized) if clause.strip()]
    target: str | None = None
    source: str | None = None
    destination: str | None = None

    verb, body = _split_verb(clauses[0])
    normalized_body = _normalize_body(body)
    head, marker, tail = _find_marker(normalized_body)
    head_phrase = _clean_phrase(head)
    if head_phrase in _NON_OBJECT_HEADS:
        # "hold position": the clause names no object, so later clauses may provide it.
        head_phrase = None
    elif verb == "wait":
        wait_object = _wait_object(normalized_body)
        if wait_object is not None:
            head_phrase, marker, tail = wait_object, None, ""

    if marker is None:
        target = head_phrase
    elif marker in _DESTINATION_MARKERS:
        target = head_phrase
        destination = _clean_phrase(tail)
    elif marker in _SOURCE_MARKERS:
        target = head_phrase
        inner_head, inner_marker, inner_tail = _find_marker(tail, markers=_DESTINATION_MARKERS)
        if inner_marker is None:
            source = _clean_phrase(tail)
        else:
            source = _clean_phrase(inner_head)
            destination = _clean_phrase(inner_tail)
    else:
        target = head_phrase
        source = _clean_phrase(tail)

    for clause in clauses[1:]:
        clause_verb, clause_body = _split_verb(clause)
        normalized_clause = _normalize_body(clause_body)
        head, marker, tail = _find_marker(normalized_clause)
        head_phrase = _clean_phrase(head)
        if head_phrase in _NON_OBJECT_HEADS:
            head_phrase = None
        elif clause_verb == "wait":
            wait_object = _wait_object(normalized_clause)
            if wait_object is not None:
                head_phrase, marker, tail = wait_object, None, ""
        if target is None and (marker is not None or clause_verb is not None):
            target = head_phrase
        if marker is None:
            continue
        if clause_verb in PLACEMENT_VERBS:
            destination = destination or _clean_phrase(tail)
        elif clause_verb is None or marker not in _DESTINATION_MARKERS:
            source = source or _clean_phrase(tail)
        else:
            destination = destination or _clean_phrase(tail)

    return ParsedInstruction(
        raw=raw,
        verb=verb,
        target=target,
        source=source,
        destination=destination,
    )


def parse_instructions(texts: Iterable[str]) -> list[ParsedInstruction]:
    """Parse an iterable of prompts, one :class:`ParsedInstruction` per prompt."""
    return [parse_instruction(text) for text in texts]


def _normalize_text(text: str) -> str:
    """Lowercase a prompt, collapse its whitespace and drop a final period."""
    collapsed = re.sub(r"\s+", " ", str(text).strip().lower())
    return collapsed.rstrip(".!?").strip()


def _split_verb(clause: str) -> tuple[str | None, str]:
    """Split a clause into its first known verb and the body that follows it."""
    match = _VERB_RE.search(clause)
    if match is None:
        return None, clause
    return match.group(1), clause[match.end() :].strip()


def _normalize_body(body: str) -> str:
    """Strip directional adverbials, manner adverbs and stranded particles/pronouns."""
    text = body.strip()
    while True:
        stripped = text
        stripped = _DIRECTIONAL_RE.sub("", stripped)
        stripped = _TRAILING_MANNER_RE.sub("", stripped)
        stripped = _TRAILING_PARTICLE_RE.sub("", stripped)
        stripped = _LEADING_RE.sub("", stripped)
        stripped = stripped.strip()
        if stripped == text:
            return text
        text = stripped


def _find_marker(body: str, markers: frozenset[str] | None = None) -> tuple[str, str | None, str]:
    """Split `body` at its first spatial marker into `(head, marker, tail)`.

    Without a marker the whole body is the head and the marker is `None`. `markers`
    restricts the search to a subset of the known markers.
    """
    pattern = _MARKER_RE if markers is None else _marker_re(markers)
    match = pattern.search(body)
    if match is None:
        return body, None, ""
    return body[: match.start()].strip(), match.group(1), body[match.end() :].strip()


def _wait_object(body: str) -> str | None:
    """Object a `wait` clause watches, or `None` when the clause is not a wait one.

    "Wait until the plastic bottle stops moving in the scene." waits on the bottle, so
    the phrase between `until`/`for` and the stative predicate is the target; the rest
    of the clause describes the predicate and the scene, not a source or a destination.
    """
    match = _WAIT_RE.match(body)
    if match is None:
        return None
    return _clean_phrase(match.group("object"))


def _clean_phrase(phrase: str | None) -> str | None:
    """Tidy one object name: determiners, punctuation and whitespace removed."""
    if phrase is None:
        return None
    text = str(phrase).strip(_PUNCTUATION)
    while True:
        stripped = _DETERMINER_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"\s+", " ", text).strip(_PUNCTUATION)
    return text or None
