from __future__ import annotations

import re
from dataclasses import dataclass


PICK_PLACE_RE = re.compile(
    r'use\s+the\s+(?P<hand>left|right)\s+hand\s+to\s+pick\s+up\s+'
    r'(?P<object>.+?)\s+from\s+(?P<source>.+?)\s+to\s+(?P<target>.+)$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedPickPlacePrompt:
    prompt: str
    hand: str | None
    object_name: str | None
    source: str | None
    target: str | None

    @property
    def matched(self) -> bool:
        return self.object_name is not None and self.source is not None and self.target is not None


def normalize_prompt(prompt: str) -> str:
    text = str(prompt or '').strip()
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.rstrip('.')


def parse_pick_place_prompt(prompt: str) -> ParsedPickPlacePrompt:
    text = normalize_prompt(prompt)
    match = PICK_PLACE_RE.search(text)
    if not match:
        return ParsedPickPlacePrompt(prompt=text, hand=None, object_name=None, source=None, target=None)
    return ParsedPickPlacePrompt(
        prompt=text,
        hand=match.group('hand').lower(),
        object_name=match.group('object').strip(),
        source=match.group('source').strip(),
        target=match.group('target').strip(),
    )


def hand_id(hand: str | None) -> int:
    if hand == 'left':
        return 1
    if hand == 'right':
        return 2
    return 0
