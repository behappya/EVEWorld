"""Instruction-following judges used in Table 1 and Table 6.

The two judges live in :mod:`~eveworld.evaluation.instruction_following.qwen_if`
and :mod:`~eveworld.evaluation.instruction_following.gemini_if`. They share the
prompt/verdict plumbing, so the helpers are re-exported here from ``qwen_if``
while each judge stays reachable under its own module name.
"""

from . import gemini_if, qwen_if
from .qwen_if import (
    DEFAULT_MODEL,
    PROMPT_PATH,
    QwenIFJudge,
    build_judge_prompt,
    parse_verdict,
    score_records,
    summarize,
)

__all__ = [
    "DEFAULT_MODEL",
    "PROMPT_PATH",
    "QwenIFJudge",
    "build_judge_prompt",
    "gemini_if",
    "parse_verdict",
    "qwen_if",
    "score_records",
    "summarize",
]
