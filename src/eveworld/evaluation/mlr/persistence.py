"""Persistence filter for MLR deviations.

A single deviating timestamp is noise, so a deviation only becomes an event once it persists
for ``k`` consecutive sampled timestamps (``k = 2`` in the released protocol). An exemption --
an instance that is plausibly hidden behind the robot arm -- resets the streak instead of
extending it, so an under-count that occlusion already explains cannot grow into an event.

Array shapes: deviations, exemptions, events and streaks are all ``(T,)``, one entry per
sampled timestamp; ``events`` is boolean and ``streak[t]`` is the number of consecutive
effective deviations ending at ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["PersistenceResult", "PersistenceTracker", "detect_events"]


@dataclass
class PersistenceResult:
    """Persistence trace of one clip.

    Attributes:
        events: Boolean array ``(T,)``; ``True`` from the first persistent timestamp onward.
        deviations: Boolean array ``(T,)`` of effective deviations (an exempted or absent
            deviation is ``False``).
        streak: Integer array ``(T,)``; consecutive effective deviations ending at ``t``.
    """

    events: np.ndarray
    deviations: np.ndarray
    streak: np.ndarray


def _check_k(k: Any) -> int:
    """Validate the required run length."""
    value = int(k)
    if value < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return value


def _effective(deviations: Any, exemptions: Any) -> np.ndarray:
    """Boolean ``(T,)`` effective deviations: an exemption cancels a deviation."""
    flags = np.asarray(deviations, dtype=bool)
    if flags.ndim != 1:
        raise ValueError(f"deviations must have shape (T,), got {flags.shape}")
    if exemptions is None:
        return flags
    skip = np.asarray(exemptions, dtype=bool)
    if skip.shape != flags.shape:
        raise ValueError(f"exemptions {skip.shape} must match deviations {flags.shape}")
    return flags & ~skip


def _streaks(active: np.ndarray) -> np.ndarray:
    """Lengths of the deviations runs ending at each position of ``active`` ``(T,)``."""
    if active.size == 0:
        return np.zeros(0, dtype=int)
    index = np.arange(active.size)
    last_reset = np.maximum.accumulate(np.where(active, -1, index))
    return index - last_reset


def detect_events(
    deviations: Sequence[Any] | np.ndarray, exemptions: Sequence[Any] | np.ndarray | None = None,
    *, k: int = 2,
) -> np.ndarray:
    """Vectorised persistence filter.

    Args:
        deviations: Boolean array ``(T,)``; ``True`` where the instance count deviates from the
            expected count at that sampled timestamp.
        exemptions: Optional boolean array ``(T,)``; ``True`` where the deviation at that
            timestamp is explained by occlusion and therefore breaks the run.
        k: Number of consecutive deviations required for an event.

    Returns:
        Boolean array ``(T,)``; ``True`` from the first timestamp whose run reaches ``k`` until
        the run is broken, matching :meth:`PersistenceTracker.update` element by element.
    """
    run_length = _check_k(k)
    active = _effective(deviations, exemptions)
    return _streaks(active) >= run_length


class PersistenceTracker:
    """Online persistence filter with the same rule as :func:`detect_events`.

    Args:
        k: Number of consecutive deviations required for an event; ``2`` follows the released
            protocol.
    """

    def __init__(self, k: int = 2) -> None:
        self.k = _check_k(k)
        self._streak = 0
        self._deviations: list[bool] = []
        self._events: list[bool] = []
        self._streaks: list[int] = []

    @property
    def streak(self) -> int:
        """Number of consecutive effective deviations seen so far."""
        return self._streak

    def update(self, deviation: bool, exempt: bool = False) -> bool:
        """Feed one sampled timestamp.

        Args:
            deviation: Whether the timestamp deviates from the expected count.
            exempt: Whether the deviation is explained by occlusion; an exemption resets the
                streak, as does a timestamp without a deviation.

        Returns:
            ``True`` once the current streak has reached ``k``, i.e. from the first persistent
            timestamp of the run onward.
        """
        if deviation and not exempt:
            self._streak += 1
        else:
            self._streak = 0
        event = self._streak >= self.k
        self._deviations.append(bool(deviation and not exempt))
        self._events.append(event)
        self._streaks.append(self._streak)
        return event

    def reset(self) -> None:
        """Clear the streak and the recorded trace."""
        self._streak = 0
        self._deviations.clear()
        self._events.clear()
        self._streaks.clear()

    def run(
        self,
        deviations: Sequence[Any] | np.ndarray,
        exemptions: Sequence[Any] | np.ndarray | None = None,
    ) -> PersistenceResult:
        """Feed a whole clip after clearing the tracker and return its trace.

        Args:
            deviations: Boolean array ``(T,)`` of per-timestamp deviations.
            exemptions: Optional boolean array ``(T,)`` of occlusion exemptions.

        Returns:
            :class:`PersistenceResult` with the ``(T,)`` event, deviation and streak arrays.
        """
        active = _effective(deviations, exemptions)
        skip = (
            np.zeros(active.shape, dtype=bool)
            if exemptions is None
            else np.asarray(exemptions, dtype=bool)
        )
        self.reset()
        for position, value in enumerate(np.asarray(deviations, dtype=bool)):
            self.update(bool(value), exempt=bool(skip[position]))
        return self.result()

    def result(self) -> PersistenceResult:
        """Trace of the timestamps fed so far, as ``(T,)`` arrays (empty before the first update)."""
        return PersistenceResult(
            events=np.asarray(self._events, dtype=bool),
            deviations=np.asarray(self._deviations, dtype=bool),
            streak=np.asarray(self._streaks, dtype=int),
        )
