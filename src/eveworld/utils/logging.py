"""Logging for the library and the entry-point scripts.

Library code never prints: every module obtains its logger once, at import time, with

    from eveworld.utils.logging import get_logger

    logger = get_logger(__name__)

Messages are rendered through ``rich`` when it is installed and through a plain stream handler
otherwise, so the same code runs where rich is unavailable. The handler is installed on the root
logger exactly once, which keeps the ``get_logger`` call in every imported module from
duplicating each line.
"""

from __future__ import annotations

import logging

__all__ = ["get_logger", "setup_logging"]

# A rich handler draws the timestamp and the level itself, so its formatter only supplies the
# message; the plain handler has to put them in the line.
_RICH_FORMAT = "%(rank_prefix)s%(message)s"
_STREAM_FORMAT = "%(rank_prefix)s[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_DEFAULT_LEVEL = logging.INFO

_handler: logging.Handler | None = None
_rank_filter: "_RankFilter | None" = None


class _RankFilter(logging.Filter):
    """Prefix every record with ``[rank N]`` for multi-process runs.

    The filter is always attached, with an empty prefix for rank 0 and for unranked runs, so
    that the ``rank_prefix`` field referenced by both format strings is always present.
    """

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self._prefix = f"[rank {self.rank}] " if self.rank > 0 else ""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank_prefix = self._prefix
        return True


def _coerce_level(level: int | str) -> int:
    """Translate a level name such as ``"info"`` into its numeric value."""
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).strip().upper())
    if not isinstance(value, int):
        raise ValueError(f"unknown log level: {level!r}")
    return value


def _make_formatter(fmt: str) -> logging.Formatter:
    """Build the formatter shared by both handlers.

    ``rank_prefix`` is filled in by :class:`_RankFilter`; the empty default keeps a record that
    reaches a formatter without passing through the filter renderable rather than raising.
    """
    return logging.Formatter(fmt, datefmt=_DATE_FORMAT, defaults={"rank_prefix": ""})


def _build_handler() -> logging.Handler:
    """Create the shared handler: a ``rich`` handler when available, a stream handler otherwise."""
    try:
        from rich.logging import RichHandler
    except ImportError:
        handler: logging.Handler = logging.StreamHandler()
        formatter = _make_formatter(_STREAM_FORMAT)
    else:
        handler = RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_path=False,
            omit_repeated_times=False,
        )
        formatter = _make_formatter(_RICH_FORMAT)
    handler.setFormatter(formatter)
    return handler


def _ensure_root_handler() -> logging.Handler:
    """Install the shared handler on the root logger on first use and return it."""
    global _handler, _rank_filter
    if _handler is not None:
        return _handler
    handler = _build_handler()
    _rank_filter = _RankFilter(0)
    handler.addFilter(_rank_filter)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET:
        root.setLevel(_DEFAULT_LEVEL)
    _handler = handler
    return handler


def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """Return the logger called ``name``, configuring the shared root handler on first use.

    ``name`` is conventionally ``__name__`` of the calling module. ``level`` sets the level of
    this logger only; when it is omitted the logger inherits the root level configured by
    :func:`setup_logging`.
    """
    _ensure_root_handler()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(_coerce_level(level))
    return logger


def setup_logging(level: int | str = "INFO", rank: int | None = None) -> None:
    """Configure the root logger for a run and tag records with the distributed rank.

    Calling this more than once (or after any number of :func:`get_logger` calls) reuses the
    handler installed by the first call, so no message is emitted twice.

    Args:
        level: Root level, as a :mod:`logging` level name or a numeric level.
        rank: Rank of this process in a multi-process job. Ranks above zero get a ``[rank N]``
            prefix so that the interleaved logs of a distributed run can be attributed; the
            main process and unranked runs stay unlabelled.
    """
    global _rank_filter
    handler = _ensure_root_handler()
    logging.getLogger().setLevel(_coerce_level(level))
    handler.removeFilter(_rank_filter)
    _rank_filter = _RankFilter(rank or 0)
    handler.addFilter(_rank_filter)
