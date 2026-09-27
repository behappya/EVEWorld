"""Filesystem helpers for annotations, splits and result tables.

The metadata that drives training and evaluation is JSON: one object per clip for the split and
metadata files, one object per line for per-sample results and audit records. :func:`write_json`
and :func:`write_jsonl` are atomic, so an interrupted job leaves the previous file intact instead
of a truncated one, and they keep the insertion order of the keys so that records diff cleanly
against the ones committed under ``results/``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TextIO

__all__ = [
    "ensure_dir",
    "read_json",
    "write_json",
    "read_jsonl",
    "write_jsonl",
    "list_files",
    "repo_root",
    "is_run_checkpoint",
]

_PYPROJECT = "pyproject.toml"
_RUN_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".bin")
_RUN_CHECKPOINT_GLOB = "checkpoint-*.pt"


def repo_root() -> Path:
    """Directory holding ``pyproject.toml``, found by walking up from this file.

    The entry-point scripts use it to reach ``configs/``, ``data/`` and ``assets/`` no matter
    which directory they were started from. An installed copy of the package that sits outside a
    checkout has no ``pyproject.toml`` above it, and then the current working directory is used.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / _PYPROJECT).is_file():
            return parent
    return Path.cwd()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` and its parents if they do not exist and return it as a :class:`Path`."""
    directory = Path(os.fspath(path))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | os.PathLike[str]) -> Any:
    """Parse a JSON file; the document may be an object, a list or a scalar."""
    with open(os.fspath(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: str | os.PathLike[str], indent: int = 2) -> Path:
    """Write ``obj`` to ``path`` as JSON.

    Parent directories are created as needed and the keys keep their insertion order, since
    sorting them would make the files harder to compare across runs. The payload is written to a
    temporary file in the destination directory and moved into place with :func:`os.replace`, so
    a reader either sees the previous file or the complete new one.
    """
    def dump(handle: TextIO) -> None:
        json.dump(obj, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")

    return _atomic_write(path, dump)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    """Read a newline-delimited JSON file into a list of records, skipping blank lines."""
    rows: list[dict] = []
    with open(os.fspath(path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | os.PathLike[str]) -> Path:
    """Write ``rows`` to ``path`` as newline-delimited JSON, one record per line.

    Used for per-clip result tables and for the VLM audit logs, which are appended to over a
    long evaluation and therefore have to be readable after a crash: the write is atomic like
    :func:`write_json`.
    """
    def dump(handle: TextIO) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return _atomic_write(path, dump)


def list_files(root: str | os.PathLike[str], suffix: str | None = None) -> list[Path]:
    """List every file below ``root``, walking subdirectories, sorted by path.

    The walk is recursive, so one call collects the episodes of every task below a dataset root
    without a nested loop. Sorting is what makes the result stable across filesystems and across
    runs.

    Args:
        root: Directory to walk; a missing directory yields an empty list.
        suffix: Optional filename ending, matched with :meth:`str.endswith`. Pass the extension
            together with its dot (``".mp4"``) to select a single media type.

    Returns:
        The matching paths, in sorted order.
    """
    root_path = Path(os.fspath(root))
    if not root_path.is_dir():
        return []
    files = [entry for entry in root_path.rglob("*") if entry.is_file()]
    if suffix is not None:
        files = [entry for entry in files if entry.name.endswith(suffix)]
    return sorted(files)


def is_run_checkpoint(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` names a checkpoint written by a training run.

    A training run writes ``checkpoint-<step>.pt`` files and a ``latest.json`` pointing at the
    most recent one, so either a file or a directory can be named to generation. A released
    backbone is a directory of its own without those names, which is how the two are told apart
    before the weight loader runs.

    Args:
        path: Value of `model.checkpoint`: a release name, a directory, or a path.

    Returns:
        ``True`` for the checkpoint of a run or for a directory holding one, ``False`` for a
        released backbone name or directory.
    """
    candidate = Path(os.fspath(path)).expanduser()
    if candidate.is_file():
        return candidate.suffix in _RUN_CHECKPOINT_SUFFIXES
    if candidate.is_dir():
        return (candidate / "latest.json").is_file() or any(
            candidate.glob(_RUN_CHECKPOINT_GLOB)
        )
    return False


def _atomic_write(path: str | os.PathLike[str], write: Callable[[TextIO], None]) -> Path:
    """Run ``write`` against a temporary file in the destination directory, then move it onto ``path``."""
    target = Path(os.fspath(path))
    ensure_dir(target.parent)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target
