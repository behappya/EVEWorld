"""Helpers for the multi-process training, inference and evaluation jobs.

Rank and world size come from the environment the launchers set (``RANK``, ``LOCAL_RANK``,
``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``); once a process group is initialised the values
reported by ``torch.distributed`` take precedence. torch is imported lazily, so the data
preparation entry points - which run single-process and may not have torch installed - can use
:func:`shard_range` and the other accessors as they are.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "init_distributed",
    "all_reduce_mean",
    "shard_range",
]

# Single-node rendezvous defaults for a job whose launcher did not export a master address.
_MASTER_ADDR = "127.0.0.1"
_MASTER_PORT = 29500


def _torch() -> Any:
    """Return the ``torch`` module when it is installed, otherwise ``None``."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _dist() -> Any:
    """Return the initialised ``torch.distributed`` module, or ``None`` outside a process group."""
    torch = _torch()
    if torch is None:
        return None
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return None
    return distributed


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default`` when it is unset."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def is_distributed() -> bool:
    """Whether the job runs across more than one process (``WORLD_SIZE > 1``)."""
    return get_world_size() > 1


def get_rank() -> int:
    """Global rank of this process: the process group when initialised, otherwise ``RANK``."""
    distributed = _dist()
    if distributed is not None:
        return int(distributed.get_rank())
    return _env_int("RANK", 0)


def get_world_size() -> int:
    """Number of processes in the job: the process group when initialised, otherwise ``WORLD_SIZE``."""
    distributed = _dist()
    if distributed is not None:
        return int(distributed.get_world_size())
    return _env_int("WORLD_SIZE", 1)


def is_main_process() -> bool:
    """Whether this is rank 0, the process that writes checkpoints, logs and result tables."""
    return get_rank() == 0


def barrier() -> None:
    """Wait until every process reaches this point; a no-op outside a process group."""
    distributed = _dist()
    if distributed is not None:
        distributed.barrier()


def init_distributed(backend: str | None = None) -> None:
    """Initialise the default process group for a multi-process job.

    Safe to call unconditionally at the top of an entry point: it returns immediately when
    ``WORLD_SIZE`` is 1, when torch is not installed, or when a process group already exists.
    The rendezvous is the ``env://`` one, so the launcher only has to export ``MASTER_ADDR`` and
    ``MASTER_PORT``; if it did not, the single-node defaults are filled in.

    Args:
        backend: Torch backend name, e.g. ``"nccl"`` or ``"gloo"``. Defaults to ``"nccl"`` when
            CUDA is available and ``"gloo"`` otherwise.
    """
    torch = _torch()
    if torch is None or get_world_size() <= 1:
        return
    distributed = torch.distributed
    if not distributed.is_available() or distributed.is_initialized():
        return
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", _MASTER_ADDR)
    os.environ.setdefault("MASTER_PORT", str(_MASTER_PORT))
    if backend == "nccl" and torch.cuda.is_available():
        # Bind the process to its node-local device before the group is built, so the first
        # collective does not land on device 0 of every rank.
        torch.cuda.set_device(_env_int("LOCAL_RANK", 0))
    distributed.init_process_group(backend=backend, init_method="env://")


def all_reduce_mean(value: float) -> float:
    """Average ``value`` across the ranks and return the result on every rank.

    Used for the metrics that are computed per shard and reported once for the whole job. Outside
    a process group the input is returned unchanged, so metric code can call it unconditionally.
    The reduction runs on the current CUDA device under ``nccl`` and on the CPU otherwise.
    """
    distributed = _dist()
    torch = _torch()
    if distributed is None or torch is None:
        return float(value)
    if distributed.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
    tensor /= distributed.get_world_size()
    return float(tensor.item())


def shard_range(
    size: int,
    rank: int | None = None,
    world_size: int | None = None,
) -> tuple[int, int]:
    """Split ``size`` items over the ranks as contiguous half-open ``(start, stop)`` ranges.

    ``rank`` and ``world_size`` default to the current distributed state, so a dataset shards
    itself with ``start, stop = shard_range(len(clips))`` and gets the full range in a
    single-process run. The remainder goes to the first shards, which keeps the shards within one
    item of each other and covers every item exactly once: 10 items over 3 ranks give ``(0, 4)``,
    ``(4, 7)`` and ``(7, 10)``.

    Args:
        size: Number of items to split, e.g. the length of a clip list.
        rank: Rank to compute the range for. Defaults to :func:`get_rank`.
        world_size: Number of shards. Defaults to :func:`get_world_size`.

    Returns:
        ``(start, stop)``, empty when ``size`` is smaller than the number of shards and ``rank``
        is one of the trailing shards.

    Raises:
        ValueError: If ``size`` is negative, ``world_size`` is below 1, or ``rank`` lies outside
            ``[0, world_size)``.
    """
    world = get_world_size() if world_size is None else int(world_size)
    index = get_rank() if rank is None else int(rank)
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    if world < 1:
        raise ValueError(f"world_size must be at least 1, got {world}")
    if not 0 <= index < world:
        raise ValueError(f"rank must be in [0, {world}), got {index}")
    base, remainder = divmod(int(size), world)
    start = index * base + min(index, remainder)
    return start, start + base + (1 if index < remainder else 0)
