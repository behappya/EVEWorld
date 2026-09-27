"""Random-seed handling.

Every entry point seeds through :func:`set_seed` before it builds a model or a dataloader, and
passes :func:`worker_init_fn` to ``DataLoader(worker_init_fn=...)`` so that parallel workers draw
different crops without breaking reproducibility. The paper's configs run with seed 42, which is
also the fallback of :func:`seed_from_config`.
"""

from __future__ import annotations

import os
import random
from typing import Any, Mapping

import numpy as np

__all__ = ["set_seed", "worker_init_fn", "seed_from_config"]

# Seed set by every released config; used when a config does not name one.
_DEFAULT_SEED = 42
_MAX_SEED = 2**31 - 1


def _torch() -> Any:
    """Return the ``torch`` module when it is installed, otherwise ``None``."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed the python, numpy and torch random number generators and return ``seed``.

    ``PYTHONHASHSEED`` is exported as well so that child processes (dataloader workers, spawned
    launchers) inherit the same hash seed; the hash seed of the running interpreter is fixed when
    it starts and cannot be changed from here.

    torch is optional: workspaces that only prepare data run this helper without it installed,
    in which case only python and numpy are seeded.

    Args:
        seed: Seed applied to every generator.
        deterministic: Also put cuDNN and the CUDA kernels on their deterministic paths and pin
            ``CUBLAS_WORKSPACE_CONFIG``. Deterministic kernels are slower and are not used by the
            reported training runs; ops without a deterministic implementation only warn, so a
            long training job does not abort on them.

    Returns:
        The seed that was applied.
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    if torch is None:
        return seed
    torch.manual_seed(seed)
    # Every CUDA device, not just the current one: a job may hold several GPUs per process.
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Seed a ``DataLoader`` worker; pass it as ``worker_init_fn``.

    The base seed is the torch seed of the worker process, which ``DataLoader`` already offsets
    per worker. ``worker_id`` is mixed in again so that workers still differ when they are
    started from a shared base seed, and so that the helper stays meaningful without torch.

    Args:
        worker_id: Worker index handed over by ``DataLoader``.
    """
    torch = _torch()
    base = int(torch.initial_seed()) if torch is not None else _DEFAULT_SEED
    set_seed((base + int(worker_id)) % _MAX_SEED)


def seed_from_config(cfg: Any) -> int:
    """Seed from the top-level ``seed`` key of a run config and return the seed that was applied.

    Accepts a :class:`~omegaconf.DictConfig` or a plain mapping. A config that leaves the key out
    falls back to the released default of 42, so a seed is always set.
    """
    if cfg is None:
        seed = _DEFAULT_SEED
    elif isinstance(cfg, Mapping):
        seed = cfg.get("seed", _DEFAULT_SEED)
    else:
        seed = getattr(cfg, "seed", _DEFAULT_SEED)
    return set_seed(_DEFAULT_SEED if seed is None else int(seed))
