"""Configuration loading and saving with OmegaConf.

Every YAML under ``configs/`` follows the same top-level layout (``name``, ``seed``,
``output_dir``, ``method``, ``model``, ``train``, ``data``, ``inference``, ``eval``), and the
scripts load one of those files through :func:`load_config`, which layers the command-line
``--key value`` overrides on top and resolves the ``${...}`` interpolations some of the fields
use (``output_dir: outputs/${name}``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

__all__ = ["load_config", "save_config", "to_dict", "cfg_get", "resolve_path"]

_MISSING = object()


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load a YAML config and apply ``key=value`` overrides on top of it.

    Args:
        path: YAML file to load. ``None`` starts from an empty config, which is what the scripts
            use when a run is fully described by its command line.
        overrides: OmegaConf dot-list entries such as ``["train.max_steps=250", "seed=42"]``.
            Nested keys, list values (``model.resolution=[768,480]``) and interpolations that
            refer to other keys of the same override set are all accepted.

    Returns:
        The merged config as a :class:`~omegaconf.DictConfig` with every interpolation resolved.

    Raises:
        ValueError: If the file does not contain a mapping.
    """
    config = OmegaConf.create({})
    if path is not None:
        loaded = OmegaConf.load(os.fspath(path))
        if not isinstance(loaded, DictConfig):
            raise ValueError(f"config file must contain a mapping: {path}")
        config = loaded
    if overrides:
        override_config = OmegaConf.from_dotlist([str(item) for item in overrides])
        config = OmegaConf.merge(config, override_config)
    OmegaConf.resolve(config)
    return config


def save_config(cfg: Any, path: str | os.PathLike[str]) -> Path:
    """Write ``cfg`` to ``path`` as YAML, creating the parent directory if needed.

    Training and inference scripts store the effective config next to their checkpoints and
    outputs, so a run can be reproduced from its own directory.
    """
    target = Path(os.fspath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(cfg, (DictConfig, ListConfig)):
        cfg = OmegaConf.create(cfg)
    OmegaConf.save(cfg, target)
    return target


def to_dict(cfg: Any) -> dict:
    """Return ``cfg`` as a plain container with every interpolation resolved.

    Nested configs become nested ``dict``, lists become ``list``, and values become their python
    scalar type; the result can be serialised with :mod:`json`.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, Mapping):
        container = dict(cfg)
    else:
        raise TypeError(f"expected a config mapping, got {type(cfg).__name__}")
    if not isinstance(container, dict):
        raise TypeError(f"expected a config mapping, got {type(container).__name__}")
    return container


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a config, supporting dotted paths and never raising on a missing key.

    ``cfg_get(cfg, "train.lr", 1e-4)`` reads a nested value without walking the config by hand.
    Missing keys, missing intermediate nodes and interpolations that cannot be resolved all
    return ``default``. :class:`~omegaconf.DictConfig`, plain mappings and objects exposing the
    path as attributes are accepted.
    """
    if cfg is None or not key:
        return default
    if isinstance(cfg, (DictConfig, ListConfig)):
        try:
            value = OmegaConf.select(cfg, key, default=_MISSING)
        except (OmegaConfBaseException, KeyError, TypeError, ValueError):
            return default
        return default if value is _MISSING else value
    node: Any = cfg
    for part in str(key).split("."):
        if isinstance(node, Mapping):
            if part not in node:
                return default
            node = node[part]
        else:
            node = getattr(node, part, _MISSING)
            if node is _MISSING:
                return default
    return node


def resolve_path(
    value: str | os.PathLike[str],
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Expand ``~`` and ``$VAR`` / ``${VAR}`` references in a configured path.

    Configs point at the environment for everything that lives outside the checkout, e.g.
    ``data_root: ${ROBOTWIN_DATA_ROOT}``. A variable that is not set is left in place rather than
    silently removed, so the caller notices the unresolved name.

    Args:
        value: Path as written in the config.
        root: Directory that relative paths are resolved against, typically
            ``eveworld.utils.io.repo_root()``. When it is ``None`` a relative path is returned
            relative to the current working directory.

    Returns:
        The expanded path.
    """
    expanded = Path(os.path.expandvars(os.fspath(value))).expanduser()
    if root is None or expanded.is_absolute():
        return expanded
    return Path(os.path.expandvars(os.fspath(root))).expanduser() / expanded
