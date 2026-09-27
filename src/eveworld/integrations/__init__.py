"""Adapters that wire the EVEWorld methods into concrete video world models.

Each subpackage targets one backbone and carries everything backbone-specific:
the config dataclass, the checkpoint loader, the dataset that yields IGR-ready
samples, the trainer and the layer hooks used to attach TIA.
"""

_SUBMODULES = frozenset({"gigaworld", "flowwam"})

__all__ = ["flowwam", "gigaworld"]


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
