"""Training-time contributions: IGR, TIA and the joint objective that combines them."""

from .joint import JointConfig, JointObjective, combine

_SUBMODULES = frozenset({"igr", "tia"})

__all__ = ["JointConfig", "JointObjective", "combine", "igr", "tia"]


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
