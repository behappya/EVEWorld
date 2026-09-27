"""Evaluation: the MLR metric and the wrappers around the external benchmarks."""

_SUBMODULES = frozenset(
    {"mlr", "instruction_following", "robotwin", "pbench", "ewmbench", "worldarena"}
)

__all__ = [
    "ewmbench",
    "instruction_following",
    "mlr",
    "pbench",
    "robotwin",
    "worldarena",
]


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
