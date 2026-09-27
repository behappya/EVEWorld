"""EVEWorld: instruction-grounded world modelling with interaction-guided repair
and temporally anchored injection.

The package is organised in four layers:

``eveworld.methods``
    The two contributions of the paper -- :mod:`~eveworld.methods.igr` and
    :mod:`~eveworld.methods.tia` -- together with the joint objective that
    trains them.
``eveworld.integrations``
    Thin adapters wiring those methods into the two video world models we build
    on, GigaWorld-0 and FlowWAM.
``eveworld.data``
    Instruction parsing, open-vocabulary grounding, instance tracking and the
    video/latent transforms shared by training, inference and evaluation.
``eveworld.evaluation``
    The Multi-Instance Localisation Rate (MLR) metric and wrappers for the
    external benchmarks (DreamGenBench, WorldArena, EWMBench, PBench, RoboTwin).

Submodules are resolved lazily, so ``import eveworld`` stays cheap and does not
drag in a deep-learning framework by itself.
"""

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "data",
    "evaluation",
    "integrations",
    "methods",
    "utils",
]

_SUBMODULES = frozenset(__all__) - {"__version__"}


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
