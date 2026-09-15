"""Physics-aware latent adapter experiments for GigaWorld-0.

The package avoids importing the baseline ``giga_world_0`` trainer at module
import time. Some local evaluation environments have diffusers/transformers
version skew, so the heavyweight trainer is loaded lazily only when requested.
"""

from .modules import PhysicsLatentEncoder
from .transforms import PhysLatentGigaWorld0Transform

__all__ = [
    'PhysLatentGigaWorld0Transform',
    'PhysicsLatentEncoder',
    'PhysicsLatentGigaWorld0Trainer',
]


def __getattr__(name):
    if name == 'PhysicsLatentGigaWorld0Trainer':
        from .trainer import PhysicsLatentGigaWorld0Trainer

        return PhysicsLatentGigaWorld0Trainer
    raise AttributeError(name)
