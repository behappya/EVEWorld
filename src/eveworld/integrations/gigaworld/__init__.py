"""GigaWorld-0 integration (DreamGen and AgiBot variants)."""

from .dataset import AgiBotDataset, DreamGenDataset, collate_fn
from .hooks import IGRCollator, build_igr_transform, mix_batches, register_tia
from .model import GigaWorldConfig, GigaWorldModel, load_backbone
from .trainer import GigaWorldTrainer

__all__ = [
    "AgiBotDataset",
    "DreamGenDataset",
    "GigaWorldConfig",
    "GigaWorldModel",
    "GigaWorldTrainer",
    "IGRCollator",
    "build_igr_transform",
    "collate_fn",
    "load_backbone",
    "mix_batches",
    "register_tia",
]
