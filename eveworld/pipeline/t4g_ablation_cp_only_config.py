"""Strict CP-only arm: Copy-Paste correction with uniform diffusion loss."""

from .t4g_aug_config import config as base_config
from .t4g_strict_ablation_common import strict_config


config = strict_config(base_config, "cp_only")
config["dataloaders"]["train"]["transform"].update(p_aug=0.5, strict_wmap=False)
