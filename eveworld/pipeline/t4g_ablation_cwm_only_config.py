"""Strict CWM-only arm: contractual weight map without Copy-Paste or CIC."""

from .t4g_strict_ablation_common import strict_config, strict_joint_metadata
from .t4g_wmaponly_config import config as base_config


config = strict_config(base_config, "cwm_only")
config["dataloaders"]["train"]["transform"]["p_aug"] = 0.0
config["models"]["t4g_lambdas"] = "0.0,0.0"
strict_joint_metadata(config)
