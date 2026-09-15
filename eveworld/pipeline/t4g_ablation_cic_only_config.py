"""Strict CIC-only arm: L_id enabled, with L_change, Copy-Paste, and CWM off."""

from .t4g_corr_config import config as base_config
from .t4g_strict_ablation_common import strict_config


config = strict_config(base_config, "cic_only")
config["models"]["t4g_lambdas"] = "0.5,0.0"
