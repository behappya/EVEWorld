"""Reproducible A-pre configuration for the six-point CFG sweep.

This is the legacy A-pre recipe with augmentation enabled, trained directly from
the pretraining transformer at seed 42.  The runner overrides output/checkpoint
paths through the generic training launcher, while this module keeps the method
recipe explicit and reviewable.
"""

from .t4g_apre_noaug_config import config as _base
import copy

config = copy.deepcopy(_base)
config["dataloaders"]["train"]["transform"]["p_aug"] = 0.5
config["project_dir"] = \
    "/data/datasets/gagi/eve_v2_outputs/t4g_cfg_repro_seed42_s300/experiments"
config["train"]["seed"] = 42
config["train"]["max_steps"] = 300
config["train"]["checkpoint_interval"] = 50
config["train"]["checkpoint_total_limit"] = 8

