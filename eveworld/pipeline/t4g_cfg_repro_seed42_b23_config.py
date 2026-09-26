"""Reproducible A-pre configuration with the correspondence layer at block 23.

Twin of `t4g_cfg_repro_seed42_config`, which keeps the historical block-22
recipe and the historical output directory.  Both entries drive the same
six-point CFG sweep; this one matches the block-23 layer selected by the
offline consistency probe (block 23 for GigaWorld-0/DreamGen and AgiBot,
block 12 for FlowWAM).
"""

from .t4g_cfg_repro_seed42_config import config as _base
import copy

config = copy.deepcopy(_base)
config["models"]["t4g_id_block"] = "block23"
config["project_dir"] = \
    "/data/datasets/gagi/eve_v2_outputs/t4g_cfg_repro_seed42_b23_s300/experiments"
