"""A-pre-noaug with the correspondence layer at block 23 instead of block 22.

Twin of `t4g_apre_noaug_config`, which stays untouched as the historical
block-22 recipe.  The offline layer-wise consistency probe in the paper
sweeps `block8`--`block26` and selects block 23 for GigaWorld-0/DreamGen and
AgiBot, so this module is the paper-side entry for the same single-variable
A-pre-noaug ablation.  Everything else (p_aug=0, five-level weightmap,
w_paste=4.0, seed 6666) is inherited unchanged.
"""

from .t4g_apre_noaug_config import config as _base
import copy

config = copy.deepcopy(_base)
config["models"]["t4g_id_block"] = "block23"
config["project_dir"] = \
    "/data/datasets/gagi/eve_v2_outputs/t4g_joint_wmapA_pre_noaug_b23/experiments"
