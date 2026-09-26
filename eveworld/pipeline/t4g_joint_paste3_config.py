"""Joint training with a 3x paste-region weight instead of the historical 4x.

Twin of `t4g_joint_config`, which keeps `t4g_w_paste=4.0` and its historical
output directory.  The paper reports binary region weighting (1x for ordinary
tokens, 3x for restoration-related and pasted regions), so this module is the
paper-side entry for the same joint recipe.  Everything else (p_aug=0.5,
five-level weightmap fallback, L_id 0.5, 200 steps) is inherited unchanged.
"""

from .t4g_joint_config import config as _base
import copy

config = copy.deepcopy(_base)
config["models"]["t4g_w_paste"] = 3.0
config["project_dir"] = \
    "/data/datasets/gagi/eve_v2_outputs/t4g_joint_w_paste3/experiments"
