"""Paired full-training config with cross-frame transport inserted at block 23.

Twin of `cic_transport_config`, which keeps the historical block-22 insertion
and its bit-exact paired campaign.  The paper reports block 23 for both
GigaWorld-0 and AgiBot (offline layer-wise consistency probe, block 12 on
FlowWAM), so this module is the paper-side entry.

The correspondence hook and the transport insertion must agree; the trainer
refuses to start when `t4g_id_block` differs from `cic_transport_after_block`,
so both move together.  Everything else (rank 64, radius 3, tau 0.07,
gamma 0.1, init seed 20260808, 300 steps, seed 42) is inherited unchanged.
"""

from __future__ import annotations

import copy

from eveworld.tia_transport.cic_transport_config import config as _base


config = copy.deepcopy(_base)
config["models"].update(
    t4g_id_block="block23",
    cic_transport_after_block="block23",
)
config["project_dir"] = (
    "/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1/"
    "cic_transport_b23_seed42_s300/experiments"
)
