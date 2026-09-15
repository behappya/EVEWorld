"""CIC-Transport training-seed replication extended to 400 steps."""

from __future__ import annotations

import copy

from eveworld.tia_transport.cic_transport_config import config as seed42_config


config = copy.deepcopy(seed42_config)
config["project_dir"] = (
    "/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1/"
    "cic_transport_seed6666_s400/experiments"
)
config["train"].update(
    max_steps=400,
    checkpoint_interval=50,
    checkpoint_total_limit=8,
    seed=6666,
)
