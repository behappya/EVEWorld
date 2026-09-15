"""Paired full-training config: historical EVEWorld plus CIC-Transport only."""

from __future__ import annotations

import copy

from eveworld.pipeline.t4g_joint_config import config as historical_config


config = copy.deepcopy(historical_config)
config["runners"] = [
    "eveworld.tia_transport.cic_transport_trainer.CICTransportJointTrainer"
]
config["project_dir"] = (
    "/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1/"
    "cic_transport_seed42_s300/experiments"
)
config["models"].update(
    cic_transport_after_block="block22",
    cic_transport_rank=64,
    cic_transport_window_radius=3,
    cic_transport_temperature=0.07,
    cic_transport_residual_scale=0.1,
    cic_transport_init_seed=20260808,
)
config["train"].update(
    max_steps=300,
    checkpoint_interval=50,
    checkpoint_total_limit=8,
    seed=42,
)
