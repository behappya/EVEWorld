"""EVE-Frontier MVP configuration.

This is a train-only development configuration.  ``FRONTIER_TRAIN_INDICES``
can select the frozen train rows without copying the packed dataset.
"""

import copy
import os
from pathlib import Path

from eveworld.alternatives.physlatent.configs.gr1_physlatent_adapter import config as _base


config = copy.deepcopy(_base)
config["runners"] = ["eveworld.EveFrontierTrainer"]
config["launch"]["deepspeed_config"]["deepspeed_config_file"] = str(
    Path(__file__).with_name("deepspeed_zero2_clip.json").resolve()
)
config["project_dir"] = os.environ.get(
    "FRONTIER_PROJECT_DIR",
    "/data/datasets/gagi/giga_world_0_outputs/eve/frontier_mvp_dev",
)
config["models"]["train_mode"] = "lora"
config["models"]["lora_rank"] = int(os.environ.get("FRONTIER_LORA_RANK", "64"))
config["models"]["physics_latent"]["enabled"] = False
config["models"]["physics_latent"]["train_backbone"] = False
config["dataloaders"]["train"]["transform"]["random_crop"] = False
config["models"]["frontier"] = dict(
    block_size=int(os.environ.get("FRONTIER_BLOCK_SIZE", "4")),
    condition_latents=int(os.environ.get("FRONTIER_CONDITION_LATENTS", "1")),
    history_sigma=float(os.environ.get("FRONTIER_HISTORY_SIGMA", "0.0001")),
    sigma_data=1.0,
    use_flow=True,
    # Keep the MVP factorization-only by default.  Enabling this runs the
    # reference-centered anti-skip objective and requires LoRA mode so the
    # pretrained base can be evaluated with its adapter disabled.
    skip_weight=float(os.environ.get("FRONTIER_SKIP_WEIGHT", "0.0")),
    skip_margin=float(os.environ.get("FRONTIER_SKIP_MARGIN", "0.05")),
    skip_future_offset=int(os.environ.get("FRONTIER_SKIP_FUTURE_OFFSET", "1")),
    skip_negative_mode=os.environ.get("FRONTIER_SKIP_NEGATIVE_MODE", "adjacent"),
    skip_detach_future=os.environ.get("FRONTIER_SKIP_DETACH_FUTURE", "0") == "1",
    skip_diagnostic=os.environ.get("FRONTIER_SKIP_DIAGNOSTIC", "0") == "1",
)

indices = os.environ.get("FRONTIER_TRAIN_INDICES", "").strip()
if indices:
    selected = [int(value) for value in indices.split(",") if value.strip()]
    config["dataloaders"]["train"]["filter"] = dict(
        mode="func",
        func="eveworld.data_curation.filters.select_data_indices",
        indices=selected,
    )

config["train"]["max_steps"] = int(os.environ.get("FRONTIER_MAX_STEPS", "100"))
config["train"]["checkpoint_interval"] = int(os.environ.get("FRONTIER_CHECKPOINT_INTERVAL", "25"))
config["train"]["checkpoint_total_limit"] = -1
config["train"]["with_ema"] = False
config["train"]["resume"] = False
config["train"]["max_grad_norm"] = float(os.environ.get("FRONTIER_MAX_GRAD_NORM", "1.0"))
config["optimizers"]["lr"] = float(os.environ.get("FRONTIER_LR", str(config["optimizers"]["lr"])))
