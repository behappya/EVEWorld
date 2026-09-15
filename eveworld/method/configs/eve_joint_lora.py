"""Parameter-matched joint-denoising LoRA control for held-out EVE runs."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from configs.giga_world_0_video_gr1_finetune import config as _base


config = copy.deepcopy(_base)
config["launch"]["deepspeed_config"]["deepspeed_config_file"] = str(
    Path(__file__).with_name("deepspeed_zero2_clip.json").resolve()
)
config["project_dir"] = os.environ.get(
    "TRAIN_PROJECT_DIR",
    "/data/datasets/gagi/giga_world_0_outputs/eve/heldout_joint_lora",
)
config["models"]["train_mode"] = "lora"
config["models"]["lora_rank"] = int(os.environ.get("FRONTIER_LORA_RANK", "64"))
config["optimizers"]["lr"] = float(
    os.environ.get("FRONTIER_LR", str(config["optimizers"]["lr"]))
)

indices = os.environ.get("HELDOUT_TRAIN_INDICES", "").strip()
if not indices:
    raise ValueError("HELDOUT_TRAIN_INDICES is required for leakage-safe joint LoRA training")
config["dataloaders"]["train"]["filter"] = dict(
    mode="func",
    func="eveworld.data_curation.filters.select_data_indices",
    indices=[int(value) for value in indices.split(",") if value.strip()],
)
config["train"]["max_steps"] = int(os.environ.get("FRONTIER_MAX_STEPS", "450"))
config["train"]["checkpoint_interval"] = int(
    os.environ.get("FRONTIER_CHECKPOINT_INTERVAL", "150")
)
config["train"]["checkpoint_total_limit"] = -1
config["train"]["gradient_accumulation_steps"] = 1
config["train"]["with_ema"] = False
config["train"]["resume"] = False
config["train"]["max_grad_norm"] = float(os.environ.get("FRONTIER_MAX_GRAD_NORM", "1.0"))
