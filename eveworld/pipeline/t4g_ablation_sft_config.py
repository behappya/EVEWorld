"""Matched Standard SFT: no Copy-Paste, CWM, or CIC."""

from configs.giga_world_0_video_gr1_finetune import config as base_config

from .t4g_strict_ablation_common import strict_config


config = strict_config(base_config, "matched_sft")
