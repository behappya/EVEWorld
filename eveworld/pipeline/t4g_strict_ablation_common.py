"""Shared invariants for the publication-grade EVEWorld ablation matrix."""

from __future__ import annotations

import copy


GAGI = "/data/datasets/gagi"
PRETRAIN = f"{GAGI}/giga_world_0_video_pretrain"
PACKED = f"{GAGI}/gr1_finetune_data/packed_data"
PROBE = f"{GAGI}/eve_v2_outputs/track4gen_probe"
ANNO_DIR = f"{PROBE}/t4g_anno"
ASSETS_DIR = f"{PROBE}/aug_assets"
WMAP_DIR = f"{PROBE}/weightmap_cache"
OUTPUT_ROOT = f"{GAGI}/eve_v2_outputs/eve_ablation_strict_v1"

EXPECTED_SAMPLES = 92
TRAINING_SEED = 42
MAX_STEPS = 250
CHECKPOINT_INTERVAL = 50
CHECKPOINT_TOTAL_LIMIT = 5


def strict_config(base_config: dict, variant: str) -> dict:
    """Apply invariants shared by every strict ablation arm."""
    config = copy.deepcopy(base_config)
    config["project_dir"] = f"{OUTPUT_ROOT}/{variant}_seed42_s250/experiments"

    train_loader = config["dataloaders"]["train"]
    train_loader["data_or_config"] = [PACKED]
    train_loader["batch_size_per_gpu"] = 1
    train_loader["num_workers"] = 6

    transform = train_loader["transform"]
    transform.update(num_frames=93, height=480, width=768, fps=16)

    models = config["models"]
    models["transformer_model_path"] = f"{PRETRAIN}/transformer"
    models["vae_model_path"] = f"{PRETRAIN}/vae"

    train = config["train"]
    train.update(
        resume=True,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=8,
        mixed_precision="bf16",
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_total_limit=CHECKPOINT_TOTAL_LIMIT,
        checkpoint_start_step=0,
        checkpoint_strict=False,
        with_ema=True,
        activation_checkpointing=True,
        seed=TRAINING_SEED,
    )

    if transform.get("type") == "T4GAugTransform":
        transform.update(
            idx2vid_path=f"{ANNO_DIR}/_packidx2vid.json",
            anno_dir=ANNO_DIR,
            assets_dir=ASSETS_DIR,
            wmap_dir=WMAP_DIR,
            wmap_values="0.5,2.0,3.0,4.0,6.0",
            strict_mapping=True,
            strict_assets=True,
            seed=20260721,
            p_corridor=0.0,
            p_alien=0.0,
        )
    return config


def strict_joint_metadata(config: dict) -> None:
    """Make the CWM assets and expected sample count explicit in runtime JSON."""
    config["models"].update(
        t4g_anno_dir=ANNO_DIR,
        t4g_idx2vid=f"{ANNO_DIR}/_packidx2vid.json",
        t4g_expected_samples=EXPECTED_SAMPLES,
        t4g_wmap_dir=WMAP_DIR,
        t4g_region_levels="0.5,2.0,3.0,4.0,6.0",
        t4g_region_names="bg0.5x,empty2x,gripB3x,obj4x,trans6x",
    )
    config["dataloaders"]["train"]["transform"]["strict_wmap"] = True
