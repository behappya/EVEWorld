#!/usr/bin/env python3
"""Validate strict EVEWorld ablation configs and their shared assets."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from eveworld.pipeline.t4g_strict_ablation_common import (
    ANNO_DIR,
    ASSETS_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_TOTAL_LIMIT,
    EXPECTED_SAMPLES,
    MAX_STEPS,
    PACKED,
    PRETRAIN,
    TRAINING_SEED,
    WMAP_DIR,
)


MODULES = {
    "matched_sft": "eveworld.pipeline.t4g_ablation_sft_config",
    "cp_only": "eveworld.pipeline.t4g_ablation_cp_only_config",
    "cwm_only": "eveworld.pipeline.t4g_ablation_cwm_only_config",
    "cic_only": "eveworld.pipeline.t4g_ablation_cic_only_config",
    "igt": "eveworld.pipeline.t4g_ablation_igt_config",
}

EXPECTED = {
    "matched_sft": ("giga_world_0.GigaWorld0Trainer", "GigaWorld0Transform", None, None),
    "cp_only": ("eveworld.pipeline.t4g_aug_trainer.T4GAugTrainer", "T4GAugTransform", 0.5, None),
    "cwm_only": ("eveworld.pipeline.t4g_joint_trainer.T4GJointTrainer", "T4GAugTransform", 0.0, "0.0,0.0"),
    "cic_only": ("eveworld.pipeline.t4g_corr_trainer.T4GCorrTrainer", "T4GCorrTransform", None, "0.5,0.0"),
    "igt": ("eveworld.pipeline.t4g_joint_trainer.T4GJointTrainer", "T4GAugTransform", 0.5, "0.0,0.0"),
}


def data_size(root: Path) -> int:
    return int(json.loads((root / "videos" / "config.json").read_text())["data_size"])


def stems(root: Path, suffix: str) -> set[str]:
    return {path.stem for path in root.glob(f"*{suffix}") if not path.name.startswith("_")}


def main() -> None:
    packed = Path(PACKED)
    assert data_size(packed) == EXPECTED_SAMPLES
    assert (Path(PRETRAIN) / "transformer" / "diffusion_pytorch_model.safetensors").is_file()

    mapping = json.loads((Path(ANNO_DIR) / "_packidx2vid.json").read_text())
    mapped = {str(value) for value in mapping.values()}
    assert len(mapping) == len(mapped) == EXPECTED_SAMPLES
    assert stems(Path(ANNO_DIR), ".json") == mapped
    assert stems(Path(ASSETS_DIR), ".npz") == mapped
    assert stems(Path(WMAP_DIR), ".npy") == mapped
    allowed_weights = {0.5, 2.0, 3.0, 4.0, 6.0}
    for path in Path(WMAP_DIR).glob("*.npy"):
        weight_map = np.load(path, mmap_mode="r")
        assert weight_map.shape == (24, 30, 48), (path, weight_map.shape)
        assert set(float(value) for value in np.unique(weight_map)) <= allowed_weights

    report = {}
    for variant, module_name in MODULES.items():
        config = importlib.import_module(module_name).config
        loader = config["dataloaders"]["train"]
        transform = loader["transform"]
        models = config["models"]
        train = config["train"]
        runner, transform_type, p_aug, lambdas = EXPECTED[variant]
        assert config["runners"] == [runner]
        assert transform["type"] == transform_type
        assert loader["data_or_config"] == [PACKED]
        assert loader["batch_size_per_gpu"] == 1
        assert train["gradient_accumulation_steps"] == 8
        assert train["max_steps"] == MAX_STEPS
        assert train["seed"] == TRAINING_SEED
        assert train["checkpoint_interval"] == CHECKPOINT_INTERVAL
        assert train["checkpoint_total_limit"] == CHECKPOINT_TOTAL_LIMIT
        assert models["transformer_model_path"] == f"{PRETRAIN}/transformer"
        if p_aug is not None:
            assert float(transform["p_aug"]) == p_aug
            assert transform["strict_mapping"] is True
            assert transform["strict_assets"] is True
        if lambdas is not None:
            assert models["t4g_lambdas"] == lambdas
        report[variant] = {
            "runner": runner,
            "transform": transform_type,
            "p_aug": p_aug,
            "t4g_lambdas": lambdas,
            "project_dir": config["project_dir"],
        }

    print(json.dumps({"ready": True, "assets": len(mapped), "variants": report}, indent=2))


if __name__ == "__main__":
    main()
