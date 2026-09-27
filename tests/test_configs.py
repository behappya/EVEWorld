"""Structural checks for the YAML under ``configs/``.

Every config is consumed through the same loader, so a typo in a top-level key
would only surface once a run had already started. These tests fail fast
instead: every file has to parse, and the blocks a given family of runs depends
on have to be present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

ALL_CONFIGS = sorted(CONFIG_ROOT.rglob("*.yaml"))
PAPER_CONFIGS = sorted((CONFIG_ROOT / "paper").rglob("*.yaml"))
ABLATION_CONFIGS = sorted((CONFIG_ROOT / "ablations").rglob("*.yaml"))
EVAL_CONFIGS = sorted((CONFIG_ROOT / "eval").rglob("*.yaml"))


def config_id(path: Path) -> str:
    return str(path.relative_to(CONFIG_ROOT))


def load_config_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_the_config_tree_is_not_empty():
    assert ALL_CONFIGS, f"no YAML found under {CONFIG_ROOT}"
    assert PAPER_CONFIGS, "the paper configs are missing"
    assert EVAL_CONFIGS, "the evaluation configs are missing"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=config_id)
def test_every_config_parses_and_names_the_run(path):
    cfg = load_config_file(path)
    assert isinstance(cfg, dict), f"{config_id(path)} is not a mapping"
    assert cfg.get("name"), f"{config_id(path)} does not name the run"
    assert cfg.get("output_dir"), f"{config_id(path)} does not set output_dir"


@pytest.mark.parametrize("path", PAPER_CONFIGS + ABLATION_CONFIGS, ids=config_id)
def test_training_configs_declare_the_full_stack(path):
    cfg = load_config_file(path)
    for block in ("method", "model", "train", "data", "inference"):
        assert block in cfg, f"{config_id(path)} has no '{block}' block"
    assert cfg["model"]["backbone"] in {"gigaworld0", "flowwam"}
    assert cfg["train"]["optimizer"] in {"came8bit", "adamw"}
    assert cfg["train"]["max_steps"] > 0


@pytest.mark.parametrize("path", ABLATION_CONFIGS, ids=config_id)
def test_ablation_configs_name_their_axis(path):
    cfg = load_config_file(path)
    assert "ablation" in cfg, f"{config_id(path)} does not name the ablation axis"


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=config_id)
def test_eval_configs_declare_metrics_and_data(path):
    cfg = load_config_file(path)
    assert cfg.get("data"), f"{config_id(path)} does not point at a dataset"
    assert cfg.get("eval", {}).get("metrics"), f"{config_id(path)} lists no metrics"


@pytest.mark.parametrize("path", PAPER_CONFIGS + ABLATION_CONFIGS, ids=config_id)
def test_seeds_are_fixed_in_training_configs(path):
    cfg = load_config_file(path)
    assert isinstance(cfg.get("seed"), int), f"{config_id(path)} has no integer seed"
