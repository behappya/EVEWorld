#!/usr/bin/env python3
"""Fail-closed preflight and checkpoint audit for the paired campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GAGI = Path("/data/datasets/gagi")
REPO = Path(__file__).resolve().parents[3]
CAMPAIGN_ROOT = GAGI / "eve_v2_outputs/eve_cic_transport_v1"
HISTORICAL_ROOT = GAGI / "eve_v2_outputs/t4g_joint_wmapA_pre_seed42_s300"
HISTORICAL_RUNTIME = (
    HISTORICAL_ROOT
    / "runtime_configs/t4g_wmapA_pre_seed42_s300.json"
)
HISTORICAL_RUNTIME_SHA256 = (
    "f7a4d60bbb954b1c65dc96d0c35ad7dd9b05677be1ba65607b1d8bc2607ea2cf"
)
PACKED_DATA = GAGI / "gr1_finetune_data/packed_data"
ANNO_DIR = GAGI / "eve_v2_outputs/track4gen_probe/t4g_anno"
ASSETS_DIR = GAGI / "eve_v2_outputs/track4gen_probe/aug_assets"
PRETRAIN_TRANSFORMER = GAGI / "giga_world_0_video_pretrain/transformer"
PRETRAIN_VAE = GAGI / "giga_world_0_video_pretrain/vae"
TRAIN_ACCELERATE = GAGI / "envs/giga_world_train_venv/bin/accelerate"
EXPECTED_STEPS = (50, 100, 150, 200, 250, 300)

VARIANTS = {
    "control": {
        "output": CAMPAIGN_ROOT / "control_repro_seed42_s300",
        "config_module": "eveworld.pipeline.t4g_joint_config",
        "runner": "eveworld.pipeline.t4g_joint_trainer.T4GJointTrainer",
    },
    "transport": {
        "output": CAMPAIGN_ROOT / "cic_transport_seed42_s300",
        "config_module": "eveworld.tia_transport.cic_transport_config",
        "runner": (
            "eveworld.tia_transport.cic_transport_trainer."
            "CICTransportJointTrainer"
        ),
    },
}

PROTECTED_ROOTS = (
    HISTORICAL_ROOT,
    GAGI / "eve_v2_outputs/eve_ablation_strict_v1",
    GAGI / "eve_v2_outputs/anchor_models",
    GAGI / "eve_v2_outputs/eval175_gen",
    GAGI / "gr1_dreamgen_eval/eval_outputs",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def assert_output_path(variant: str, output: Path) -> None:
    expected = VARIANTS[variant]["output"].resolve()
    actual = output.resolve()
    if actual != expected:
        raise ValueError(f"{variant} output must be exactly {expected}, got {actual}")
    for protected in PROTECTED_ROOTS:
        protected = protected.resolve()
        if actual == protected or protected in actual.parents:
            raise ValueError(f"refusing protected output path: {actual}")


def count_packed_rows() -> int:
    with (PACKED_DATA / "labels/data.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    return len(rows)


def runtime_from_module(variant: str) -> dict[str, Any]:
    info = VARIANTS[variant]
    config = copy.deepcopy(importlib.import_module(info["config_module"]).config)
    output = info["output"]
    config["project_dir"] = str(output / "experiments")
    config["launch"]["gpu_ids"] = list(range(8))
    config["launch"]["executable"] = str(TRAIN_ACCELERATE)
    train_loader = config["dataloaders"]["train"]
    train_loader["data_or_config"] = [str(PACKED_DATA)]
    train_loader["batch_size_per_gpu"] = 1
    train_loader["num_workers"] = 6
    transform = train_loader["transform"]
    transform.update(num_frames=93, height=480, width=768, fps=16)
    config["models"]["transformer_model_path"] = str(PRETRAIN_TRANSFORMER)
    config["models"]["vae_model_path"] = str(PRETRAIN_VAE)
    config["train"].update(
        max_steps=300,
        gradient_accumulation_steps=8,
        checkpoint_interval=50,
        checkpoint_total_limit=8,
        seed=42,
        mixed_precision="bf16",
        with_ema=True,
        activation_checkpointing=True,
    )
    config["train"].pop("max_epochs", None)
    return config


def normalize_runtime(config: dict[str, Any], variant: str) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["project_dir"] = "<PAIRED_OUTPUT>/experiments"
    if variant == "transport":
        normalized["runners"] = [VARIANTS["control"]["runner"]]
        for key in (
            "cic_transport_after_block",
            "cic_transport_rank",
            "cic_transport_window_radius",
            "cic_transport_temperature",
            "cic_transport_residual_scale",
            "cic_transport_init_seed",
        ):
            normalized["models"].pop(key, None)
    return normalized


def semantic_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        output = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                output.append(f"{path}: missing-left")
            elif key not in right:
                output.append(f"{path}: missing-right")
            else:
                output.extend(semantic_differences(left[key], right[key], path))
        return output
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{prefix}: length {len(left)} != {len(right)}"]
        output = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            output.extend(
                semantic_differences(left_item, right_item, f"{prefix}[{index}]")
            )
        return output
    return [] if left == right else [f"{prefix}: {left!r} != {right!r}"]


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO.parent, text=True
    ).strip()


def preflight(variant: str, prepare: bool) -> dict[str, Any]:
    info = VARIANTS[variant]
    output = info["output"]
    assert_output_path(variant, output)
    if output.exists():
        raise FileExistsError(
            f"refusing existing output (no implicit resume/overwrite): {output}"
        )

    required_files = (
        PACKED_DATA / "config.json",
        PACKED_DATA / "labels/data.pkl",
        ANNO_DIR / "_packidx2vid.json",
        PRETRAIN_TRANSFORMER / "config.json",
        PRETRAIN_TRANSFORMER / "diffusion_pytorch_model.safetensors",
        PRETRAIN_VAE / "config.json",
        PRETRAIN_VAE / "diffusion_pytorch_model.safetensors",
        HISTORICAL_RUNTIME,
    )
    for path in required_files:
        assert_file(path)

    historical_hash = sha256(HISTORICAL_RUNTIME)
    if historical_hash != HISTORICAL_RUNTIME_SHA256:
        raise ValueError(
            f"historical runtime hash changed: {historical_hash}"
        )
    packed_rows = count_packed_rows()
    idx2vid = json.loads(
        (ANNO_DIR / "_packidx2vid.json").read_text(encoding="utf-8")
    )
    anno_files = [
        path for path in ANNO_DIR.glob("*.json") if not path.name.startswith("_")
    ]
    asset_files = list(ASSETS_DIR.glob("*.npz"))
    counts = {
        "packed_rows": packed_rows,
        "idx2vid_rows": len(idx2vid),
        "annotation_files": len(anno_files),
        "asset_files": len(asset_files),
    }
    if any(value != 92 for value in counts.values()):
        raise ValueError(f"strict 92-sample check failed: {counts}")

    planned = runtime_from_module(variant)
    historical = json.loads(HISTORICAL_RUNTIME.read_text(encoding="utf-8"))
    differences = semantic_differences(
        normalize_runtime(planned, variant),
        normalize_runtime(historical, "control"),
    )
    if differences:
        raise ValueError("runtime drift: " + "; ".join(differences[:20]))

    source_files = (
        Path(__file__),
        Path(__file__).with_name("cic_transport_transformer.py"),
        Path(__file__).with_name("cic_transport_trainer.py"),
        Path(__file__).with_name("cic_transport_config.py"),
        REPO / "eveworld/pipeline/t4g_joint_trainer.py",
        REPO / "eveworld/pipeline/t4g_corr_trainer.py",
    )
    report = {
        "schema": "eve-cic-transport-preflight-v1",
        "created_at": utc_now(),
        "variant": variant,
        "output": str(output),
        "config_module": info["config_module"],
        "historical_runtime": str(HISTORICAL_RUNTIME),
        "historical_runtime_sha256": historical_hash,
        "normalized_runtime_differences": differences,
        "counts": counts,
        "expected_checkpoints": list(EXPECTED_STEPS),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "planned_runtime": planned,
    }
    if prepare:
        audit_dir = output / "audit"
        audit_dir.mkdir(parents=True, exist_ok=False)
        (audit_dir / "preflight.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (audit_dir / "PREPARED").write_text(utc_now() + "\n", encoding="utf-8")
    return report


def node_preflight(variant: str, output: Path) -> dict[str, Any]:
    assert_output_path(variant, output)
    assert_file(output / "audit/preflight.json")
    assert_file(output / "audit/PREPARED")
    models = output / "experiments/models"
    if models.exists() and any(models.glob("checkpoint*")):
        raise FileExistsError(f"refusing to resume checkpoints under {models}")
    return {
        "schema": "eve-cic-transport-node-preflight-v1",
        "checked_at": utc_now(),
        "variant": variant,
        "output": str(output),
        "ready": True,
    }


def find_weight(directory: Path) -> Path:
    candidates = (
        directory / "diffusion_pytorch_model.bin",
        directory / "diffusion_pytorch_model.safetensors",
    )
    matches = [path for path in candidates if path.is_file() and path.stat().st_size > 0]
    if len(matches) != 1:
        raise ValueError(f"expected one model weight in {directory}, got {matches}")
    return matches[0]


def audit(variant: str) -> dict[str, Any]:
    output = VARIANTS[variant]["output"]
    assert_output_path(variant, output)
    models = output / "experiments/models"
    checkpoints = []
    for step in EXPECTED_STEPS:
        matches = list(models.glob(f"checkpoint_epoch_*_step_{step}"))
        if len(matches) != 1:
            raise ValueError(f"step {step}: expected one checkpoint, got {matches}")
        checkpoint = matches[0]
        row: dict[str, Any] = {"step": step, "directory": str(checkpoint)}
        for kind in ("transformer", "transformer_ema"):
            model_dir = checkpoint / kind
            config_path = model_dir / "config.json"
            assert_file(config_path)
            weight_path = find_weight(model_dir)
            item = {
                "directory": str(model_dir),
                "config_sha256": sha256(config_path),
                "weight": str(weight_path),
                "weight_bytes": weight_path.stat().st_size,
            }
            sidecar = model_dir / "cic_transport_config.json"
            if variant == "transport":
                assert_file(sidecar)
                item["cic_transport_config_sha256"] = sha256(sidecar)
            elif sidecar.exists():
                raise ValueError(f"control unexpectedly has transport sidecar: {sidecar}")
            row[kind] = item
        checkpoints.append(row)

    report = {
        "schema": "eve-cic-transport-checkpoint-audit-v1",
        "audited_at": utc_now(),
        "variant": variant,
        "output": str(output),
        "expected_steps": list(EXPECTED_STEPS),
        "checkpoints": checkpoints,
        "ready": True,
    }
    audit_path = output / "audit/checkpoint_audit.json"
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--variant", choices=VARIANTS, required=True)
    preflight_parser.add_argument("--prepare", action="store_true")
    node_parser = subparsers.add_parser("node-preflight")
    node_parser.add_argument("--variant", choices=VARIANTS, required=True)
    node_parser.add_argument("--output", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--variant", choices=VARIANTS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        report = preflight(args.variant, args.prepare)
    elif args.command == "node-preflight":
        report = node_preflight(args.variant, args.output)
    else:
        report = audit(args.variant)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
