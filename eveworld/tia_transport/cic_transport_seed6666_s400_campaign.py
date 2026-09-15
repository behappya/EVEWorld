#!/usr/bin/env python3
"""Fail-closed preflight and checkpoint audit for Transport seed6666 s400."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eveworld.tia_transport import cic_transport_campaign as paired


CONFIG_MODULE = (
    "eveworld.tia_transport.cic_transport_seed6666_s400_config"
)
OUTPUT = paired.CAMPAIGN_ROOT / "cic_transport_seed6666_s400"
HISTORICAL_SEED6666_RUNTIME = (
    paired.GAGI
    / "eve_v2_outputs/t4g_joint_wmapA_pre/runtime_configs/"
    "t4g_wmapA_pre300_resume150.json"
)
EXPECTED_STEPS = tuple(range(50, 401, 50))
RUNNER = (
    "eveworld.tia_transport.cic_transport_trainer."
    "CICTransportJointTrainer"
)
CHECKPOINT_RE = re.compile(r"checkpoint_epoch_\d+_step_(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_output(output: Path) -> None:
    if output.resolve() != OUTPUT.resolve():
        raise ValueError(f"output must be exactly {OUTPUT}, got {output}")
    for protected in paired.PROTECTED_ROOTS:
        protected = protected.resolve()
        actual = output.resolve()
        if actual == protected or protected in actual.parents:
            raise ValueError(f"refusing protected output path: {actual}")


def planned_runtime() -> dict[str, Any]:
    config = copy.deepcopy(importlib.import_module(CONFIG_MODULE).config)
    config["project_dir"] = str(OUTPUT / "experiments")
    config["launch"]["gpu_ids"] = list(range(8))
    config["launch"]["executable"] = str(paired.TRAIN_ACCELERATE)
    train_loader = config["dataloaders"]["train"]
    train_loader["data_or_config"] = [str(paired.PACKED_DATA)]
    train_loader["batch_size_per_gpu"] = 1
    train_loader["num_workers"] = 6
    train_loader["transform"].update(
        num_frames=93, height=480, width=768, fps=16
    )
    config["models"]["transformer_model_path"] = str(
        paired.PRETRAIN_TRANSFORMER
    )
    config["models"]["vae_model_path"] = str(paired.PRETRAIN_VAE)
    config["train"].update(
        max_steps=400,
        gradient_accumulation_steps=8,
        checkpoint_interval=50,
        checkpoint_total_limit=8,
        seed=6666,
        mixed_precision="bf16",
        with_ema=True,
        activation_checkpointing=True,
    )
    config["train"].pop("max_epochs", None)
    return config


def normalize_against_seed6666(config: dict[str, Any]) -> dict[str, Any]:
    normalized = paired.normalize_runtime(config, "transport")
    # The only intentional budget difference from old seed6666 is 400 vs 300.
    normalized["train"]["max_steps"] = 300
    return normalized


def preflight(prepare: bool) -> dict[str, Any]:
    assert_output(OUTPUT)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing output: {OUTPUT}")

    required = (
        paired.PACKED_DATA / "config.json",
        paired.PACKED_DATA / "labels/data.pkl",
        paired.ANNO_DIR / "_packidx2vid.json",
        paired.PRETRAIN_TRANSFORMER / "config.json",
        paired.PRETRAIN_TRANSFORMER / "diffusion_pytorch_model.safetensors",
        paired.PRETRAIN_VAE / "config.json",
        paired.PRETRAIN_VAE / "diffusion_pytorch_model.safetensors",
        HISTORICAL_SEED6666_RUNTIME,
    )
    for path in required:
        paired.assert_file(path)

    idx2vid = json.loads(
        (paired.ANNO_DIR / "_packidx2vid.json").read_text(encoding="utf-8")
    )
    counts = {
        "packed_rows": paired.count_packed_rows(),
        "idx2vid_rows": len(idx2vid),
        "annotation_files": len(
            [
                path
                for path in paired.ANNO_DIR.glob("*.json")
                if not path.name.startswith("_")
            ]
        ),
        "asset_files": len(list(paired.ASSETS_DIR.glob("*.npz"))),
    }
    if any(value != 92 for value in counts.values()):
        raise ValueError(f"strict 92-sample check failed: {counts}")

    planned = planned_runtime()
    historical = json.loads(
        HISTORICAL_SEED6666_RUNTIME.read_text(encoding="utf-8")
    )
    differences = paired.semantic_differences(
        normalize_against_seed6666(planned),
        paired.normalize_runtime(historical, "control"),
    )
    if differences:
        raise ValueError("runtime drift: " + "; ".join(differences[:20]))

    source_files = (
        Path(__file__),
        Path(__file__).with_name("cic_transport_seed6666_s400_config.py"),
        Path(__file__).with_name("cic_transport_transformer.py"),
        Path(__file__).with_name("cic_transport_trainer.py"),
    )
    report = {
        "schema": "eve-cic-transport-seed6666-s400-preflight-v1",
        "created_at": utc_now(),
        "output": str(OUTPUT),
        "config_module": CONFIG_MODULE,
        "runner": RUNNER,
        "historical_seed6666_runtime": str(HISTORICAL_SEED6666_RUNTIME),
        "historical_seed6666_runtime_sha256": paired.sha256(
            HISTORICAL_SEED6666_RUNTIME
        ),
        "normalized_runtime_differences_after_budget_alignment": differences,
        "intentional_differences": {
            "project_dir": str(OUTPUT / "experiments"),
            "runner": RUNNER,
            "transport_adapter": True,
            "max_steps": {"historical": 300, "planned": 400},
        },
        "counts": counts,
        "expected_checkpoints": list(EXPECTED_STEPS),
        "planned_runtime": planned,
        "source_sha256": {str(path): paired.sha256(path) for path in source_files},
    }
    if prepare:
        audit_dir = OUTPUT / "audit"
        audit_dir.mkdir(parents=True, exist_ok=False)
        (audit_dir / "preflight.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (audit_dir / "PREPARED").write_text(utc_now() + "\n", encoding="utf-8")
    return report


def node_preflight(output: Path, allow_resume: bool) -> dict[str, Any]:
    assert_output(output)
    paired.assert_file(output / "audit/preflight.json")
    paired.assert_file(output / "audit/PREPARED")
    checkpoints = list((output / "experiments/models").glob("checkpoint_epoch_*"))
    if checkpoints and not allow_resume:
        raise FileExistsError("fresh run refuses existing checkpoints")
    if allow_resume and not checkpoints:
        raise FileNotFoundError("resume requested but no checkpoints exist")
    return {
        "schema": "eve-cic-transport-seed6666-s400-node-preflight-v1",
        "checked_at": utc_now(),
        "output": str(output),
        "allow_resume": allow_resume,
        "existing_checkpoints": [str(path) for path in sorted(checkpoints)],
        "ready": True,
    }


def audit() -> dict[str, Any]:
    assert_output(OUTPUT)
    models = OUTPUT / "experiments/models"
    checkpoint_dirs = list(models.glob("checkpoint_epoch_*_step_*"))
    by_step: dict[int, list[Path]] = {}
    for path in checkpoint_dirs:
        match = CHECKPOINT_RE.search(path.name)
        if match:
            by_step.setdefault(int(match.group(1)), []).append(path)
    if set(by_step) != set(EXPECTED_STEPS):
        raise ValueError(
            f"checkpoint steps mismatch: expected {EXPECTED_STEPS}, got {sorted(by_step)}"
        )

    rows = []
    for step in EXPECTED_STEPS:
        matches = by_step[step]
        if len(matches) != 1:
            raise ValueError(f"step {step}: expected one checkpoint, got {matches}")
        checkpoint = matches[0]
        row: dict[str, Any] = {"step": step, "directory": str(checkpoint)}
        for kind in ("transformer", "transformer_ema"):
            model_dir = checkpoint / kind
            config_path = model_dir / "config.json"
            sidecar = model_dir / "cic_transport_config.json"
            paired.assert_file(config_path)
            paired.assert_file(sidecar)
            weight = paired.find_weight(model_dir)
            row[kind] = {
                "directory": str(model_dir),
                "config_sha256": paired.sha256(config_path),
                "cic_transport_config_sha256": paired.sha256(sidecar),
                "weight": str(weight),
                "weight_bytes": weight.stat().st_size,
            }
        rows.append(row)

    report = {
        "schema": "eve-cic-transport-seed6666-s400-checkpoint-audit-v1",
        "audited_at": utc_now(),
        "output": str(OUTPUT),
        "expected_steps": list(EXPECTED_STEPS),
        "checkpoints": rows,
        "ready": True,
    }
    (OUTPUT / "audit/checkpoint_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--prepare", action="store_true")
    node_parser = subparsers.add_parser("node-preflight")
    node_parser.add_argument("--output", type=Path, required=True)
    node_parser.add_argument("--allow-resume", action="store_true")
    subparsers.add_parser("audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        report = preflight(args.prepare)
    elif args.command == "node-preflight":
        report = node_preflight(args.output, args.allow_resume)
    else:
        report = audit()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
