#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any


try:
    import pkg_resources

    if not hasattr(pkg_resources, "packaging") and hasattr(pkg_resources, "extern"):
        pkg_resources.packaging = pkg_resources.extern.packaging
except Exception:
    pass


DEFAULT_OUTPUT_ROOT = Path("/data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality")
DEFAULT_DOMAIN_SUMMARY = Path(
    "/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval/"
    "20260613_131107_qwen36vl/qwen_vqa_summary.json"
)

PAPER_TO_VBENCH_DIM = {
    "i2v-bg": "i2v_background",
    "i2v-s": "i2v_subject",
    "aes": "aesthetic_quality",
    "img": "imaging_quality",
    "bg-con": "background_consistency",
    "mot": "motion_smoothness",
    "sub-con": "subject_consistency",
    "o-con": "overall_consistency",
}

I2V_DIMS = {"i2v_subject", "i2v_background", "camera_motion"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VBench quality metrics for PBench Robot generated videos.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--full-info", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--domain-summary", type=Path, default=DEFAULT_DOMAIN_SUMMARY)
    parser.add_argument("--resolution-name", default="pbench_robot")
    parser.add_argument("--dimensions", nargs="*", default=list(PAPER_TO_VBENCH_DIM.values()))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local", action="store_true", help="Use VBench local checkpoint paths under VBENCH_CACHE_DIR.")
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument("--imaging-quality-preprocessing-mode", default="longer")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def score_value(result: Any) -> float:
    value = result[0] if isinstance(result, (list, tuple)) else result
    return float(value)


def patch_torch_load_for_legacy_vbench_checkpoints() -> None:
    import torch

    original_load = torch.load

    def torch_load_compat(*load_args: Any, **load_kwargs: Any) -> Any:
        target = str(load_args[0]) if load_args else str(load_kwargs.get("f", ""))
        if target.endswith("/amt-s.pth") and "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = False
        return original_load(*load_args, **load_kwargs)

    torch.load = torch_load_compat


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    full_info = (args.full_info or output_root / "pbench_robot_vbench_full_info.json").expanduser().resolve()
    work_dir = (args.work_dir or output_root / "vbench_work").expanduser().resolve()
    output_dir = (args.output_dir or output_root / "quality_eval").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not full_info.exists():
        raise FileNotFoundError(f"Missing full_info JSON. Run prepare_pbench_vbench_inputs.py first: {full_info}")

    import torch
    from vbench.utils import init_submodules as init_vbench_submodules
    from vbench2_beta_i2v.utils import init_submodules as init_i2v_submodules

    patch_torch_load_for_legacy_vbench_checkpoints()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    dimensions = list(dict.fromkeys(args.dimensions))

    old_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        print("============================================", flush=True)
        print("PBench Robot VBench quality eval", flush=True)
        print(f"device:          {device}", flush=True)
        print(f"work_dir:        {work_dir}", flush=True)
        print(f"full_info:       {full_info}", flush=True)
        print(f"output_dir:      {output_dir}", flush=True)
        print(f"dimensions:      {dimensions}", flush=True)
        print(f"resolution_name: {args.resolution_name}", flush=True)
        print("============================================", flush=True)

        i2v_dimensions = [dim for dim in dimensions if dim in I2V_DIMS]
        vbench_dimensions = [dim for dim in dimensions if dim not in I2V_DIMS]
        submodules: dict[str, Any] = {}
        if i2v_dimensions:
            submodules.update(
                init_i2v_submodules(
                    i2v_dimensions,
                    local=args.local,
                    read_frame=args.read_frame,
                    resolution=args.resolution_name,
                )
            )
        if vbench_dimensions:
            submodules.update(
                init_vbench_submodules(
                    vbench_dimensions,
                    local=args.local,
                    read_frame=args.read_frame,
                )
            )

        raw_results: dict[str, Any] = {}
        paper_metrics: dict[str, float] = {}
        for dim in dimensions:
            raw_result_path = output_dir / f"{dim}_raw_result.json"
            if args.skip_existing and raw_result_path.exists():
                print(f"==> Skipping {dim}, existing result found: {raw_result_path}", flush=True)
                result = load_json(raw_result_path)
                raw_results[dim] = result
                score = score_value(result)
                for paper_name, vbench_name in PAPER_TO_VBENCH_DIM.items():
                    if vbench_name == dim:
                        paper_metrics[paper_name] = score * 100.0
                        print(f"{paper_name}: {paper_metrics[paper_name]:.4f}", flush=True)
                        break
                continue

            print(f"==> Evaluating {dim}", flush=True)
            module_name = f"vbench2_beta_i2v.{dim}" if dim in I2V_DIMS else f"vbench.{dim}"
            module = importlib.import_module(module_name)
            if dim == "i2v_background" and hasattr(module, "dreamsim"):
                original_dreamsim = module.dreamsim

                def dreamsim_with_explicit_runtime(*dreamsim_args: Any, **dreamsim_kwargs: Any) -> Any:
                    dreamsim_kwargs.setdefault("device", device)
                    dreamsim_kwargs.setdefault("cache_dir", str(work_dir / "models"))
                    return original_dreamsim(*dreamsim_args, **dreamsim_kwargs)

                module.dreamsim = dreamsim_with_explicit_runtime
            func = getattr(module, f"compute_{dim}")
            kwargs = {}
            if dim == "imaging_quality":
                kwargs["imaging_quality_preprocessing_mode"] = args.imaging_quality_preprocessing_mode
            result = func(str(full_info), device, submodules[dim], **kwargs)
            raw_results[dim] = result
            score = score_value(result)
            for paper_name, vbench_name in PAPER_TO_VBENCH_DIM.items():
                if vbench_name == dim:
                    paper_metrics[paper_name] = score * 100.0
                    print(f"{paper_name}: {paper_metrics[paper_name]:.4f}", flush=True)
                    break
            save_json(raw_result_path, result)

        quality_values = [paper_metrics[name] for name in PAPER_TO_VBENCH_DIM if name in paper_metrics]
        quality_score = sum(quality_values) / len(quality_values) if quality_values else None

        domain_score = None
        domain_summary = None
        if args.domain_summary and args.domain_summary.exists():
            domain_summary = load_json(args.domain_summary)
            domain_score = domain_summary.get("domain_score_like")

        overall = None
        if quality_score is not None and domain_score is not None:
            overall = (float(domain_score) + float(quality_score)) / 2.0

        payload = {
            "domain_score_like": domain_score,
            "quality_score": quality_score,
            "overall_score_like": overall,
            "quality_metrics": paper_metrics,
            "dimensions": dimensions,
            "full_info": str(full_info),
            "work_dir": str(work_dir),
            "output_dir": str(output_dir),
            "domain_summary": str(args.domain_summary) if args.domain_summary else None,
            "domain_summary_payload": domain_summary,
        }
        save_json(output_dir / "pbench_robot_quality_overall_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
