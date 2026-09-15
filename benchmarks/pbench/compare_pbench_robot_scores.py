#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DOMAIN_SUMMARY = Path(
    "/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval/"
    "20260613_131107_qwen36vl/qwen_vqa_summary.json"
)
DEFAULT_QUALITY_SUMMARY = Path(
    "/data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality/"
    "quality_eval/pbench_robot_quality_overall_summary.json"
)

PAPER_DOMAIN = 88.20
PAPER_OVERALL = 82.07
PAPER_QUALITY_INFERRED = 2 * PAPER_OVERALL - PAPER_DOMAIN


def load_json(path: Path) -> dict:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def delta(value: float | None, target: float) -> str:
    return "NA" if value is None else f"{value - target:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local PBench Robot scores with the paper numbers.")
    parser.add_argument("--domain-summary", type=Path, default=DEFAULT_DOMAIN_SUMMARY)
    parser.add_argument("--quality-summary", type=Path, default=DEFAULT_QUALITY_SUMMARY)
    args = parser.parse_args()

    domain = load_json(args.domain_summary) if args.domain_summary.exists() else {}
    quality = load_json(args.quality_summary) if args.quality_summary.exists() else {}

    local_domain = quality.get("domain_score_like", domain.get("domain_score_like"))
    local_quality = quality.get("quality_score")
    local_overall = quality.get("overall_score_like")

    print("PBench Robot score comparison")
    print("=" * 64)
    print(f"{'Metric':<18}{'Local':>12}{'Paper':>12}{'Delta':>12}")
    print("-" * 64)
    print(f"{'Domain':<18}{fmt(local_domain):>12}{PAPER_DOMAIN:>12.4f}{delta(local_domain, PAPER_DOMAIN):>12}")
    print(
        f"{'Quality':<18}{fmt(local_quality):>12}{PAPER_QUALITY_INFERRED:>12.4f}"
        f"{delta(local_quality, PAPER_QUALITY_INFERRED):>12}"
    )
    print(f"{'Overall':<18}{fmt(local_overall):>12}{PAPER_OVERALL:>12.4f}{delta(local_overall, PAPER_OVERALL):>12}")
    print("=" * 64)
    print(f"Domain summary:  {args.domain_summary}")
    print(f"Quality summary: {args.quality_summary}")

    metrics = quality.get("quality_metrics") or {}
    if metrics:
        print("\nQuality metrics")
        for name in sorted(metrics):
            print(f"  {name:<10} {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
