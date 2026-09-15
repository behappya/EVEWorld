#!/usr/bin/env python3
"""Compare paired EVAL-175 VideoPhy PA-II raw scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


SPLITS = ("gr1_env", "gr1_object", "gr1_behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline", default="round0")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--pa2-filename", default="pa2_eval175_pa_ii.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aliases(path_value: str) -> set[str]:
    path = Path(path_value).expanduser()
    values = {str(path), str(path.absolute())}
    try:
        values.add(str(path.resolve()))
    except OSError:
        pass
    return values


def load_scores(manifest_path: Path, score_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_jsonl(manifest_path)
    path_to_sample: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in manifest:
        sample_key = row["sample_key"]
        metadata[sample_key] = row
        for field in ("video_path", "pa2_video_path"):
            if row.get(field):
                for alias in aliases(row[field]):
                    path_to_sample[alias] = sample_key

    scores: dict[str, dict[str, Any]] = {}
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            video_path = row.get("video_path") or row.get("videopath") or ""
            sample_key = next((path_to_sample[item] for item in aliases(video_path) if item in path_to_sample), "")
            if not sample_key:
                raise ValueError(f"{score_path}:{line_number}: cannot map {video_path}")
            if sample_key in scores:
                raise ValueError(f"{score_path}:{line_number}: duplicate {sample_key}")
            scores[sample_key] = {
                "raw_score": float(row["raw_score"]),
                "prediction": int(float(row["prediction"])),
            }
    if len(scores) != len(manifest):
        raise ValueError(f"{score_path}: mapped {len(scores)} scores for {len(manifest)} manifest rows")
    return scores, manifest


def bootstrap_mean_ci(deltas: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def distribution(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
        "positive_at_0.5": int((values >= 0.5).sum()),
        "count_at_0.25": int((values >= 0.25).sum()),
        "count_at_0.1": int((values >= 0.1).sum()),
    }


def compare_group(
    keys: list[str],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_values = np.asarray([baseline[key]["raw_score"] for key in keys], dtype=np.float64)
    candidate_values = np.asarray([candidate[key]["raw_score"] for key in keys], dtype=np.float64)
    deltas = candidate_values - baseline_values
    nonzero = deltas[deltas != 0]
    if len(nonzero):
        test = wilcoxon(candidate_values, baseline_values, alternative="two-sided", zero_method="wilcox")
        wilcoxon_statistic = float(test.statistic)
        wilcoxon_p = float(test.pvalue)
    else:
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0
    return {
        "count": len(keys),
        "baseline": distribution(baseline_values),
        "candidate": distribution(candidate_values),
        "delta": {
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "bootstrap_mean_95": bootstrap_mean_ci(deltas, bootstrap_samples, seed),
            "wins": int((deltas > 0).sum()),
            "losses": int((deltas < 0).sum()),
            "ties": int((deltas == 0).sum()),
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_p": wilcoxon_p,
        },
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    manifest_root = args.manifest_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_scores, baseline_manifest = load_scores(
        manifest_root / f"{args.baseline}.jsonl",
        results_root / args.baseline / args.pa2_filename,
    )
    candidate_scores, candidate_manifest = load_scores(
        manifest_root / f"{args.candidate}.jsonl",
        results_root / args.candidate / args.pa2_filename,
    )
    baseline_samples = {row["sample_key"] for row in baseline_manifest}
    candidate_samples = {row["sample_key"] for row in candidate_manifest}
    if baseline_samples != candidate_samples:
        raise ValueError("Baseline and candidate manifests do not contain identical sample keys")
    metadata = {row["sample_key"]: row for row in baseline_manifest}

    groups: dict[str, Any] = {}
    for group in (*SPLITS, "overall"):
        keys = sorted(
            key for key in baseline_samples if group == "overall" or metadata[key]["split"] == group
        )
        groups[group] = compare_group(
            keys,
            baseline_scores,
            candidate_scores,
            args.bootstrap_samples,
            args.seed + len(groups),
        )

    payload = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "score": "VideoPhy PA-II raw entailment probability",
        "binary_threshold": 0.5,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "groups": groups,
    }
    json_path = output_dir / f"pa2_{args.candidate}_vs_{args.baseline}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# PA-II raw: {args.candidate} vs {args.baseline}",
        "",
        "| Split | N | Baseline mean | Candidate mean | Mean delta | 95% bootstrap CI | W/L/T | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in (*SPLITS, "overall"):
        result = groups[group]
        delta = result["delta"]
        ci = delta["bootstrap_mean_95"]
        lines.append(
            f"| {group} | {result['count']} | {fmt(result['baseline']['mean'])} | "
            f"{fmt(result['candidate']['mean'])} | {fmt(delta['mean'])} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {delta['wins']}/{delta['losses']}/{delta['ties']} | "
            f"{delta['wilcoxon_p']:.4g} |"
        )
    markdown_path = output_dir / f"pa2_{args.candidate}_vs_{args.baseline}.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
