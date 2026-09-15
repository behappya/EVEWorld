#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EVAL_ROOT = Path("/data/datasets/gagi/giga_world_0_outputs")
OUT_CSV = EVAL_ROOT / "pbench_quality_upto19_paper_style_summary.csv"
OUT_JSON = EVAL_ROOT / "pbench_quality_upto19_paper_style_summary.json"


ROWS = [
    {
        "label": "Pretrain 3.8s",
        "model": "Pretrain",
        "target": "3.8s",
        "quality": EVAL_ROOT
        / "pbench_robot_vbench_quality/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "pbench_robot_qwen_vqa_eval/pbench_robot_3p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 82.4345,
    },
    {
        "label": "Pretrain 5.8s",
        "model": "Pretrain",
        "target": "5.8s",
        "quality": EVAL_ROOT
        / "pbench_robot_vbench_quality_length_sweep/pbench_robot_5p8s_full_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "pbench_robot_qwen_vqa_eval/pbench_robot_5p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 79.8739,
    },
    {
        "label": "Pretrain 9.8s",
        "model": "Pretrain",
        "target": "9.8s",
        "quality": EVAL_ROOT
        / "pbench_robot_vbench_quality_length_sweep/pbench_robot_9p8s_full_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "pbench_robot_qwen_vqa_eval/pbench_robot_9p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 74.3225,
    },
    {
        "label": "Pretrain 15.8s",
        "model": "Pretrain",
        "target": "15.8s",
        "quality": EVAL_ROOT
        / "pbench_robot_vbench_quality_length_sweep/pbench_robot_15p8s_full_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "pbench_robot_qwen_vqa_eval/pbench_robot_15p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 67.5409,
    },
    {
        "label": "Pretrain 19.8s",
        "model": "Pretrain",
        "target": "19.8s",
        "quality": EVAL_ROOT
        / "pbench_robot_vbench_quality_length_sweep/pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "pbench_robot_qwen_vqa_eval/pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 66.9737,
    },
    {
        "label": "GR1/SFT 3.8s",
        "model": "GR1/SFT",
        "target": "3.8s",
        "quality": EVAL_ROOT
        / "gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_3p8s_full_gr1_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_3p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 80.9784,
    },
    {
        "label": "GR1/SFT 5.8s",
        "model": "GR1/SFT",
        "target": "5.8s",
        "quality": EVAL_ROOT
        / "gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_5p8s_full_gr1_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_5p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 79.8917,
    },
    {
        "label": "GR1/SFT 9.8s",
        "model": "GR1/SFT",
        "target": "9.8s",
        "quality": EVAL_ROOT
        / "gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_9p8s_full_gr1_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_9p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 78.8952,
    },
    {
        "label": "GR1/SFT 15.8s",
        "model": "GR1/SFT",
        "target": "15.8s",
        "quality": EVAL_ROOT
        / "gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_15p8s_full_gr1_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_15p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 68.1180,
    },
    {
        "label": "GR1/SFT 19.8s",
        "model": "GR1/SFT",
        "target": "19.8s",
        "quality": EVAL_ROOT
        / "gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_19p8s_full_gr1_pbench_len_sweep_full_8gpu/quality_eval/pbench_robot_quality_overall_summary.json",
        "domain": EVAL_ROOT
        / "gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_19p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json",
        "domain_fallback": 66.7622,
    },
]

QUALITY_KEYS = ["i2v-bg", "i2v-s", "aes", "img", "bg-con", "mot", "sub-con", "o-con"]
FIELDNAMES = [
    "模型/时长",
    *QUALITY_KEYS,
    "Domain Score",
    "Quality Score",
    "Overall Score",
    "quality_summary",
    "domain_summary",
    "domain_source",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def main() -> None:
    rows: list[dict[str, Any]] = []
    for spec in ROWS:
        quality = load_json(spec["quality"])
        domain = load_json(spec["domain"])
        domain_score = None
        domain_source = ""
        if domain and domain.get("domain_score_like") is not None:
            domain_score = float(domain["domain_score_like"])
            domain_source = "summary"
        elif spec.get("domain_fallback") is not None:
            domain_score = float(spec["domain_fallback"])
            domain_source = "fallback"

        metrics = quality.get("quality_metrics", {}) if quality else {}
        quality_score = quality.get("quality_score") if quality else None
        overall = None
        if quality_score is not None and domain_score is not None:
            overall = (float(quality_score) + float(domain_score)) / 2.0

        row = {
            "模型/时长": spec["label"],
            "Domain Score": fmt(domain_score),
            "Quality Score": fmt(quality_score),
            "Overall Score": fmt(overall),
            "quality_summary": str(spec["quality"]) if spec["quality"].is_file() else "",
            "domain_summary": str(spec["domain"]) if spec["domain"].is_file() else "",
            "domain_source": domain_source,
        }
        for key in QUALITY_KEYS:
            row[key] = fmt(metrics.get(key))
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print()
    print("| " + " | ".join(FIELDNAMES[:12]) + " |")
    print("| " + " | ".join(["---", *[":---:" for _ in FIELDNAMES[1:12]]]) + " |")
    for row in rows:
        print("| " + " | ".join(row[name] or "待补" for name in FIELDNAMES[:12]) + " |")


if __name__ == "__main__":
    main()
