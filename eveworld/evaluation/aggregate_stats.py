#!/usr/bin/env python3
"""EVE · 多模型/多seed 统计聚合(纯 CPU)。
读 metrics_dir 下 <model>[.seedK].json(process_metrics 产物),
输出各指标均值 + bootstrap 95%CI + 相对 baseline 的配对显著性提示。
用法: python3 aggregate_stats.py --metrics-dir DIR --out-dir OUT
"""
import argparse, json, glob, os, re
import numpy as np

METRIC_KEYS = ["PCR_premature", "MBC_motion_before_contact", "TELE_teleport",
               "COMP_completeness", "CBE_cause_before_effect", "LAZINESS_RATE"]


def boot_ci(vals, n=2000, seed=0):
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.RandomState(seed)
    arr = np.array(vals, dtype=float)
    means = [arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def model_name(fn):
    b = os.path.basename(fn).replace(".json", "")
    return re.sub(r"[._]seed\d+$", "", b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.metrics_dir, "*.json")))
    # 按模型聚合各 seed 的 per-video lazy(用于配对检验)
    by_model = {}
    for f in files:
        j = json.load(open(f))
        m = model_name(f)
        by_model.setdefault(m, {"summaries": [], "per_video": {}})
        by_model[m]["summaries"].append(j["summary"])
        for r in j.get("per_video", []):
            by_model[m]["per_video"].setdefault(r["video"], []).append(r["lazy"])
    report = {}
    for m, d in by_model.items():
        report[m] = {"n_seeds": len(d["summaries"])}
        for k in METRIC_KEYS:
            vals = [s[k] for s in d["summaries"] if k in s]
            mean, lo, hi = boot_ci(vals) if len(vals) > 1 else (vals[0] if vals else float("nan"), float("nan"), float("nan"))
            report[m][k] = {"mean": mean, "ci95": [lo, hi]}
    # 相对 baseline 的配对显著性(McNemar 近似:逐视频 lazy 0/1)
    base = next((m for m in by_model if "base" in m or "pretrain" in m), None)
    if base:
        for m in by_model:
            if m == base:
                continue
            common = [v for v in by_model[m]["per_video"] if v in by_model[base]["per_video"]]
            if not common:
                continue
            bl = np.array([np.mean(by_model[base]["per_video"][v]) > 0.5 for v in common])
            ml = np.array([np.mean(by_model[m]["per_video"][v]) > 0.5 for v in common])
            b01 = int(((bl == 1) & (ml == 0)).sum())   # baseline lazy, model 修复
            b10 = int(((bl == 0) & (ml == 1)).sum())   # baseline ok, model 变差
            report[m]["vs_baseline"] = {"fixed": b01, "worsened": b10, "n": len(common),
                                         "hint": "fixed>>worsened 说明有效;建议再跑精确 McNemar 检验"}
    json.dump(report, open(os.path.join(a.out_dir, "stats_report.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", os.path.join(a.out_dir, "stats_report.json"))


if __name__ == "__main__":
    main()
