#!/usr/bin/env python3
"""EVE P0 · 标注一致性(κ)+ 自动度量校准(可测性判据)。

用法:
  # 双标注者一致性
  python3 agreement.py kappa --a annotator1.jsonl --b annotator2.jsonl
  # 自动度量 vs 人工(相关性)
  python3 agreement.py calib --labels annotator1.jsonl --metrics summary.json
"""
import argparse, json
import numpy as np

DIMS = ["premature", "mbc", "teleport", "incomplete", "not_executable", "lazy_any"]


def load(path):
    return {json.loads(l)["video"]: json.loads(l) for l in open(path) if l.strip()}


def cohen_kappa(y1, y2):
    y1, y2 = np.array(y1), np.array(y2)
    po = (y1 == y2).mean()
    p1 = ((y1 == 1).mean() * (y2 == 1).mean()) + ((y1 == 0).mean() * (y2 == 0).mean())
    return float((po - p1) / (1 - p1 + 1e-9))


def cmd_kappa(a):
    A, B = load(a.a), load(a.b)
    keys = [k for k in A if k in B]
    print(f"匹配 {len(keys)} 条")
    for d in DIMS:
        y1 = [A[k][d] for k in keys if A[k].get(d) is not None and B[k].get(d) is not None]
        y2 = [B[k][d] for k in keys if A[k].get(d) is not None and B[k].get(d) is not None]
        if y1:
            print(f"  {d:18s} κ={cohen_kappa(y1, y2):.3f}  (n={len(y1)})")


def cmd_calib(a):
    lab = load(a.labels)
    mets = {r["video"]: r for r in json.load(open(a.metrics))["per_video"]}
    keys = [k for k in lab if k in mets]
    print(f"匹配 {len(keys)} 条 (人工 vs 自动)")
    # 人工 lazy_any vs 自动 lazy
    yh = np.array([lab[k]["lazy_any"] for k in keys])
    ya = np.array([mets[k]["lazy"] for k in keys])
    if len(yh):
        acc = (yh == ya).mean()
        try:
            from scipy.stats import spearmanr
            rho = spearmanr(yh, ya)[0]
        except Exception:
            rho = float(np.corrcoef(yh, ya)[0, 1]) if len(yh) > 1 else float("nan")
        print(f"  agreement(lazy): {acc:.3f}   spearman: {rho:.3f}")
        print("  可测性判据: agreement≥0.7 或 spearman≥0.6 视为通过")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("kappa"); p1.add_argument("--a", required=True); p1.add_argument("--b", required=True)
    p2 = sub.add_parser("calib"); p2.add_argument("--labels", required=True); p2.add_argument("--metrics", required=True)
    a = ap.parse_args()
    {"kappa": cmd_kappa, "calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    main()
