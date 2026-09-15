#!/usr/bin/env python3
"""8 卡分发 tia_probe.py 并汇总逐层 EPE 曲线选 ℓ*。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

N_GPU = int(os.environ.get("N_GPU", "8"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/data/datasets/gagi/flowwam/igr/tia_probe"

procs = []
for g in range(N_GPU):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "tia_probe.py"),
         "--shard", f"{g}/{N_GPU}"] + sys.argv[1:],
        env=env,
    )
    procs.append(p)
rc = 0
for p in procs:
    p.wait()
    rc = rc or p.returncode

# 汇总（v2: moving 对 mean EPE 主指标 + top-1 命中率副指标）
agg = {}
for f in os.listdir(OUT_DIR):
    if not f.startswith("probe_shard_"):
        continue
    for k, v in json.load(open(os.path.join(OUT_DIR, f))).items():
        agg.setdefault(k, []).extend(v if isinstance(v, list) else [v])

summary = {}
for k, vals in agg.items():
    vals = [v for v in vals if isinstance(v, dict)]
    mov = [v["mean_moving"] for v in vals if not np.isnan(v.get("mean_moving", np.nan))]
    hit = [v["hit_moving"] for v in vals if not np.isnan(v.get("hit_moving", np.nan))]
    n_mov = sum(v.get("n_moving", 0) for v in vals)
    if not mov:
        continue
    bi, tf = k.split("|")
    summary.setdefault(int(bi), {})[float(tf)] = {
        "mean_moving_epe": float(np.mean(mov)),
        "hit_moving": float(np.mean(hit)) if hit else None,
        "n_moving_pairs": n_mov, "n_videos": len(mov)}

table = []
for bi in sorted(summary):
    m = float(np.mean([d["mean_moving_epe"] for d in summary[bi].values()]))
    h = float(np.mean([d["hit_moving"] for d in summary[bi].values() if d["hit_moving"] is not None]))
    table.append((bi, m, h, summary[bi]))
table_sorted = sorted(table, key=lambda x: x[1])
best = table_sorted[0] if table_sorted else None

report = {
    "metric": "mean EPE over moving pairs (token cells), lower is better",
    "per_block": {str(bi): per_t for bi, _, _, per_t in table},
    "block_mean_moving_epe": {str(bi): m for bi, m, _, _ in table},
    "block_hit_moving": {str(bi): h for bi, _, h, _ in table},
    "best_block": best[0] if best else None,
    "best_mean_moving_epe": best[1] if best else None,
}
json.dump(report, open(os.path.join(OUT_DIR, "probe_summary.json"), "w"), indent=1)
print("=== 逐 block: moving 对 mean EPE / top-1 命中率 ===")
for bi, m, h, _ in table:
    mark = "  <== ℓ*" if best and bi == best[0] else ""
    print(f"block {bi:>2}: epe={m:.3f} hit={h:.3f}{mark}")
print(f"DISPATCH_DONE rc={rc}")
sys.exit(rc)
