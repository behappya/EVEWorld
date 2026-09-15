#!/usr/bin/env python3
"""8 卡数据并行分发 gdino_annotate.py（kjob 铁律：Python 编排, 无 bash wait/&）。"""
from __future__ import annotations

import os
import subprocess
import sys

N_GPU = int(os.environ.get("N_GPU", "8"))
HERE = os.path.dirname(os.path.abspath(__file__))

procs = []
for g in range(N_GPU):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "gdino_annotate.py"),
         "--shard", f"{g}/{N_GPU}"] + sys.argv[1:],
        env=env,
    )
    procs.append(p)

rc = 0
for p in procs:
    p.wait()
    rc = rc or p.returncode
print(f"DISPATCH_DONE rc={rc}", flush=True)
sys.exit(rc)
