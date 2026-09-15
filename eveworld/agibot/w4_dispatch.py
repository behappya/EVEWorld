#!/usr/bin/env python3
"""w4_detect 的 8 卡分发器(kjob 内跑, subprocess 各绑一卡)。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("N_SHARDS", "8"))

procs = []
for i in range(N):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
    p = subprocess.Popen([sys.executable, f"{HERE}/w4_detect.py",
                          "--shard-index", str(i), "--num-shards", str(N)], env=env)
    procs.append(p)
rc = 0
for p in procs:
    if p.wait() != 0:
        rc = 1
sys.exit(rc)
