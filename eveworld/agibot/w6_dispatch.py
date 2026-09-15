#!/usr/bin/env python3
"""w6_aug_prep 的多进程分发(kjob 内, CPU 并行)。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("N_SHARDS", "16"))

procs = [subprocess.Popen([sys.executable, f"{HERE}/w6_aug_prep.py",
                           "--shard-index", str(i), "--num-shards", str(N),
                           "--device", "cpu"])
         for i in range(N)]
rc = 0
for p in procs:
    if p.wait() != 0:
        rc = 1
sys.exit(rc)
