#!/usr/bin/env python3
"""agi_aug_prep 8 卡分发器 (kjob 内跑)。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get('N_SHARDS', '8'))

procs = []
for i in range(N):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED='1')
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, 'agi_aug_prep.py'),
         '--shard-index', str(i), '--num-shards', str(N)],
        env=env, stdout=None, stderr=subprocess.STDOUT)
    procs.append(p)
rc = 0
for p in procs:
    if p.wait() != 0:
        rc = 1
sys.exit(rc)
