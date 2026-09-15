#!/usr/bin/env python3
"""8 卡分发 t4g_detect（Python 编排, 避免 kjob heredoc 缩进坑）。"""
import os
import subprocess
import sys

ngpu = int(sys.argv[1])
out_dir = sys.argv[2]
tp = sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))

procs = []
for g in range(ngpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
    p = subprocess.Popen(
        [tp, os.path.join(here, 't4g_detect.py'),
         '--shard-index', str(g), '--num-shards', str(ngpu), '--out-dir', out_dir],
        env=env, stdout=None, stderr=subprocess.STDOUT)
    procs.append(p)

rc = 0
for p in procs:
    rc |= p.wait()
sys.exit(rc)
