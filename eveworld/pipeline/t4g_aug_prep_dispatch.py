#!/usr/bin/env python3
"""T4G-AUG-PREP 分发: 8 卡各跑一个视频分片 (kjobctl 静态扫描兼容, 无 wait/&)。"""
import os
import subprocess
import sys

ngpu, tp = int(sys.argv[1]), sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
extra = []
for env_name, flag in (('AUG_OUT_DIR', '--out-dir'),
                       ('AUG_ANNO_DIR', '--anno-dir'),
                       ('AUG_VIDEO_ROOT', '--video-root')):
    if os.environ.get(env_name):
        extra.extend([flag, os.environ[env_name]])
if os.environ.get('AUG_VIDS'):
    extra.append('--vids')
    extra.extend(os.environ['AUG_VIDS'].replace(',', ' ').split())
procs = []
for g in range(ngpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
    procs.append(subprocess.Popen(
        [tp, os.path.join(here, 't4g_aug_prep.py'),
         '--shard-index', str(g), '--num-shards', str(ngpu), *extra], env=env))
rc = 0
for p in procs:
    rc |= p.wait()
sys.exit(rc)
