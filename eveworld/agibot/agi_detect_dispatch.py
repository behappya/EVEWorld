#!/usr/bin/env python3
"""AgiBot 检测 8 卡分发器 (kjob 内跑): 轮1 agi_detect, 轮2 agi_gripper_detect。

Python 编排 subprocess 各钉 CUDA_VISIBLE_DEVICES, kjob 脚本零 bash 并发语法。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get('N_SHARDS', '8'))


def run_round(script):
    procs = []
    for i in range(N):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED='1')
        p = subprocess.Popen(
            [sys.executable, os.path.join(HERE, script),
             '--shard-index', str(i), '--num-shards', str(N)],
            env=env, stdout=None, stderr=subprocess.STDOUT)
        procs.append(p)
    rc = 0
    for p in procs:
        if p.wait() != 0:
            rc = 1
    return rc


rc1 = run_round('agi_detect.py')
print(f'[dispatch] agi_detect round rc={rc1}', flush=True)
rc2 = run_round('agi_gripper_detect.py')
print(f'[dispatch] agi_gripper_detect round rc={rc2}', flush=True)
sys.exit(rc1 or rc2)
