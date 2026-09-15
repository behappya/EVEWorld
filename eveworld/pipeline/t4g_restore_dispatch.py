#!/usr/bin/env python3
"""T4G-RESTORE (E12) 分发+合并: 8 卡跑一个模型, 合并出补回率判决表。
每卡跑同一 transformer (round0 或 s150), 视频分片。TRANSFORMER_DIR env 覆盖模型。"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

KINDS = ['target', 'distractor', 'background']


def merge(out_dir):
    recs, model = [], '?'
    for f in sorted(glob.glob(f'{out_dir}/partial_shard*.json')):
        d = json.load(open(f))
        recs += d['records']; model = d.get('model', model)
    if not recs:
        print('no partials!', file=sys.stderr); sys.exit(1)
    sigmas = sorted({r['sigma'] for r in recs})
    summary = {'model': model, 'n_records': len(recs), 'sigmas': sigmas, 'table': {}}
    print(f'\n===== T4G-RESTORE (E12) 消失补回判决 · model={model} =====')
    print('restore: 1=完全补回缺失物体, 0=保留缺失 (照单收下"自信的空")')
    for mode in ['full', 'half']:
        print(f'\n-- erase mode={mode} --')
        print(f'{"sigma":>6} | ' + ' | '.join(f'{k:>10}' for k in KINDS) + ' |  T−bg(direction)')
        for s in sigmas:
            row = {}
            for k in KINDS:
                vals = [r['restore'] for r in recs if r['kind'] == k and r['mode'] == mode and r['sigma'] == s]
                dirs = [r['direction'] for r in recs if r['kind'] == k and r['mode'] == mode and r['sigma'] == s]
                row[k] = (float(np.median(vals)) if vals else float('nan'),
                          float(np.median(dirs)) if dirs else float('nan'), len(vals))
                summary['table'][f'{mode}|s{s}|{k}'] = dict(restore=row[k][0], direction=row[k][1], n=row[k][2])
            tb = row['target'][0] - row['background'][0]
            print(f'{s:>6} | ' + ' | '.join(f'{row[k][0]:>5.3f}(n{row[k][2]:>2})' for k in KINDS)
                  + f' |  {tb:+.3f} (dir T={row["target"][1]:.2f})')
    print('\n判读: target/distractor 显著 > background -> 模型会查首帧库存补货 (消失是训练教偏);')
    print('      target/distractor ≈ background 或全低 -> 保管员镜像成立, 补货技能缺失 -> 消失贴必开。')
    json.dump(summary, open(f'{out_dir}/restore_summary.json', 'w'), indent=1)
    print(f'MERGED -> {out_dir}/restore_summary.json')


def main():
    ngpu, out_dir, tp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    extra = []
    if os.environ.get('TRANSFORMER_DIR'):
        extra += ['--transformer-dir', os.environ['TRANSFORMER_DIR']]
    if os.environ.get('MODEL_DIR'):
        extra += ['--model-dir', os.environ['MODEL_DIR']]
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
        procs.append(subprocess.Popen(
            [tp, os.path.join(here, 't4g_restore_probe.py'),
             '--shard-index', str(g), '--num-shards', str(ngpu), '--out-dir', out_dir] + extra,
            env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    merge(out_dir)
    sys.exit(rc)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--merge-only':
        merge(sys.argv[2])
    else:
        main()
