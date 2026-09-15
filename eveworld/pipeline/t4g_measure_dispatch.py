#!/usr/bin/env python3
import os, subprocess, sys, json, glob
ngpu, out_dir, tp, vids = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
procs = []
for g in range(ngpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
    p = subprocess.Popen([tp, os.path.join(here, 't4g_measure.py'),
        '--vids', vids, '--shard-index', str(g), '--num-shards', str(ngpu),
        '--out', f'{out_dir}/part_{g}.json'], env=env, stdout=None, stderr=subprocess.STDOUT)
    procs.append(p)
rc = 0
for p in procs: rc |= p.wait()
# merge
merged = {}
for f in glob.glob(f'{out_dir}/part_*.json'):
    for k, v in json.load(open(f)).items():
        m = merged.setdefault(k, {'epe': [], 'sep': [], 'objp10': [], 'bgp90': [], 'n': 0})
        if v.get('epe_median') is not None: m['epe'].append(v['epe_median'])
        if v.get('separation') is not None: m['sep'].append(v['separation'])
        if v.get('obj_self_p10') is not None: m['objp10'].append(v['obj_self_p10'])
        if v.get('bg_p90') is not None: m['bgp90'].append(v['bg_p90'])
        m['n'] += v.get('n', 0)
import numpy as np
final = {k: {'epe': float(np.mean(v['epe'])) if v['epe'] else None,
             'separation': float(np.mean(v['sep'])) if v['sep'] else None,
             'obj_p10': float(np.mean(v['objp10'])) if v['objp10'] else None,
             'bg_p90': float(np.mean(v['bgp90'])) if v['bgp90'] else None} for k, v in merged.items()}
json.dump(final, open(f'{out_dir}/measure_summary.json', 'w'), indent=1)
print('MERGED ->', f'{out_dir}/measure_summary.json')
sys.exit(rc)
