#!/usr/bin/env python3
import os, subprocess, sys, json, glob
import numpy as np
ngpu, out_dir, tp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(out_dir, exist_ok=True)
procs=[]
for g in range(ngpu):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
    procs.append(subprocess.Popen([tp, os.path.join(here,'t4g_change_test.py'),
        '--shard-index',str(g),'--num-shards',str(ngpu),'--limit','11',
        '--out',f'{out_dir}/part_{g}.json'], env=env, stdout=None, stderr=subprocess.STDOUT))
rc=0
for p in procs: rc|=p.wait()
merged={}
for f in glob.glob(f'{out_dir}/part_*.json'):
    for k,v in json.load(open(f)).items():
        m=merged.setdefault(k,{x:[] for x in ['at','pre','floor','mover','gapf','dropp']}); m['n']=m.get('n',0)+v['n']
        m['at'].append(v['self_at']); m['pre'].append(v['self_pre']); m['floor'].append(v['floor_static'])
        m['mover'].append(v['self_mover']); m['gapf'].append(v['gap_vs_floor']); m['dropp'].append(v['drop_vs_pre'])
final={k:{'self_at':float(np.mean(v['at'])),'self_pre':float(np.mean(v['pre'])),
          'floor_static':float(np.mean(v['floor'])),'self_mover':float(np.mean(v['mover'])),
          'gap_vs_floor':float(np.mean(v['gapf'])),'drop_vs_pre':float(np.mean(v['dropp'])),'n':v['n']}
       for k,v in merged.items()}
json.dump(final, open(f'{out_dir}/change_summary.json','w'), indent=1)
print('MERGED ->', f'{out_dir}/change_summary.json'); sys.exit(rc)
