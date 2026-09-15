#!/usr/bin/env python3
import os, subprocess, sys, json, glob
import numpy as np
ngpu, out_dir, tp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
procs=[]
for g in range(ngpu):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
    procs.append(subprocess.Popen([tp, os.path.join(here,'t4g_dup_test.py'),
        '--shard-index',str(g),'--num-shards',str(ngpu),'--limit','12',
        '--out',f'{out_dir}/part_{g}.json'], env=env, stdout=None, stderr=subprocess.STDOUT))
rc=0
for p in procs: rc|=p.wait()
merged={}
for f in glob.glob(f'{out_dir}/part_*.json'):
    for k,v in json.load(open(f)).items():
        m=merged.setdefault(k,{'dup':[],'bg':[],'sep':[],'n':0})
        m['dup'].append(v['sim_dup_p50']); m['bg'].append(v['bg_p90']); m['sep'].append(v['sep_dup']); m['n']+=v['n_dup']
final={k:{'sim_dup':float(np.mean(v['dup'])),'bg_p90':float(np.mean(v['bg'])),
          'sep_dup':float(np.mean(v['sep'])),'n':v['n']} for k,v in merged.items()}
json.dump(final, open(f'{out_dir}/dup_summary.json','w'), indent=1)
print('MERGED ->', f'{out_dir}/dup_summary.json')
sys.exit(rc)
