#!/usr/bin/env python3
"""T4G-EXAM 分发+合并: 枚举八臂 eval175 视频 -> 8 卡分片体检 -> 病例率总表。"""
import glob
import json
import os
import subprocess
import sys

GAGI = '/data/datasets/gagi'
GEN = f'{GAGI}/eve_v2_outputs/eval175_gen'
INP = f'{GAGI}/gr1_dreamgen_eval/giga_input'
ARMS = ['pretrain', 'gr1_2b', 'round0', 't4g_wmapA_s50', 't4g_wmapA_s100',
        't4g_wmapA_s150', 't4g_wmaponly_s150']
SPLITS = ['gr1_env', 'gr1_object', 'gr1_behavior']


def build_list(out_dir, arms):
    fp = os.path.join(out_dir, 'exam_list.tsv')
    n = 0
    with open(fp, 'w') as f:
        for arm in arms:
            for sp in SPLITS:
                prompts = [it['prompt'] for it in json.load(open(f'{INP}/eval175_{sp}.json'))]
                vdir = f'{GEN}/{arm}/{sp}/generated_only'
                if not os.path.isdir(vdir):
                    continue
                vids = {int(x.split('_', 1)[0]): x for x in os.listdir(vdir) if x.endswith('.mp4')}
                for i, p in enumerate(prompts):
                    if i in vids:
                        f.write(f'{arm}\t{vdir}/{vids[i]}\t{p}\n')
                        n += 1
    print(f'[exam] list: {n} videos -> {fp}')
    return fp


def merge(out_dir):
    recs = []
    for f in glob.glob(f'{out_dir}/exam_part*.json'):
        recs += json.load(open(f))
    arms = sorted({r['arm'] for r in recs})
    print(f'\n===== T4G-EXAM 恒存性体检 (GDINO 确定性数病例) =====')
    print(f'{"臂":20s} {"n":>4s} {"不可检":>5s} | {"DUP复制率":>8s} {"VANISH消失率":>10s} {"平均峰值数":>8s}')
    summary = {}
    for arm in arms:
        rs = [r for r in recs if r['arm'] == arm]
        ok = [r for r in rs if not r['undetectable']]
        und = len(rs) - len(ok)
        dup = sum(1 for r in ok if r['dup']) / max(1, len(ok))
        van = sum(1 for r in ok if r['vanish']) / max(1, len(ok))
        mx = sum(r['max_count'] - r['inv0'] for r in ok) / max(1, len(ok))
        summary[arm] = dict(n=len(rs), undetectable=und, dup_rate=dup, vanish_rate=van,
                            mean_excess_peak=mx)
        print(f'{arm:20s} {len(rs):>4d} {und:>5d} | {dup:>8.3f} {van:>10.3f} {mx:>8.2f}')
    json.dump(dict(summary=summary, records=recs), open(f'{out_dir}/exam_summary.json', 'w'))
    print(f'MERGED -> {out_dir}/exam_summary.json')


def main():
    ngpu, out_dir, tp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    arms = os.environ.get('EXAM_ARMS', '').split() or ARMS
    lf = build_list(out_dir, arms)
    here = os.path.dirname(os.path.abspath(__file__))
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
        procs.append(subprocess.Popen(
            [tp, os.path.join(here, 't4g_exam.py'), '--list-file', lf,
             '--shard-index', str(g), '--num-shards', str(ngpu),
             '--out', f'{out_dir}/exam_part{g}.json'], env=env))
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
