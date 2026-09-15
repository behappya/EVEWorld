#!/usr/bin/env python3
"""EXAM v3 分发+合并: 生成含条件图路径的清单 -> 8卡分片 -> 汇总(含 inv0 来源统计)。"""
import glob
import json
import os
import subprocess
import sys

GAGI = '/data/datasets/gagi'
INP = f'{GAGI}/gr1_dreamgen_eval/giga_input'
SPLITS = ['gr1_env', 'gr1_object', 'gr1_behavior']


def cond_map():
    m = {}
    for sp in SPLITS:
        spk = sp.replace('gr1_', '')
        for it in json.load(open(f'{INP}/eval175_{sp}.json')):
            idx = int(it['request_id'].split('_')[-1])
            m[(spk, idx)] = (it['image'], it['prompt'])
    return m


def build_list(out_dir, gen_root, arms):
    cm = cond_map()
    fp = os.path.join(out_dir, 'exam_list_v3.tsv')
    n = 0
    with open(fp, 'w') as f:
        for arm in arms:
            for sp in SPLITS:
                spk = sp.replace('gr1_', '')
                vdir = f'{gen_root}/{arm}/{sp}/generated_only'
                if not os.path.isdir(vdir):
                    continue
                for v in sorted(os.listdir(vdir)):
                    if not v.endswith('.mp4'):
                        continue
                    idx = int(v.split('_', 1)[0])
                    img, prompt = cm.get((spk, idx), ('', ''))
                    f.write(f'{arm}\t{vdir}/{v}\t{prompt}\t{img}\n')
                    n += 1
    print(f'[v3] list: {n} videos -> {fp}')
    return fp


def merge(out_dir):
    recs = []
    for f in glob.glob(f'{out_dir}/exam_v3_part*.json'):
        recs += json.load(open(f))
    summary = {}
    for arm in sorted({r['arm'] for r in recs}):
        rs = [r for r in recs if r['arm'] == arm]
        ok = [r for r in rs if not r['undetectable']]
        und = len(rs) - len(ok)
        cond_src = sum(1 for r in ok if r.get('inv0_src') == 'cond')
        dup = sum(1 for r in ok if r['dup']) / max(1, len(ok))
        van = sum(1 for r in ok if r['vanish']) / max(1, len(ok))
        summary[arm] = dict(n=len(rs), eligible=len(ok), undetectable=und,
                            inv0_from_cond=cond_src, dup_rate=dup, vanish_rate=van)
        print(f"{arm:32s} elig={len(ok):>3}/{len(rs)} (cond={cond_src}) DUP={dup*100:.1f}% VAN={van*100:.1f}%")
    json.dump(dict(summary=summary, records=recs), open(f'{out_dir}/exam_v3_summary.json', 'w'))
    print(f'-> {out_dir}/exam_v3_summary.json')


def main():
    ngpu, out_dir, tp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    gen_root = os.environ['EXAM_GEN']
    arms = os.environ['EXAM_ARMS'].split()
    os.makedirs(out_dir, exist_ok=True)
    lf = build_list(out_dir, gen_root, arms)
    here = os.path.dirname(os.path.abspath(__file__))
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
        procs.append(subprocess.Popen(
            [tp, os.path.join(here, 't4g_exam_v3.py'), '--list-file', lf,
             '--shard-index', str(g), '--num-shards', str(ngpu),
             '--out', f'{out_dir}/exam_v3_part{g}.json'], env=env))
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
