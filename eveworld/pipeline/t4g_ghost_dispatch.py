#!/usr/bin/env python3
"""T4G-GHOST 分发+合并: 8 卡各跑一个视频分片, 合并出影子判决表 (配对差+bootstrap CI)。"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

ZONES = ['B_pre', 'B_post', 'A_post', 'distr', 'static_far', 'corridor']
BINS = ['Bbin_bin0', 'Bbin_bin1', 'Bbin_bin2', 'Bbin_bin3']
PATH = [f'path_dt{d}' for d in range(1, 9)]


def boot_ci(vals, n=10000, seed=0):
    v = np.array([x for x in vals if x == x])
    if len(v) < 3:
        return float('nan'), float('nan'), len(v)
    rng = np.random.RandomState(seed)
    meds = np.median(v[rng.randint(0, len(v), (n, len(v)))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)), len(v)


def merge(out_dir):
    pv, paste = {}, []
    for f in sorted(glob.glob(f'{out_dir}/partial_shard*.json')):
        d = json.load(open(f))
        pv.update(d['per_video'])
        paste += d['paste']
    if not pv and not paste:
        print('no partials!', file=sys.stderr)
        sys.exit(1)
    sigmas = sorted({float(k[1:]) for v in pv.values() for k in v if k.startswith('s')})
    summary = {'n_videos': len(pv), 'sigmas': sigmas, 'zones': {}, 'paired': {}, 'bins': {},
               'path_profile': {}, 'tol': {}, 'paste': {}, 'per_video': pv}
    print(f'\n===== T4G-GHOST 影子探针判决表 (n={len(pv)} 条) =====')
    for s in sigmas:
        key = f's{s}'
        rows = [v[key] for v in pv.values() if key in v]
        zmed = {z: float(np.nanmedian([r.get(z, np.nan) for r in rows])) for z in ZONES}
        summary['zones'][key] = zmed
        # 配对差 (每视频自身控制)
        pr = {}
        for a, b in [('B_pre', 'static_far'), ('B_pre', 'distr'), ('A_post', 'static_far')]:
            diffs = [r.get(a, np.nan) - r.get(b, np.nan) for r in rows]
            lo, hi, n = boot_ci(diffs)
            med = float(np.nanmedian([d for d in diffs if d == d])) if n else float('nan')
            npos = int(sum(1 for d in diffs if d == d and d > 0))
            pr[f'{a}-{b}'] = dict(median=med, ci=[lo, hi], n=n, n_pos=npos)
        summary['paired'][key] = pr
        summary['bins'][key] = {b: float(np.nanmedian([r.get(b, np.nan) for r in rows])) for b in BINS}
        summary['path_profile'][key] = {p: float(np.nanmedian([r.get(p, np.nan) for r in rows])) for p in PATH}
        summary['tol'][key] = {q: float(np.nanmedian([r.get(q, np.nan) for r in rows]))
                               for q in ('tol_p50', 'tol_p90', 'tol_p95')}
        pd = pr['B_pre-static_far']
        print(f'\n- sigma={s}: B_pre={zmed["B_pre"]:.4f} far={zmed["static_far"]:.4f} '
              f'distr={zmed["distr"]:.4f} A_post={zmed["A_post"]:.4f} corridor={zmed["corridor"]:.4f}')
        print(f'    paired B_pre-far: med={pd["median"]:+.4f} CI[{pd["ci"][0]:+.4f},{pd["ci"][1]:+.4f}] '
              f'pos={pd["n_pos"]}/{pd["n"]}')
        print(f'    B时间纹 bins(0-3): ' + ' '.join(f'{summary["bins"][key][b]:.4f}' for b in BINS)
              + '   (递增=影子ramp, 平=纹理混淆)')
        print(f'    路径涂抹 dt1-8: ' + ' '.join(f'{summary["path_profile"][key][p]:.4f}' for p in PATH))
        print(f'    tol标定 far p50/p90/p95: ' + ' '.join(f'{summary["tol"][key][q]:.4f}'
              for q in ('tol_p50', 'tol_p90', 'tol_p95')))
    if paste:
        agg = {}
        for e in paste:
            for r in e['results']:
                agg.setdefault((r['alpha'], r['sigma']), {'ret': [], 'dir': []})
                agg[(r['alpha'], r['sigma'])]['ret'].append(r['retention'])
                agg[(r['alpha'], r['sigma'])]['dir'].append(r['direction'])
        print(f'\n- 保留探针 (n={len(paste)} 条): retention(1=完整保留复制品) / direction')
        for (a, s), v in sorted(agg.items()):
            summary['paste'][f'a{a}_s{s}'] = dict(retention=float(np.median(v['ret'])),
                                                  direction=float(np.median(v['dir'])), n=len(v['ret']))
            print(f'    alpha={a} sigma={s}: ret={np.median(v["ret"]):.3f} dir={np.median(v["dir"]):.3f} n={len(v["ret"])}')
    print('\n(方向性判读: 影子实锤需 paired CI>0 且时间纹递增 且人眼 PNG 可见; 结论落文档前必看 eye_png/)')
    json.dump(summary, open(f'{out_dir}/ghost_summary.json', 'w'), indent=1)
    print(f'MERGED -> {out_dir}/ghost_summary.json')


def main():
    ngpu, out_dir, tp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    extra = []
    if os.environ.get('TRANSFORMER_DIR'):
        extra += ['--transformer-dir', os.environ['TRANSFORMER_DIR']]
    if os.environ.get('PASTE_ONLY') == '1':
        extra += ['--paste-only']
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
        procs.append(subprocess.Popen(
            [tp, os.path.join(here, 't4g_ghost_probe.py'),
             '--shard-index', str(g), '--num-shards', str(ngpu), '--out-dir', out_dir] + extra,
            env=env, stdout=None, stderr=subprocess.STDOUT))
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
