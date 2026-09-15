#!/usr/bin/env python3
"""SELF-CASE 挖掘编排 (70 号 A1): 单节点 8 卡, 三段串行。

  1) 建清单 TSV: pool_round0_f93 92x8 -> (arm, video, prompt, cond_img)
  2) t4g_exam_v3 8 分片体检 -> 合并 -> dup 名单
  3) selfcase_build 8 分片构建病例 -> 合并 -> case bank + 报告

用法: python selfcase_mine_dispatch.py <num_gpus> <out_dir> <worker_python>
env:  POOL_DIR / IT2V / CASE_DIR / PREVIEW_DIR 可覆盖
"""
import glob
import json
import os
import subprocess
import sys

POOL_DIR = os.environ.get(
    'POOL_DIR', '/data/datasets/gagi/eve_v2_outputs/pool_round0_f93')
IT2V = os.environ.get(
    'IT2V', '/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json')


def run_shards(cmd_tpl, n, tag):
    procs = []
    for i in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        cmd = [c.format(i=i) for c in cmd_tpl]
        print(f'[dispatch] {tag} shard{i}: {" ".join(cmd)}', flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    bad = [i for i, p in enumerate(procs) if p.wait() != 0]
    if bad:
        raise RuntimeError(f'{tag} shards failed: {bad}')


def main():
    n = int(sys.argv[1])
    out_dir = sys.argv[2]
    py = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    case_dir = os.environ.get('CASE_DIR', os.path.join(out_dir, 'case_bank'))
    preview_dir = os.environ.get('PREVIEW_DIR', os.path.join(out_dir, 'previews'))

    # ---- 1) TSV ----
    it2v = {int(str(r['request_id']).split('_')[0]): r for r in json.load(open(IT2V))}
    tsv = os.path.join(out_dir, 'selfcase_gr92.tsv')
    n_rows = 0
    with open(tsv, 'w') as f:
        for seed_dir in sorted(glob.glob(os.path.join(POOL_DIR, 'seed*_f93'))):
            seed = os.path.basename(seed_dir).replace('seed', '').replace('_f93', '')
            for mp4 in sorted(glob.glob(os.path.join(seed_dir, 'generated_only', '*.mp4'))):
                idx = int(os.path.basename(mp4).split('_')[0])
                meta = it2v.get(idx)
                if meta is None:
                    continue
                f.write(f"round0_s{seed}\t{mp4}\t{meta['prompt']}\t{meta['image']}\n")
                n_rows += 1
    print(f'[dispatch] TSV rows={n_rows} -> {tsv}', flush=True)
    if n_rows == 0:
        raise RuntimeError('empty TSV')

    # ---- 2) exam v3 (SKIP_EXAM=1 且已有合并结果时直接复用) ----
    exam_json = os.path.join(out_dir, 'exam_v3_round0_gr92.json')
    if os.environ.get('SKIP_EXAM') == '1' and os.path.exists(exam_json):
        merged = json.load(open(exam_json))
        print(f'[dispatch] SKIP_EXAM: reuse {exam_json} ({len(merged)} videos)', flush=True)
    else:
        run_shards([py, 't4g_exam_v3.py', '--list-file', tsv,
                    '--shard-index', '{i}', '--num-shards', str(n),
                    '--out', os.path.join(out_dir, 'exam_shard{i}.json')], n, 'exam')
        merged = []
        for i in range(n):
            merged += json.load(open(os.path.join(out_dir, f'exam_shard{i}.json')))
        for rec in merged:   # exam v3 只存 basename, 且各 seed 同名 -> 由 arm 回填全路径
            seed = rec.get('arm', '').split('_s')[-1]
            rec['video'] = os.path.join(POOL_DIR, f'seed{seed}_f93', 'generated_only',
                                        rec['video'])
        json.dump(merged, open(exam_json, 'w'))
    dup = [r for r in merged if r.get('dup')]
    und = sum(1 for r in merged if r.get('undetectable'))
    print(f'[dispatch] exam done: {len(merged)} videos, dup={len(dup)}, '
          f'undetectable={und}, dup_rate={100.0 * len(dup) / max(1, len(merged) - und):.2f}%',
          flush=True)

    # ---- 3) build ----
    run_shards([py, 'selfcase_build.py', '--exam-json', exam_json, '--it2v', IT2V,
                '--out-dir', case_dir, '--preview-dir', preview_dir,
                '--shard-index', '{i}', '--num-shards', str(n),
                '--out', os.path.join(out_dir, 'build_shard{i}.json')], n, 'build')
    recs = []
    for i in range(n):
        recs += json.load(open(os.path.join(out_dir, f'build_shard{i}.json')))
    ok = [r for r in recs if r['status'] == 'ok']
    by_status, by_vid = {}, {}
    for r in recs:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
    for r in ok:
        by_vid[r['gt_vid']] = by_vid.get(r['gt_vid'], 0) + 1
    report = dict(pool=POOL_DIR, videos=len(merged), dup=len(dup),
                  cases=len(ok), case_vids=len(by_vid), by_status=by_status,
                  by_vid=by_vid, records=recs)
    json.dump(report, open(os.path.join(out_dir, 'case_bank_report.json'), 'w'), indent=1)
    print(f'[dispatch] CASE BANK: {len(ok)} cases over {len(by_vid)} conditions; '
          f'status={by_status}', flush=True)
    print('MINE_DONE', flush=True)


if __name__ == '__main__':
    main()
