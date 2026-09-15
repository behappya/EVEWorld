#!/usr/bin/env python3
"""汇总病例库 (ICH 训练用): 五池 case_bank 的 status=='ok' 病例 npz 软链到
/data/.../selfcase/case_bank_all/ (40 例 = 四池 34 + mine_pretrain 6)。

命名: {vid}__{pool}_{seed_tag}.npz —— 保留 `vid__` 前缀 (T4GSelfCaseTransform 的
case_index 键 = 文件名 split('__')[0]), 池前缀防跨池撞名 (如 3__s1234 同时在
mine_round0 与 mine_gr1_2b)。CPU 本机可跑, 幂等 (--force 重建)。
"""
import argparse
import json
import os

POOL_ROOT = '/data/datasets/gagi/eve_v2_outputs/selfcase'
POOLS = ['mine_round0', 'mine_gr1_2b', 'mine_s150', 'mine_wmapA_pre_s250', 'mine_pretrain']
OUT_DEFAULT = os.path.join(POOL_ROOT, 'case_bank_all')
IDX2VID = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno/_packidx2vid.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--expect', type=int, default=40)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    known_vids = set(json.load(open(IDX2VID)).values()) if os.path.exists(IDX2VID) else set()

    total, unmatched = 0, []
    for pool in POOLS:
        report = os.path.join(POOL_ROOT, pool, 'case_bank_report.json')
        bank = os.path.join(POOL_ROOT, pool, 'case_bank')
        n = 0
        for r in json.load(open(report))['records']:
            if r.get('status') != 'ok':
                continue
            case_id = r['case']['case_id']
            src = os.path.join(bank, case_id + '.npz')
            assert os.path.exists(src), f'缺 npz: {src}'
            vid, rest = case_id.split('__', 1)
            dst = os.path.join(args.out, f'{vid}__{pool}_{rest}.npz')
            if os.path.lexists(dst):
                if not args.force and os.path.realpath(dst) == os.path.realpath(src):
                    n += 1
                    continue
                os.remove(dst)
            os.symlink(src, dst)
            n += 1
            if known_vids and vid not in known_vids:
                unmatched.append(f'{pool}:{case_id}')
        total += n
        print(f'[case-merge] {pool}: {n} ok cases')

    links = sorted(f for f in os.listdir(args.out) if f.endswith('.npz'))
    vids = sorted({f.split('__')[0] for f in links})
    print(f'[case-merge] -> {args.out}: {len(links)} npz / {len(vids)} vids '
          f'(vids={vids})')
    if unmatched:
        print(f'[case-merge] WARN: {len(unmatched)} 例 vid 不在 idx2vid 映射内 '
              f'(训练时不会被抽中): {unmatched}')
    assert total == args.expect, f'期望 {args.expect} 例, 实得 {total}'
    assert len(links) == args.expect, f'目录残留旧链: {len(links)} != {args.expect}'
    print('CASE_MERGE_OK')


if __name__ == '__main__':
    main()
