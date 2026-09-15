"""生成 _packidx2vid.json 并 md5 校验 packed 与 clean 目录逐条一致 (方案 Phase 1)。

权威顺序 = sorted(clean stems) (pack_data.py 的 sorted-glob 词典序)。
校验: agibot_ewm_packed/videos/data/{i}.mp4 的 md5 == agibot_ewm_clean/{stem_i}.mp4。
一致则现有 packed 可直接复用; 任一不合即报错, 需从 clean 目录重打包。
"""

import argparse
import hashlib
import json
import os
import random

CLEAN = '/data/datasets/gagi/agibot_ewm_clean'
PACKED = '/data/datasets/gagi/agibot_ewm_packed'
ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/agibot_t4g_probe/t4g_anno'


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', default=CLEAN)
    ap.add_argument('--packed', default=PACKED)
    ap.add_argument('--anno-dir', default=ANNO_DIR)
    ap.add_argument('--sample', type=int, default=40,
                    help='md5 抽查条数 (0=全量)')
    args = ap.parse_args()

    stems = sorted(
        f[:-4] for f in os.listdir(args.clean)
        if f.endswith('.mp4') and not f.endswith('_trans.mp4')
    )
    assert len(stems) == 777, len(stems)

    packed_dir = os.path.join(args.packed, 'videos', 'data')
    packed_n = len([f for f in os.listdir(packed_dir) if f.endswith('.mp4')])
    assert packed_n == 777, f'packed data_size={packed_n} != 777'

    idxs = list(range(777))
    if args.sample and args.sample < 777:
        rng = random.Random(0)
        idxs = sorted(rng.sample(idxs, args.sample))
        # 首尾必查 (顺序错位最先在两端暴露)
        for forced in (0, 776):
            if forced not in idxs:
                idxs.append(forced)
        idxs = sorted(set(idxs))
    bad = []
    for i in idxs:
        a = md5(os.path.join(packed_dir, f'{i}.mp4'))
        b = md5(os.path.join(args.clean, f'{stems[i]}.mp4'))
        if a != b:
            bad.append((i, stems[i]))
    if bad:
        raise SystemExit(f'[a2] md5 mismatch {len(bad)}/{len(idxs)}: {bad[:5]} → 需从 clean 重打包')

    os.makedirs(args.anno_dir, exist_ok=True)
    out = os.path.join(args.anno_dir, '_packidx2vid.json')
    with open(out, 'w') as f:
        json.dump({str(i): s for i, s in enumerate(stems)}, f, indent=0)
    print(f'[a2] ok: md5 checked {len(idxs)}/777, idx2vid → {out}')


if __name__ == '__main__':
    main()
