#!/usr/bin/env python3
"""gr92 重考池 -> Gemini qwen_if 协议 manifest (jsonl)。

复用 eval_gemini_dreamgen_qwen_protocol.py 的输入契约(key/video_path/prompt 必填,
split 进聚合)。gr92 池 = 92 训练条件 x 4 协议 seed, 文件名前缀=全局 request_id
(分片生成的池必须先跑 selfcase_pool_fix_names.py)。split 记 seed 便于分 seed 聚合。

用法: python t4g_gr92_gemini_manifest.py <pool_root> <model_name> <out.jsonl>
"""
import json
import os
import sys
from glob import glob

IT2V = '/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json'


def main():
    pool_root, model_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    it2v = {int(str(r['request_id']).split('_')[0]): r for r in json.load(open(IT2V))}
    rows = []
    for seed_dir in sorted(glob(os.path.join(pool_root, 'seed*_f93'))):
        seed = os.path.basename(seed_dir).replace('seed', '').replace('_f93', '')
        for mp4 in sorted(glob(os.path.join(seed_dir, 'generated_only', '*.mp4'))):
            idx = int(os.path.basename(mp4).split('_')[0])
            meta = it2v.get(idx)
            if meta is None:
                raise SystemExit(f'idx {idx} 不在 it2v: {mp4}')
            rows.append(dict(
                key=f'{model_name}/seed{seed}/{idx}',
                sample_key=f'gr92/{idx}',
                model=model_name,
                inference_seed=int(seed),
                split=f'seed{seed}',
                index=idx,
                request_id=str(meta['request_id']),
                prompt=meta['prompt'],
                video_path=mp4,
            ))
    assert len(rows) == 368, f'期望 368 行, 实际 {len(rows)}'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'[manifest] {model_name}: {len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()
