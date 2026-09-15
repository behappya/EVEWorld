#!/usr/bin/env python3
"""X8: 组装 lazy13 统一对比文件夹（幂等，可反复运行补齐新完成的臂）。

结构: compare_lazy13/<臂>/seed<S>/<全局编号_任务名>.mp4
臂: gr1_download(2b推理) / round0 / anmix_s50 / anmix_s200 / anmixv3_ep16..20
文件名以 2b 推理输出的全局编号命名为准, 子集(0..12)编号的产物按 mapping 重命名。
"""
import json
import shutil
from glob import glob
from pathlib import Path

GAGI = Path('/data/datasets/gagi/eve_v2_outputs')
CMP = GAGI / 'compare_lazy13'
MAP = json.load(open(GAGI / 'probe/anmixv3_epscan_lazy13_mapping.json'))
SEEDS = ['6666', '1234', '2025', '777', '42', '314', '2718', '999']
TARGET_IDX = [m['orig_idx'] for m in MAP]

# 全局编号 -> 规范文件名（取自 2b 推理输出）
canon = {}
ref = GAGI / 'probe/gr1_2b_dreamgen/seed42_f93/generated_only'
for i in TARGET_IDX:
    hits = sorted(glob(str(ref / f'{i}_*.mp4')))
    assert hits, f'2b 输出缺 idx {i}'
    canon[i] = Path(hits[0]).name

# (臂名, 源目录模板, 编号体系: global|subset)
ARMS = [
    ('gr1_download', GAGI / 'probe/gr1_2b_dreamgen/seed{S}_f93/generated_only', 'global'),
    ('round0',       GAGI / 'pool_round0_f93/seed{S}_f93/generated_only',       'global'),
    ('anmix_s50',    GAGI / 'probe/anmixv1_lazy13/anmix_s50/seed{S}_f93/generated_only',  'subset'),
    ('anmix_s200',   GAGI / 'probe/anmixv1_lazy13/anmix_s200/seed{S}_f93/generated_only', 'subset'),
] + [
    (f'anmixv3_ep{ep}', GAGI / ('probe/anmixv3_epscan_lazy13/ep%d/seed{S}_f93/generated_only' % ep), 'subset')
    for ep in (16, 17, 18, 19)  # ep20 经用户裁决弃用 (2026-07-19)
]

total_missing = 0
for arm, tmpl, scheme in ARMS:
    done = 0
    missing = []
    for S in SEEDS:
        src_dir = Path(str(tmpl).replace('{S}', S))
        dst_dir = CMP / arm / f'seed{S}'
        dst_dir.mkdir(parents=True, exist_ok=True)
        for m in MAP:
            gi, si = m['orig_idx'], m['new_idx']
            dst = dst_dir / canon[gi]
            if dst.exists():
                done += 1
                continue
            pat = f'{gi}_*.mp4' if scheme == 'global' else f'{si}_*.mp4'
            hits = sorted(glob(str(src_dir / pat)))
            if hits:
                shutil.copy2(hits[0], dst)
                done += 1
            else:
                missing.append(f'seed{S}:idx{gi}')
    total_missing += len(missing)
    tag = 'OK' if not missing else f'缺 {len(missing)} (如 {missing[:3]})'
    print(f'{arm:16s} {done}/104  {tag}')

(CMP / 'mapping.json').write_text(json.dumps(MAP, ensure_ascii=False, indent=1))
print(f'\n-> {CMP}  (缺口 {total_missing}, 对应臂的生成完成后重跑本脚本即可补齐)')
