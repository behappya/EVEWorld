#!/bin/bash
# P0-probe 第二步: 判分探针产物并与 Round-0 池同 prompt 同 seed 直接对比。
# 用法: bash p0_probe_score.sh <TAG>
set -eu
source /home/jovyan/miniconda/etc/profile.d/conda.sh && conda activate giga_models
REPO=giga-world-0
TAG="${1:?TAG}"
ROOT=/data/datasets/gagi/eve_v2_outputs/probe/${TAG}
OUT=/data/datasets/gagi/eve_v2_outputs/scores/probe_${TAG}
QWEN_BASE="${QWEN_BASE:-127.0.0.1}"
mkdir -p "$OUT"

cd "$REPO"
for seed in 6666 1234 2025 777 42 314 2718 999; do
  VDIR="$ROOT/seed${seed}_f93/generated_only"
  [[ -d "$VDIR" ]] || continue
  python eveworld/evaluation/tea/qwen_laziness.py --video-dir "$VDIR" --out-root "$OUT" \
    --run-name "seed${seed}_B" --qwen-base "$QWEN_BASE" --judge b --frame-offset 0.5 --concurrency 96
done

python3 - "$TAG" <<'EOF'
import csv, glob, json, os, sys
from collections import defaultdict
TAG = sys.argv[1]
def load(d):
    m = defaultdict(dict)
    for f in glob.glob(f'{d}/seed*_B_laziness.csv'):
        seed = os.path.basename(f).split('_')[0].replace('seed','')
        for r in csv.DictReader(open(f)):
            if r['laziness_severity'] and r.get('parsed_ok')=='1':
                m[os.path.basename(r['video_path'])][seed] = float(r['laziness_severity'])
    return m
probe = load(f'/data/datasets/gagi/eve_v2_outputs/scores/probe_{TAG}')
base  = load('/data/datasets/gagi/eve_v2_outputs/scores/pool_round0_f93')
common = sorted(set(probe) & set(base), key=lambda x: int(x.split('_')[0]))
mean = lambda x: sum(x)/len(x)
diffs, p_all, b_all = [], [], []
for pf in common:
    seeds = set(probe[pf]) & set(base[pf])
    if not seeds: continue
    pm, bm = mean([probe[pf][s] for s in seeds]), mean([base[pf][s] for s in seeds])
    diffs.append(pm-bm); p_all.append(pm); b_all.append(bm)
n = len(diffs)
d = mean(diffs)
wins = sum(1 for x in diffs if x < 0); ties = sum(1 for x in diffs if x == 0)
print(f'\n===== 探针裁决: {TAG} vs Round-0 (judge B, {n} 个共同 prompt, 逐 prompt 同 seed 配对) =====')
print(f'{TAG}: {mean(p_all):.3f}  |  Round-0: {mean(b_all):.3f}  |  差值: {d:+.3f} (负=改善)')
print(f'逐 prompt: 改善 {wins} / 持平 {ties} / 变差 {n-wins-ties}')
verdict = '✅ 方向正确' if d < -0.1 and wins >= n/2 else ('⚠️ 无明显变化' if abs(d) <= 0.1 else '❌ 变差, 止损检查')
print(f'裁决: {verdict}  (Gate-3 正式门槛: 全量40eval改善>=0.2 + 人眼可辨)')
json.dump({'tag':TAG,'n':n,'probe_mean':mean(p_all),'round0_mean':mean(b_all),'delta':d,
           'wins':wins,'ties':ties}, open(f'/data/datasets/gagi/eve_v2_outputs/scores/probe_{TAG}/verdict.json','w'), indent=1)
EOF
