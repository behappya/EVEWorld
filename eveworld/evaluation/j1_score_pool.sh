#!/bin/bash
# J1: qwen 双裁判批量判分（HTTP 并发, 跑在 workspace, 不占训练卡）。
# 用法:
#   bash j1_score_pool.sh longpool     # 判 L0 长档池 (gr1_sft_ema f125, 8 seed)
#   bash j1_score_pool.sh round0       # 判 T1 Round-0 池 (f93, 8 seed) — T1 完成后跑
# 每个 seed 目录出 judge B (主) 与 judge A (选择用) 两份分数, 断点续跑(--resume 默认开)。
set -eu
source /home/jovyan/miniconda/etc/profile.d/conda.sh && conda activate "${CONDA_ENV:-EVEWorld}"
REPO=giga-world-0
QWEN_BASE="${QWEN_BASE:-127.0.0.1}"   # Qwen3.6-35B-A3B @ 8000, 2026-07-18 新起
POOL="${1:-longpool}"

case "$POOL" in
  longpool)
    ROOT=/data/datasets/gagi/eve_v2_outputs/longpool_f125/gr1_sft_ema
    TAG=longpool_f125 ; SUFFIX=f125 ;;
  round0)
    ROOT=/data/datasets/gagi/eve_v2_outputs/pool_round0_f93
    TAG=pool_round0_f93 ; SUFFIX=f93 ;;
  *) echo "未知池: $POOL (longpool|round0)"; exit 1 ;;
esac
OUT=/data/datasets/gagi/eve_v2_outputs/scores/$TAG
mkdir -p "$OUT"

cd "$REPO"
for seed in 6666 1234 2025 777 42 314 2718 999; do
  VDIR="$ROOT/seed${seed}_${SUFFIX}/generated_only"
  [[ -d "$VDIR" ]] || { echo "跳过(目录缺): $VDIR"; continue; }
  n=$(ls "$VDIR"/*.mp4 2>/dev/null | wc -l)
  echo "== seed $seed ($n 条) judge B =="
  python eveworld/evaluation/tea/qwen_laziness.py \
    --video-dir "$VDIR" --out-root "$OUT" --run-name "seed${seed}_B" \
    --qwen-base "$QWEN_BASE" --judge b --frame-offset 0.5 --concurrency 96
  echo "== seed $seed judge A =="
  python eveworld/evaluation/tea/qwen_laziness.py \
    --video-dir "$VDIR" --out-root "$OUT" --run-name "seed${seed}_A" \
    --qwen-base "$QWEN_BASE" --judge a --concurrency 96
done
echo ""
echo "全部完成: $OUT/seed*_{A,B}* ; 下一步: 汇总统计 + (round0)Gate-2 对比 / (longpool)长档 severity 分布"