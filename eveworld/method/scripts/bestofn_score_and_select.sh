#!/usr/bin/env bash
set -uo pipefail
# EVE · 轨B best-of-N 后处理: 对已生成的 N 个 seed 目录, 跑【裁判A(选优)+裁判B(独立评测)】
# 再做【选优/评测分离】分析(打破循环). 生成阶段用 launch_bestofn_generate_kjob.sh 先跑完。
#
# ★ 为什么两个裁判(方案27 §二十): best-of-N 若用同一裁判选优又汇报, 取N个带噪测量最小值
#   天然低于均值(赢家诅咒)。用裁判A选、独立裁判B(不同问题分解+不同抽帧)评被选视频, 才能
#   证明改善是真降语义偷懒、非拟合裁判A噪声。
#
# 用法:
#   SEEDS="6666 1234 2025 777 42 314 2718 999" LIMIT=0 \
#     bash eveworld/method/scripts/bestofn_score_and_select.sh
#   PY=<有openai+cv2的python> 覆盖评测环境(默认 dreamgenbench_eval_venv)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
BON="${BON_ROOT:-${GAGI}/eve_outputs/bestofn}"
TQ="${TQ_ROOT:-${GAGI}/eve_outputs/tea_qwen}"
SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
NUM_FRAMES="${NUM_FRAMES:-93}"
LIMIT="${LIMIT:-0}"
CONC="${CONCURRENCY:-64}"
PY="${PY:-/data/datasets/gagi/envs/dreamgenbench_eval_venv/bin/python}"
QWEN_BASE="${QWEN_BASE:-127.0.0.1}"

echo "[bon] seeds=[${SEEDS}] frames=${NUM_FRAMES} limit=${LIMIT} py=${PY}"

sel_csvs=(); eval_csvs=(); cand_dirs=(); present_seeds=()
for sd in ${SEEDS}; do
  gen="${BON}/seed${sd}_f${NUM_FRAMES}/generated_only"
  if [[ ! -d "${gen}" ]]; then echo "[skip] 缺 ${gen}"; continue; fi
  cand_dirs+=("${gen}"); present_seeds+=("${sd}")
  a_csv="${TQ}/bon_seed${sd}_laziness.csv"
  b_csv="${TQ}/bon_seed${sd}_B_laziness.csv"
  sel_csvs+=("${a_csv}"); eval_csvs+=("${b_csv}")
  # 裁判A(5签名, 选优); resume 默认开, 已评的跳过
  echo "----- seed=${sd} 裁判A -----"
  "${PY}" eveworld/evaluation/tea/qwen_laziness.py --video-dir "${gen}" --run-name "bon_seed${sd}" \
    --judge a --qwen-base "${QWEN_BASE}" --concurrency "${CONC}" --limit "${LIMIT}" \
    2>&1 | grep -E "mean_severity|errors|completed [0-9]+/[0-9]+$" | tail -3
  # 裁判B(独立: 过程完整度分解 + 抽帧相位0.5)
  echo "----- seed=${sd} 裁判B(独立) -----"
  "${PY}" eveworld/evaluation/tea/qwen_laziness.py --video-dir "${gen}" --run-name "bon_seed${sd}_B" \
    --judge b --frame-offset 0.5 --qwen-base "${QWEN_BASE}" --concurrency "${CONC}" --limit "${LIMIT}" \
    2>&1 | grep -E "mean_severity|errors|completed [0-9]+/[0-9]+$" | tail -3
done

echo ""
echo "[bon] === 选优/评测分离分析 (N=${#cand_dirs[@]}) ==="
"${PY}" eveworld/method/scripts/best_of_n_select.py \
  --cand-dirs "${cand_dirs[@]}" \
  --sel-csvs  "${sel_csvs[@]}" \
  --eval-csvs "${eval_csvs[@]}" \
  --seeds "${present_seeds[@]}" --scaling \
  --out "${BON}/selection_separated_f${NUM_FRAMES}.json"

echo ""
echo "[bon] 完成。结果: ${BON}/selection_separated_f${NUM_FRAMES}.json"
echo "  论文主报: reduction_B(独立裁判确认的降幅) + scaling(N=2/4/8) + circularity_gap(去循环证据)"
