#!/usr/bin/env bash
set -uo pipefail
# EVE · 轨B best-of-N: 提交 N 个不同 seed 的生成 job(复用 generate_eag, 默认 w=0 普通生成)。
# 生成完对每个 seed 目录跑 eveworld/evaluation/tea/qwen_laziness.py, 再用 best_of_n_select.py 选优。
#
# 用法:
#   SEEDS="6666 1234 2025 777" LIMIT=16 bash eveworld/method/scripts/launch_bestofn_generate_kjob.sh
#   NUM_FRAMES=125 SEEDS="6666 1234 2025" bash ...     # 长视频 best-of-N
#   EAG_WEIGHT=0.03 SEEDS="..." bash ...               # best-of-N 叠加 EAG(两轨结合)
#   DRY_RUN=1 bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_eve_eag_generate.sh}"

GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
BON_ROOT="${BON_ROOT:-${GAGI}/eve_outputs/bestofn}"
# 全量放大: 8 seed(N=8, 支持 N=2/4/8 scaling 消融) + LIMIT=0(全92条)。
# ★ 冒烟才用 LIMIT=16; 论文全量必须 LIMIT=0。
SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
NUM_FRAMES="${NUM_FRAMES:-93}"
LIMIT="${LIMIT:-0}"
EAG_WEIGHT="${EAG_WEIGHT:-0}"       # 0=普通生成候选; >0=候选也带EAG(两轨结合)

echo "[EVE] best-of-N 生成. seeds=[${SEEDS}] frames=${NUM_FRAMES} limit=${LIMIT} eag_w=${EAG_WEIGHT}"
tag_wt=$([[ "${EAG_WEIGHT}" != "0" ]] && echo "_eag${EAG_WEIGHT}" || echo "")

for sd in ${SEEDS}; do
  save="${BON_ROOT}/seed${sd}${tag_wt}_f${NUM_FRAMES}"
  echo "----- 提交 seed=${sd} -> ${save} -----"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] EAG_WEIGHT=${EAG_WEIGHT} SEED=${sd} SAVE_DIR=${save} NUM_FRAMES=${NUM_FRAMES} LIMIT=${LIMIT}"
    continue
  fi
  ( "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
      "REPO_DIR=${REPO_DIR}" "GAGI_ROOT=${GAGI}" \
      "DATA_PATH=${DATA_PATH}" "LAM=${LAM}" "SAVE_DIR=${save}" \
      "EAG_WEIGHT=${EAG_WEIGHT}" "NUM_FRAMES=${NUM_FRAMES}" "SEED=${sd}" "LIMIT=${LIMIT}" ) &
  sleep 8
done
wait
echo ""
echo "[EVE] 全部 seed 提交完成。生成完后按 seed 跑 Qwen 打分, 再选优:"
echo "  # 1) 每个 seed 目录跑 Qwen:"
echo "  for sd in ${SEEDS}; do"
echo "    python eveworld/evaluation/tea/qwen_laziness.py --video-dir ${BON_ROOT}/seed\${sd}${tag_wt}_f${NUM_FRAMES}/generated_only \\"
echo "      --run-name bon_seed\${sd} --concurrency 64 --limit ${LIMIT}; done"
echo "  # 2) 选优(--cand-dirs 与 --qwen-csvs 按 seed 顺序一一对应):"
echo "  python eveworld/method/scripts/best_of_n_select.py --cand-dirs <各seed generated_only> \\"
echo "    --qwen-csvs <各seed csv> --seeds ${SEEDS} --out ${BON_ROOT}/selection.json"
