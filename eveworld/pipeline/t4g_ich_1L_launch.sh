#!/usr/bin/env bash
# ICH 训练 · 臂 A (单层 block16) —— kjob 提交入口 (71 号 B1 G3, 50 步探针)。
# 与臂 B (t4g_ich_3L_launch.sh) 唯一差异 = config (t4g_ich_blocks)。
# 三层透传铁律同 t4g_selfcase_launch.sh。默认只打印计划, `submit` 才提交。
#
# 哨兵:
#   grep 'ich-sentinel' : det_pos/det_neg 应分离 (病例格/背景格 M 均值), gate/wout_norm
#   grep 'aug-sentinel' : ret_proxy 应从 ~0.9+ 下降 (A1 主判据), l_rest 应平

set -euo pipefail

REPO_DIR="giga-world-0"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
OUT_ROOT="/data/datasets/gagi/eve_v2_outputs/t4g_ich_1L"
ROUND0_EMA_TRANSFORMER="/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno"
CASE_DIR="/data/datasets/gagi/eve_v2_outputs/selfcase/case_bank_all"

BASE_CONFIG_MODULE="eveworld.pipeline.t4g_ich_1L_config"
RUN_NAME="${RUN_NAME:-t4g_ich_1L_s50}"
MAX_STEPS="${MAX_STEPS:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"

# ---- 预检 ----
[[ -f "${PACKED}/config.json" ]]                                   || { echo "缺 ${PACKED}/config.json"; exit 1; }
[[ -f "${ROUND0_EMA_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || { echo "缺底座"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                          || { echo "缺 idx2vid 映射"; exit 1; }
N_CASES=$(ls "${CASE_DIR}"/*.npz 2>/dev/null | wc -l || echo 0)
[[ "${N_CASES}" -ge 30 ]] || { echo "case_bank_all 只有 ${N_CASES} 个 (<30), 先跑 t4g_ich_case_merge.py"; exit 1; }

TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments"

run_submit() {
  cd "${REPO_DIR}"
  mkdir -p "${OUT_ROOT}"
  PACKED_DATA_DIR="${PACKED}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  MAX_STEPS="${MAX_STEPS}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  BATCH_SIZE_PER_GPU=1 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
    ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
      "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}" \
      "TRANSFORMER_MODEL_PATH=${ROUND0_EMA_TRANSFORMER}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
=========== ICH 臂 A (block16 单层) 50 步训练 (未提交) ===========
  config     = ${BASE_CONFIG_MODULE}
  底座       = ${ROUND0_EMA_TRANSFORMER}
  case_bank  = ${CASE_DIR} (${N_CASES} 个)
  batch      = 8x1xGA8=64, steps=${MAX_STEPS}, 每 ${CHECKPOINT_INTERVAL} 步存档
  提交       : bash $(basename "$0") submit
  哨兵       : grep -E 'ich-sentinel|aug-sentinel' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
=================================================================
EOF
  ;;
esac
