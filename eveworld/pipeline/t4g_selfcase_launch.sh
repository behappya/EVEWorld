#!/usr/bin/env bash
# SELF-CASE 自病例修复训练 —— kjob 提交入口 (70 号 A1)。
# 与 CP 探针受控对照: 同底座(round0_ema)/同预算逻辑, 唯一差异=病例来源(自挖掘 vs 人工贴)。
# 三层透传铁律同 t4g_aug_launch.sh。默认只打印计划, `submit` 才提交。
#
# 哨兵 (grep 'aug-sentinel'):
#   applied    病例样本占比 (≈ p_case × 病例覆盖率)
#   ret_proxy  复制品保留代理, 应从 ~0.9+ 下降   <- 主判据
#   l_rest     病例区外误差, 应平 (不换病)
#   param_disp 参数位移健康

set -euo pipefail

REPO_DIR="giga-world-0"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
OUT_ROOT="/data/datasets/gagi/eve_v2_outputs/t4g_selfcase"
ROUND0_EMA_TRANSFORMER="/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno"
CASE_DIR="/data/datasets/gagi/eve_v2_outputs/selfcase/mine_round0/case_bank"

BASE_CONFIG_MODULE="eveworld.pipeline.t4g_selfcase_config"
RUN_NAME="${RUN_NAME:-t4g_selfcase_s250}"
MAX_STEPS="${MAX_STEPS:-250}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"

# ---- 预检 ----
[[ -f "${PACKED}/config.json" ]]                                   || { echo "缺 ${PACKED}/config.json"; exit 1; }
[[ -f "${ROUND0_EMA_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || { echo "缺底座"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                          || { echo "缺 idx2vid 映射"; exit 1; }
N_CASES=$(ls "${CASE_DIR}"/*.npz 2>/dev/null | wc -l || echo 0)
[[ "${N_CASES}" -ge 8 ]] || { echo "case_bank 只有 ${N_CASES} 个 (<8), 先跑 selfcase_mine_kjob.sh"; exit 1; }

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
=========== SELF-CASE 自病例修复训练 (未提交) ===========
  config     = ${BASE_CONFIG_MODULE}
  底座       = ${ROUND0_EMA_TRANSFORMER}
  case_bank  = ${CASE_DIR} (${N_CASES} 个)
  batch      = 8x1xGA8=64, steps=${MAX_STEPS}, 每 ${CHECKPOINT_INTERVAL} 步存档
  提交       : bash $(basename "$0") submit
  哨兵       : grep 'aug-sentinel' ${TRAIN_PROJECT_DIR}/logs/train_*.log
=========================================================
EOF
  ;;
esac
