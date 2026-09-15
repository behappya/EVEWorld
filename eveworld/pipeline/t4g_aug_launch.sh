#!/usr/bin/env bash
# 合成复制增广 —— 50 步探针 kjob 提交入口 (44 号阶段1')。
# 三层透传铁律同 t4g_corr_launch.sh: BASE_CONFIG_MODULE/TRANSFORMER_MODEL_PATH/
# CHECKPOINT_TOTAL_LIMIT 必须走尾随 KEY=VALUE。默认只打印计划, `submit` 才提交。
#
# 哨兵 (grep 'aug-sentinel'):
#   applied      投毒样本占比 (~50%)
#   l_paste      贴入区对干净目标的裸误差, 应随步降
#   delta        贴入幅度 (恒定参照)
#   ret_proxy    sqrt(l_paste/delta) ≈ retention, 应从 ~0.9+ 下降  <- 主判据
#   l_rest       非贴入区误差, 应平 (不换病)
#   param_disp   参数位移健康
# gate: ret_proxy 显著降 + l_rest 平 + 训后 E10 复测(held-out) + 人眼假货消失真物完好。

set -euo pipefail

REPO_DIR="giga-world-0"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
OUT_ROOT="/data/datasets/gagi/eve_v2_outputs/t4g_aug"
ROUND0_EMA_TRANSFORMER="/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno"
ASSETS_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets"

BASE_CONFIG_MODULE="eveworld.pipeline.t4g_aug_config"
RUN_NAME="${RUN_NAME:-t4g_aug_probe50}"
PROBE_STEPS="${PROBE_STEPS:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"

# ---- 预检 ----
[[ -f "${PACKED}/config.json" ]]                                   || { echo "缺 ${PACKED}/config.json"; exit 1; }
[[ -f "${ROUND0_EMA_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || { echo "缺底座"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                          || { echo "缺 idx2vid 映射"; exit 1; }
N_ASSETS=$(ls "${ASSETS_DIR}"/*.npz 2>/dev/null | wc -l || echo 0)
[[ "${N_ASSETS}" -ge 60 ]] || { echo "aug_assets 只有 ${N_ASSETS} 个 (<60), 先跑 t4g_aug_prep_kjob.sh"; exit 1; }

TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments"

run_submit() {
  cd "${REPO_DIR}"
  mkdir -p "${OUT_ROOT}"
  PACKED_DATA_DIR="${PACKED}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  MAX_STEPS="${PROBE_STEPS}" \
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
      "MAX_STEPS=${PROBE_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
=========== 合成复制增广 · 50 步探针 (未提交) ===========
  config     = ${BASE_CONFIG_MODULE}
  底座       = ${ROUND0_EMA_TRANSFORMER}
  assets     = ${ASSETS_DIR} (${N_ASSETS} 个)
  batch      = 8x1xGA8=64, steps=${PROBE_STEPS}
  提交       : bash $(basename "$0") submit
  哨兵       : grep 'aug-sentinel' ${TRAIN_PROJECT_DIR}/logs/train_*.log
=========================================================
EOF
  ;;
esac
