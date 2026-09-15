#!/usr/bin/env bash
# 联合训练 (L_id + 增广) —— 200 步全量 kjob 提交入口 (44 号阶段2)。
# 透传铁律同 t4g_corr_launch.sh。默认打印计划, `submit` 才提交。
# 哨兵: grep 'aug-sentinel' (ret_proxy 降/l_rest 平) + grep 't4g-sentinel' (L_id 降/B 闸正常)。
# 前 20 步内任一哨兵异常 -> 杀作业查因 (探针纪律)。

set -euo pipefail

REPO_DIR="giga-world-0"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
# 底座可 env 覆盖 (BASE_TRANSFORMER=pretrain路径 可跳过 round0 直训)
BASE_TRANSFORMER_PATH="${BASE_TRANSFORMER:-/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer}"
ANNO_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno"
ASSETS_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets"

# config/run 可 env 覆盖 (Run A=joint, Run B=wmaponly 消融)
BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.pipeline.t4g_joint_config}"
OUT_ROOT="${OUT_ROOT:-/data/datasets/gagi/eve_v2_outputs/t4g_joint}"
RUN_NAME="${RUN_NAME:-t4g_joint200}"
MAX_STEPS="${MAX_STEPS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"

[[ -f "${PACKED}/config.json" ]] || { echo "缺 packed"; exit 1; }
[[ -f "${BASE_TRANSFORMER_PATH}/diffusion_pytorch_model.safetensors" ]] || { echo "缺底座"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]] || { echo "缺 idx2vid"; exit 1; }
N_ASSETS=$(ls "${ASSETS_DIR}"/*.npz 2>/dev/null | wc -l || echo 0)
[[ "${N_ASSETS}" -ge 60 ]] || { echo "aug_assets 不足 (${N_ASSETS})"; exit 1; }

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
      "TRANSFORMER_MODEL_PATH=${BASE_TRANSFORMER_PATH}" \
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
=========== 联合训练 L_id+增广 · ${MAX_STEPS} 步 (未提交) ===========
  config   = ${BASE_CONFIG_MODULE}
  底座     = round0_ema
  assets   = ${N_ASSETS} 个 | anno = t4g_anno
  batch    = 8x1xGA8=64 | 存档每 ${CHECKPOINT_INTERVAL} 步 (4 档选档防背死)
  提交     : bash $(basename "$0") submit
=====================================================================
EOF
  ;;
esac
