#!/usr/bin/env bash
# ICH 决定性一跑 · 臂 B (三层) + 正例加权 —— kjob 提交入口 (71 号 B1 G3 加时赛)。
# 背景: 50 步判决 det 分离仅 +0.01 (warmup 吃掉全程 + 正例 ~2-5% 不平衡),
#       ret_proxy 1L=0.350/3L=0.371 未跑赢 A1 基线 0.282。
# 变更 vs t4g_ich_3L_launch.sh:
#   MAX_STEPS=100 (warmup 仍 50 -> 后 50 步满 λ_det)
#   T4G_DET_POS_WEIGHT=25 (BCE 正例加权, 走 env; trainer env 优先于 config,
#     生效验证 = run.log 里 [t4g-ich] 行 pos_weight=25.0)
#   OUT_ROOT/RUN_NAME 全新目录, 不与旧档混淆。
#
# 哨兵:
#   grep 'ich-sentinel' : step60+ det_pos/det_neg 应拉开 (>0.05 为佳)
#   grep 'aug-sentinel' : ret_proxy 对照 A1@100 步

set -euo pipefail

REPO_DIR="giga-world-0"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
OUT_ROOT="/data/datasets/gagi/eve_v2_outputs/t4g_ich_3L_pw"
ROUND0_EMA_TRANSFORMER="/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno"
CASE_DIR="/data/datasets/gagi/eve_v2_outputs/selfcase/case_bank_all"

BASE_CONFIG_MODULE="eveworld.pipeline.t4g_ich_3L_config"
RUN_NAME="${RUN_NAME:-t4g_ich_3L_pw_s100}"
MAX_STEPS="${MAX_STEPS:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
DET_POS_WEIGHT="${DET_POS_WEIGHT:-25}"
DET_WARMUP="${DET_WARMUP:-50}"

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
      "GRADIENT_ACCUMULATION_STEPS=8" \
      "T4G_DET_POS_WEIGHT=${DET_POS_WEIGHT}" \
      "T4G_DET_WARMUP=${DET_WARMUP}"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
====== ICH 臂 B + pos_weight 决定性一跑 100 步 (未提交) ======
  config       = ${BASE_CONFIG_MODULE}
  底座         = ${ROUND0_EMA_TRANSFORMER}
  case_bank    = ${CASE_DIR} (${N_CASES} 个)
  batch        = 8x1xGA8=64, steps=${MAX_STEPS}, 每 ${CHECKPOINT_INTERVAL} 步存档
  pos_weight   = ${DET_POS_WEIGHT} (env T4G_DET_POS_WEIGHT, warmup=${DET_WARMUP})
  提交         : bash $(basename "$0") submit
  生效验证     : grep 'pos_weight' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log  (应 =25.0)
  哨兵         : grep -E 'ich-sentinel|aug-sentinel' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
=============================================================
EOF
  ;;
esac
