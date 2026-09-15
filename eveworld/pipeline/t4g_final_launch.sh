#!/usr/bin/env bash
# 终局合体臂 (72 号) —— kjob 提交入口。
# 正式: 400 步/每 50 存档, RUN_NAME=t4g_final_s400。
# smoke: `SMOKE=1 bash ... submit` -> 4 步/每 2 存档/T4G_A2_P=0.5 (S1/S2/S3/S6),
#        RUN_NAME=t4g_final_smoke, 独立 OUT_ROOT 子目录, 不污染正式档。
# K 最终事实 = T4G_A2_K (由 kprobe 判决填入)。
#
# 哨兵:
#   grep 'a2-sentinel'  : sft/a2 配比, a2_hit, l_a2, m_sft/m_a2 (R1: m 骤降=报警)
#   grep 't4g-sentinel' : L_id 裸值/生效帧 (corr 组件)
#   grep 't4g-final'    : ICH-D 装载/hook 挂载

set -euo pipefail

REPO_DIR="giga-world-0"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
PRETRAIN_TRANSFORMER="/data/datasets/gagi/giga_world_0_video_pretrain/transformer"
ANNO_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno"
ICH_D_WEIGHTS="/data/datasets/gagi/eve_v2_outputs/selfcase/cic_match/ich_d_frozen_lr.npz"

BASE_CONFIG_MODULE="eveworld.pipeline.t4g_final_config"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  OUT_ROOT="/data/datasets/gagi/eve_v2_outputs/t4g_final/smoke"
  RUN_NAME="${RUN_NAME:-t4g_final_smoke}"
  MAX_STEPS="${MAX_STEPS:-4}"
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2}"
  A2_P="${A2_P:-0.5}"                 # smoke: 抬高 A2 占比, 两分支必现
else
  OUT_ROOT="/data/datasets/gagi/eve_v2_outputs/t4g_final"
  RUN_NAME="${RUN_NAME:-t4g_final_s400}"
  MAX_STEPS="${MAX_STEPS:-400}"
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
  A2_P="${A2_P:-0.25}"
fi
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
A2_K="${A2_K:-4}"                     # kprobe 判决值; 提交前必须确认

# ---- 预检 ----
[[ -f "${PACKED}/config.json" ]]                                          || { echo "缺 ${PACKED}/config.json"; exit 1; }
[[ -f "${PRETRAIN_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]]    || { echo "缺 pretrain 底座"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                                  || { echo "缺 idx2vid 映射"; exit 1; }
[[ -f "${ICH_D_WEIGHTS}" ]]                                               || { echo "缺固化 LR"; exit 1; }

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
      "TRANSFORMER_MODEL_PATH=${PRETRAIN_TRANSFORMER}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8" \
      "T4G_A2_K=${A2_K}" \
      "T4G_A2_P=${A2_P}"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
========== 终局合体臂 (未提交, SMOKE=${SMOKE}) ==========
  config     = ${BASE_CONFIG_MODULE}
  底座       = ${PRETRAIN_TRANSFORMER} (pretrain)
  组件       = SFT + L_id(0.5) + ICH-D固化头 + A2(p=${A2_P}, K=${A2_K})
  batch      = 8x1xGA8=64, steps=${MAX_STEPS}, 每 ${CHECKPOINT_INTERVAL} 步存档
  提交       : [SMOKE=1] bash $(basename "$0") submit
  哨兵       : grep -E 'a2-sentinel|t4g-sentinel' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
=========================================================
EOF
  ;;
esac
