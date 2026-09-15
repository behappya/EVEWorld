#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
PACKED="${GAGI}/gr1_finetune_data/packed_data_t4g_nohuman_v2"
PRETRAIN="${GAGI}/giga_world_0_video_pretrain/transformer"
OUT_ROOT="${GAGI}/eve_v2_outputs/t4g_joint_wmapA_pre_cleanv2_armfixv4_u3"
RUN_NAME="${RUN_NAME:-t4g_wmapA_pre_cleanv2_armfixv4_u3_s150}"
BASE_CONFIG_MODULE="eveworld.pipeline.t4g_joint_cleanv2_config"
MAX_STEPS="${MAX_STEPS:-150}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-3}"
PYTHON="/home/jovyan/miniconda/envs/giga_models/bin/python"

run_check() {
  [[ -f "${PACKED}/config.json" ]] || { echo "missing packed data: ${PACKED}"; exit 1; }
  [[ -f "${PRETRAIN}/diffusion_pytorch_model.safetensors" ]] || { echo "missing pretrain"; exit 1; }
  cd "${REPO_DIR}"
  "${PYTHON}" eveworld/pipeline/t4g_cleanv2_preflight.py
  "${PYTHON}" -m py_compile \
    eveworld/pipeline/t4g_aug_trainer.py \
    eveworld/pipeline/t4g_joint_trainer.py \
    eveworld/pipeline/t4g_joint_cleanv2_config.py
}

run_submit() {
  run_check
  cd "${REPO_DIR}"
  SKIP_TRAIN_ENV_SETUP=1 \
  PACKED_DATA_DIR="${PACKED}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments" \
  RUN_NAME="${RUN_NAME}" \
  MAX_STEPS="${MAX_STEPS}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT}" \
  BATCH_SIZE_PER_GPU=1 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
    ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
      "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}" \
      "TRANSFORMER_MODEL_PATH=${PRETRAIN}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${OUT_ROOT}/experiments" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8"
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  *)
    cat <<EOF
A-pre-clean-v4-u3
  packed: ${PACKED}
  base:   ${PRETRAIN}
  output: ${OUT_ROOT}
  train:  150 steps, checkpoints 50/100/150, effective batch 64
  loss:   L_id=0.5, background=1x, marked/paste=3x

Commands:
  bash $(basename "$0") check
  bash $(basename "$0") submit
EOF
    ;;
esac
