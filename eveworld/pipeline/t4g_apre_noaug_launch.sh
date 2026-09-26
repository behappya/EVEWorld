#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
PACKED="${GAGI}/gr1_finetune_data/packed_data"
PRETRAIN="${GAGI}/giga_world_0_video_pretrain/transformer"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_v2_outputs/t4g_joint_wmapA_pre_noaug}"
RUN_NAME="${RUN_NAME:-t4g_wmapA_pre_noaug_s300}"
# Paper-side twin: eveworld.pipeline.t4g_apre_noaug_b23_config (block 23 layer).
BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.pipeline.t4g_apre_noaug_config}"
MAX_STEPS="${MAX_STEPS:-300}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
SEED="6666"
PYTHON="/home/jovyan/miniconda/envs/EVEWorld/bin/python"

run_check() {
  cd "${REPO_DIR}"
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${PYTHON}" eveworld/pipeline/t4g_apre_noaug_preflight.py
  "${PYTHON}" -m py_compile \
    "${BASE_CONFIG_MODULE//.//}.py" \
    eveworld/pipeline/t4g_apre_noaug_preflight.py
}

run_submit() {
  run_check
  if find "${OUT_ROOT}/experiments/models" -maxdepth 1 -type d -name 'checkpoint*' \
      -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to resume existing checkpoints under ${OUT_ROOT}" >&2
    exit 1
  fi

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
      "GRADIENT_ACCUMULATION_STEPS=8" \
      "SEED=${SEED}"
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  *)
    cat <<EOF
A-pre-noaug (single-variable ablation of legacy A-pre)
  data:   original packed_data, 92 samples
  base:   ${PRETRAIN}
  output: ${OUT_ROOT}
  train:  ${MAX_STEPS} steps, checkpoints every ${CHECKPOINT_INTERVAL}, seed ${SEED}
  method: legacy L_id + legacy five-level weightmap; p_aug=0

Commands:
  bash $(basename "$0") check
  bash $(basename "$0") submit

Both layer recipes ship side by side, neither is a default:
  BASE_CONFIG_MODULE=eveworld.pipeline.t4g_apre_noaug_config     block 22 (legacy)
  BASE_CONFIG_MODULE=eveworld.pipeline.t4g_apre_noaug_b23_config block 23 (paper probe)
  OUT_ROOT=<dir> RUN_NAME=<name> MAX_STEPS=<n> also overridable
EOF
    ;;
esac
