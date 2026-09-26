#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
PACKED="${GAGI}/gr1_finetune_data/packed_data_t4g_nohuman_v2"
PRETRAIN="${GAGI}/giga_world_0_video_pretrain/transformer"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_v2_outputs/t4g_joint_wmapA_pre_cleanv2_armfixv4_u3}"
RUN_NAME="${RUN_NAME:-t4g_wmapA_pre_cleanv2_armfixv4_u3_s150}"
BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.pipeline.t4g_joint_cleanv2_config}"
MAX_STEPS="${MAX_STEPS:-150}"
# Sample-count gate: the nohuman_v2 subset holds 91 clips (video 32 excluded);
# set T4G_EXPECTED_SAMPLES=<n> to gate a differently sized anno dir instead.
EXPECTED_SAMPLES="${T4G_EXPECTED_SAMPLES:-91}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-3}"
PYTHON="/home/jovyan/miniconda/envs/EVEWorld/bin/python"

run_check() {
  [[ -f "${PACKED}/config.json" ]] || { echo "missing packed data: ${PACKED}"; exit 1; }
  [[ -f "${PRETRAIN}/diffusion_pytorch_model.safetensors" ]] || { echo "missing pretrain"; exit 1; }
  cd "${REPO_DIR}"
  "${PYTHON}" eveworld/pipeline/t4g_cleanv2_preflight.py --expected "${EXPECTED_SAMPLES}"
  "${PYTHON}" -m py_compile \
    eveworld/pipeline/t4g_aug_trainer.py \
    eveworld/pipeline/t4g_joint_trainer.py \
    "${BASE_CONFIG_MODULE//.//}.py"
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
      "GRADIENT_ACCUMULATION_STEPS=8" \
      "T4G_EXPECTED_SAMPLES=${EXPECTED_SAMPLES}"
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
  samples: ${EXPECTED_SAMPLES} (T4G_EXPECTED_SAMPLES=<n> to gate another anno set)

Commands:
  bash $(basename "$0") check
  bash $(basename "$0") submit
EOF
    ;;
esac
