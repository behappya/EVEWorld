#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for GigaWorld-0 GR1 fine-tuning.
#
# Local side effects:
#   1. prepare/check the lightweight training venv unless SKIP_TRAIN_ENV_SETUP=1
#   2. submit the training payload through kjobctl
#
# The full model is loaded only inside the GPU kjob.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh}"
export TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
export DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_finetune_data}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
export MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune}"
export TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR:-${OUTPUT_ROOT}/experiments}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_train_gr1}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
export ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
export RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-${OUTPUT_ROOT}/runtime_configs}"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-${RUNTIME_CONFIG_DIR}/${RUN_NAME}.json}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
export GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${LOG_DIR}/${RUN_NAME}_gpu_memory_samples.csv}"
export GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${LOG_DIR}/${RUN_NAME}_gpu_memory_peak.json}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export MAX_STEPS="${MAX_STEPS:-200}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
export CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-5}"
export TRANSFORMER_MODEL_PATH="${TRANSFORMER_MODEL_PATH:-${MODEL_DIR}/transformer}"

if [[ "${SKIP_TRAIN_ENV_SETUP:-0}" != "1" ]]; then
  echo "Preparing isolated GigaWorld training venv..."
  "${REPO_DIR}/scripts/setup_gigaworld_train_venv.sh"
else
  echo "Skipping training venv setup because SKIP_TRAIN_ENV_SETUP=1"
fi

if [[ ! -f "${PACKED_DATA_DIR}/config.json" ]]; then
  echo "Missing packed data: ${PACKED_DATA_DIR}/config.json" >&2
  echo "Run first: ./benchmarks/dreamgenbench/launch_gr1_pack_data_kjob.sh" >&2
  exit 1
fi

echo
echo "Submitting GigaWorld-0 GR1 fine-tune kjob"
echo "Repo dir:        ${REPO_DIR}"
echo "Train venv:      ${TRAIN_VENV}"
echo "Packed data:     ${PACKED_DATA_DIR}"
echo "Model dir:       ${MODEL_DIR}"
echo "Project dir:     ${TRAIN_PROJECT_DIR}"
echo "Run name:        ${RUN_NAME}"
echo "Run log:         ${RUN_LOG}"
echo "Runtime config:  ${RUNTIME_CONFIG}"
echo "GPU samples:     ${GPU_MEMORY_SAMPLES}"
echo "GPU peak:        ${GPU_MEMORY_PEAK}"
echo "Job script:      ${JOB_SCRIPT}"
echo "GPU ids:         ${GPU_IDS}"
echo "Steps:           ${MAX_STEPS}"
echo "Batch/GPU:       ${BATCH_SIZE_PER_GPU}"
echo "Grad accum:      ${GRADIENT_ACCUMULATION_STEPS}"
echo "Checkpoint int:  ${CHECKPOINT_INTERVAL}"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "TRAIN_VENV=${TRAIN_VENV}" \
  "DATA_ROOT=${DATA_ROOT}" \
  "PACKED_DATA_DIR=${PACKED_DATA_DIR}" \
  "MODEL_DIR=${MODEL_DIR}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}" \
  "LOG_DIR=${LOG_DIR}" \
  "RUN_LOG=${RUN_LOG}" \
  "ENV_FILE=${ENV_FILE}" \
  "RUNTIME_CONFIG_DIR=${RUNTIME_CONFIG_DIR}" \
  "RUNTIME_CONFIG=${RUNTIME_CONFIG}" \
  "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}" \
  "GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}" \
  "GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}" \
  "GPU_IDS=${GPU_IDS}" \
  "MAX_STEPS=${MAX_STEPS}" \
  "BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU}" \
  "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" \
  "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
  "$@"
