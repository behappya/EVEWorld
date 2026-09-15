#!/usr/bin/env bash
set -euo pipefail

# Submit 1-node / 8-GPU GR1 fine-tuned generation for DreamGenBench inputs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_gr1_dreamgen_generation.sh}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
export PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export USE_EMA="${USE_EMA:-1}"
export EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
export DATA_PATH="${DATA_PATH:-${EVAL_ROOT}/giga_input/gr1_dreamgen_it2v.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_gr1_dreamgen_8gpu}"
export SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
export MODEL_DIR="${MODEL_DIR:-${SAVE_DIR}/model}"
export LOG_FILE="${LOG_FILE:-${SAVE_DIR}/run.log}"
export ENV_FILE="${ENV_FILE:-${SAVE_DIR}/run.env}"
export SUBMIT_LOG="${SUBMIT_LOG:-${SAVE_DIR}/submit.log}"
export SUMMARY_PATH="${SUMMARY_PATH:-${SAVE_DIR}/generation_summary.json}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
export GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${SAVE_DIR}/gpu_memory_samples.csv}"
export GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${SAVE_DIR}/gpu_memory_peak.json}"
export DATA_LIMIT="${DATA_LIMIT:-0}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
export FPS="${FPS:-16}"
export NUM_FRAMES="${NUM_FRAMES:-93}"
export HEIGHT="${HEIGHT:-480}"
export WIDTH="${WIDTH:-768}"
export SEED="${SEED:-6666}"

mkdir -p "${SAVE_DIR}"

cat > "${SAVE_DIR}/launcher.env" <<EOF
TIMESTAMP=${TIMESTAMP}
REPO_DIR=${REPO_DIR}
JOB_SCRIPT=${JOB_SCRIPT}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
PRETRAIN_DIR=${PRETRAIN_DIR}
USE_EMA=${USE_EMA}
EVAL_ROOT=${EVAL_ROOT}
DATA_PATH=${DATA_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_NAME=${RUN_NAME}
SAVE_DIR=${SAVE_DIR}
MODEL_DIR=${MODEL_DIR}
LOG_FILE=${LOG_FILE}
ENV_FILE=${ENV_FILE}
SUBMIT_LOG=${SUBMIT_LOG}
SUMMARY_PATH=${SUMMARY_PATH}
GPU_IDS=${GPU_IDS}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}
GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}
GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}
DATA_LIMIT=${DATA_LIMIT}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}
FPS=${FPS}
NUM_FRAMES=${NUM_FRAMES}
HEIGHT=${HEIGHT}
WIDTH=${WIDTH}
SEED=${SEED}
EOF

{
  echo "============================================"
  echo "Submitting GR1 DreamGen generation kjob"
  echo "Repo dir:       ${REPO_DIR}"
  echo "Job script:     ${JOB_SCRIPT}"
  echo "Checkpoint dir: ${CHECKPOINT_DIR}"
  echo "Pretrain dir:   ${PRETRAIN_DIR}"
  echo "Use EMA:        ${USE_EMA}"
  echo "Data path:      ${DATA_PATH}"
  echo "Save dir:       ${SAVE_DIR}"
  echo "Submit log:     ${SUBMIT_LOG}"
  echo "Run log:        ${LOG_FILE}"
  echo "Summary:        ${SUMMARY_PATH}"
  echo "GPU ids:        ${GPU_IDS}"
  echo "GPU samples:    ${GPU_MEMORY_SAMPLES}"
  echo "GPU peak:       ${GPU_MEMORY_PEAK}"
  echo "Data limit:     ${DATA_LIMIT}"
  echo "Steps:          ${NUM_INFERENCE_STEPS}"
  echo "Frames/FPS:     ${NUM_FRAMES}/${FPS}"
  echo "Size:           ${HEIGHT}x${WIDTH}"
  echo "Launcher env:   ${SAVE_DIR}/launcher.env"
  echo "============================================"
  echo
  echo "Preferred log monitor:"
  echo "  tail -f ${LOG_FILE}"
  echo
  echo "If Kubernetes pod logs are needed, avoid -f unless necessary:"
  echo "  kubectl get pods --sort-by=.metadata.creationTimestamp | tail -n 20"
  echo "  kubectl logs POD_NAME -c job-container"
  echo
} | tee -a "${SUBMIT_LOG}"

"${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "CHECKPOINT_DIR=${CHECKPOINT_DIR}" \
  "PRETRAIN_DIR=${PRETRAIN_DIR}" \
  "USE_EMA=${USE_EMA}" \
  "EVAL_ROOT=${EVAL_ROOT}" \
  "SUMMARY_PATH=${SUMMARY_PATH}" \
  "DATA_LIMIT=${DATA_LIMIT}" \
  2>&1 | tee -a "${SUBMIT_LOG}"
