#!/usr/bin/env bash
set -euo pipefail

# Submit a small DreamGen generation spot-check using the PhysLatent adapter.
# Defaults are conservative: 4 samples, 93 frames, 30 denoising steps, 8 GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_gr1_dreamgen_generation.sh}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/datasets/gagi/giga_world_0_outputs/physlatent_gigaworld/experiments_adapter_warmup/models/checkpoint_epoch_100_step_200}"
export PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export USE_EMA="${USE_EMA:-0}"
export PHYSICS_LATENT_MODEL_PATH="${PHYSICS_LATENT_MODEL_PATH:-${CHECKPOINT_DIR}/physics_latent_encoder}"
export PHYSLATENT_UNCOND_MODE="${PHYSLATENT_UNCOND_MODE:-shared}"
export PYTHON_BIN="${PYTHON_BIN:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"

export EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
export DATA_PATH="${DATA_PATH:-${EVAL_ROOT}/giga_input/gr1_dreamgen_it2v.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_physlatent_adapter_step200_spotcheck}"
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

export DATA_LIMIT="${DATA_LIMIT:-4}"
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
PHYSICS_LATENT_MODEL_PATH=${PHYSICS_LATENT_MODEL_PATH}
PHYSLATENT_UNCOND_MODE=${PHYSLATENT_UNCOND_MODE}
PYTHON_BIN=${PYTHON_BIN}
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
  echo "Submitting PhysLatent DreamGen spot-check kjob"
  echo "Repo dir:             ${REPO_DIR}"
  echo "Job script:           ${JOB_SCRIPT}"
  echo "Checkpoint dir:       ${CHECKPOINT_DIR}"
  echo "Physics latent path:  ${PHYSICS_LATENT_MODEL_PATH}"
  echo "PhysLatent CFG mode:  ${PHYSLATENT_UNCOND_MODE}"
  echo "Python:               ${PYTHON_BIN}"
  echo "Pretrain dir:         ${PRETRAIN_DIR}"
  echo "Use EMA:              ${USE_EMA}"
  echo "Data path:            ${DATA_PATH}"
  echo "Save dir:             ${SAVE_DIR}"
  echo "Submit log:           ${SUBMIT_LOG}"
  echo "Run log:              ${LOG_FILE}"
  echo "Summary:              ${SUMMARY_PATH}"
  echo "GPU ids:              ${GPU_IDS}"
  echo "Data limit:           ${DATA_LIMIT}"
  echo "Steps:                ${NUM_INFERENCE_STEPS}"
  echo "Frames/FPS/Size:      ${NUM_FRAMES}/${FPS}/${HEIGHT}x${WIDTH}"
  echo "Launcher env:         ${SAVE_DIR}/launcher.env"
  echo "============================================"
  echo
  echo "Monitor:"
  echo "  tail -f ${LOG_FILE}"
  echo
  echo "After completion:"
  echo "  cat ${SUMMARY_PATH}"
  echo
} | tee -a "${SUBMIT_LOG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not submitting kjob." | tee -a "${SUBMIT_LOG}"
  exit 0
fi

"${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "CHECKPOINT_DIR=${CHECKPOINT_DIR}" \
  "PRETRAIN_DIR=${PRETRAIN_DIR}" \
  "USE_EMA=${USE_EMA}" \
  "PHYSICS_LATENT_MODEL_PATH=${PHYSICS_LATENT_MODEL_PATH}" \
  "PHYSLATENT_UNCOND_MODE=${PHYSLATENT_UNCOND_MODE}" \
  "PYTHON_BIN=${PYTHON_BIN}" \
  "EVAL_ROOT=${EVAL_ROOT}" \
  "DATA_PATH=${DATA_PATH}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "RUN_NAME=${RUN_NAME}" \
  "SAVE_DIR=${SAVE_DIR}" \
  "MODEL_DIR=${MODEL_DIR}" \
  "LOG_FILE=${LOG_FILE}" \
  "ENV_FILE=${ENV_FILE}" \
  "SUMMARY_PATH=${SUMMARY_PATH}" \
  "GPU_IDS=${GPU_IDS}" \
  "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}" \
  "GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}" \
  "GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}" \
  "DATA_LIMIT=${DATA_LIMIT}" \
  "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}" \
  "FPS=${FPS}" \
  "NUM_FRAMES=${NUM_FRAMES}" \
  "HEIGHT=${HEIGHT}" \
  "WIDTH=${WIDTH}" \
  "SEED=${SEED}" \
  2>&1 | tee -a "${SUBMIT_LOG}"
