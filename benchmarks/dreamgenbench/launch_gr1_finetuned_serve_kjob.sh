#!/usr/bin/env bash
set -euo pipefail

# Start a persistent HTTP service for a locally fine-tuned GR1 checkpoint.
# This script only creates symlinks and submits a kjob; model loading happens
# inside the kjob pod.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
USE_EMA="${USE_EMA:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/serving}"
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_gr1_finetuned_serve}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
MODEL_DIR="${MODEL_DIR:-${SAVE_DIR}/model}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/serve.log}"
ENV_FILE="${ENV_FILE:-${SAVE_DIR}/serve.env}"
RESULTS_DIR="${RESULTS_DIR:-${SAVE_DIR}/results}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${SAVE_DIR}/gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${SAVE_DIR}/gpu_memory_peak.json}"

GPU_IDS="${GPU_IDS:-0}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda:0}"

if [[ "${USE_EMA}" == "1" ]]; then
  TRANSFORMER_SRC="${CHECKPOINT_DIR}/transformer_ema"
else
  TRANSFORMER_SRC="${CHECKPOINT_DIR}/transformer"
fi

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing directory: ${path}" >&2
    exit 1
  fi
}

require_any_file() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [[ -f "${path}" ]]; then
      return 0
    fi
  done
  echo "Missing required ${label}. Checked:" >&2
  for path in "$@"; do
    echo "  ${path}" >&2
  done
  exit 1
}

require_dir "${CHECKPOINT_DIR}"
require_dir "${TRANSFORMER_SRC}"
require_dir "${PRETRAIN_DIR}/text_encoder"
require_dir "${PRETRAIN_DIR}/vae"
require_any_file "transformer weights" \
  "${TRANSFORMER_SRC}/diffusion_pytorch_model.bin" \
  "${TRANSFORMER_SRC}/diffusion_pytorch_model.safetensors"

mkdir -p "${MODEL_DIR}" "${SAVE_DIR}" "${RESULTS_DIR}"
ln -sfn "${TRANSFORMER_SRC}" "${MODEL_DIR}/transformer"
ln -sfn "${PRETRAIN_DIR}/text_encoder" "${MODEL_DIR}/text_encoder"
ln -sfn "${PRETRAIN_DIR}/vae" "${MODEL_DIR}/vae"

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${REPO_DIR}/scripts/kjob_gigaworld0_video_pretrain_serve.sh}"
export MODEL_DIR
export OUTPUT_ROOT
export RUN_NAME
export SAVE_DIR
export LOG_FILE
export ENV_FILE
export GPU_IDS
export GPU_MONITOR_INTERVAL
export GPU_MEMORY_SAMPLES
export GPU_MEMORY_PEAK
export HOST
export PORT
export DEVICE

echo "Submitting GR1 fine-tuned GigaWorld-0 service"
echo "Repo dir:        ${REPO_DIR}"
echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Transformer src: ${TRANSFORMER_SRC}"
echo "Model dir:       ${MODEL_DIR}"
echo "Save dir:        ${SAVE_DIR}"
echo "Serve log:       ${LOG_FILE}"
echo "GPU samples:     ${GPU_MEMORY_SAMPLES}"
echo "GPU peak:        ${GPU_MEMORY_PEAK}"
echo "GPU ids:         ${GPU_IDS}"
echo "Port:            ${PORT}"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "SERVE_LOG=${LOG_FILE}" \
  "RESULTS_DIR=${RESULTS_DIR}" \
  "HOST=${HOST}" \
  "PORT=${PORT}" \
  "DEVICE=${DEVICE}"
