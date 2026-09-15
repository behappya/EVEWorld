#!/usr/bin/env bash
set -euo pipefail

# Thin submitter for the persistent GigaWorld-0 HTTP service.
# Local side effects are intentionally limited to invoking kjobctl through
# submit_gigaworld0_kjob.sh. The model is loaded only by the kjob payload.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${REPO_DIR}/scripts/kjob_gigaworld0_video_pretrain_serve.sh}"
export MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gigaworld0_serving}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_serve}"
export SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
export LOG_FILE="${LOG_FILE:-${SAVE_DIR}/serve.log}"
export ENV_FILE="${ENV_FILE:-${SAVE_DIR}/serve.env}"
export GPU_IDS="${GPU_IDS:-0}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_DIR="${RESULTS_DIR:-${SAVE_DIR}/results}"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "SERVE_LOG=${LOG_FILE}" \
  "RESULTS_DIR=${RESULTS_DIR}" \
  "HOST=${HOST}" \
  "PORT=${PORT}" \
  "DEVICE=${DEVICE}"
