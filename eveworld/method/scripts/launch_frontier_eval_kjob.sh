#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/kjob_frontier_eval.sh}"

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}"
done

export REPO_DIR
export JOB_SCRIPT
export TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/eve}"
export RUN_NAME="${RUN_NAME:-frontier_preference_eval_$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"

: "${OUTPUT_JSON:?OUTPUT_JSON is required}"
: "${CHECKPOINT_CONTROL:?CHECKPOINT_CONTROL is required}"
: "${CHECKPOINT_ANTI:?CHECKPOINT_ANTI is required}"

for checkpoint in "${CHECKPOINT_CONTROL}" "${CHECKPOINT_ANTI}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint must be a weight file: ${checkpoint}" >&2
    exit 2
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] JOB_SCRIPT=${JOB_SCRIPT} RUN_NAME=${RUN_NAME}"
  exit 0
fi

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "REPO_DIR=${REPO_DIR}" \
  "JOB_SCRIPT=${JOB_SCRIPT}" \
  "TRAIN_VENV=${TRAIN_VENV}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "RUN_NAME=${RUN_NAME}" \
  "LOG_DIR=${LOG_DIR}" \
  "RUN_LOG=${RUN_LOG}" \
  "OUTPUT_JSON=${OUTPUT_JSON}" \
  "CHECKPOINT_CONTROL=${CHECKPOINT_CONTROL}" \
  "CHECKPOINT_ANTI=${CHECKPOINT_ANTI}" \
  "INDICES=${INDICES:-91,36,57,82,79,23,5,16}" \
  "SIGMA=${SIGMA:-0.5}" \
  "SEED=${SEED:-20260715}" \
  "GPU_ID=${GPU_ID:-0}" \
  "NEGATIVE_MODE=${NEGATIVE_MODE:-adjacent}" \
  "$@"
