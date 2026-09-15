#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Bad argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/kjob_frontier_generate.sh}"
TASK_SPECS="${TASK_SPECS:?TASK_SPECS is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
export JOB_SCRIPT

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "job=${JOB_SCRIPT}"
  echo "tasks=${TASK_SPECS}"
  echo "out=${OUT_ROOT} limit=${LIMIT:-8} steps=${STEPS:-30}"
  exit 0
fi

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "REPO_DIR=${REPO_DIR}" \
  "TASK_SPECS=${TASK_SPECS}" \
  "OUT_ROOT=${OUT_ROOT}" \
  "EXPECTED_SPLIT=${EXPECTED_SPLIT:-train}" \
  "LIMIT=${LIMIT:-8}" \
  "STEPS=${STEPS:-30}" \
  "NUM_FRAMES=${NUM_FRAMES:-93}" \
  "HEIGHT=${HEIGHT:-480}" \
  "WIDTH=${WIDTH:-768}" \
  "FPS=${FPS:-16}" \
  "BLOCK_SIZE=${BLOCK_SIZE:-4}" \
  "BOUNDARY_GUARD_VELOCITY_SCALE=${BOUNDARY_GUARD_VELOCITY_SCALE:-0.5}" \
  "BOUNDARY_GUARD_DECAY=${BOUNDARY_GUARD_DECAY:-1.0}" \
  "SKIP_EXISTING=${SKIP_EXISTING:-0}"
