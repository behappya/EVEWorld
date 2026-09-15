#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_lora_eval_8gpu.sh}"
TASK_SPECS="${TASK_SPECS:?TASK_SPECS is required}"
OUT_ROOT="${OUT_ROOT:-/data/datasets/gagi/eve_outputs/lad_lora_eval}"
LIMIT="${LIMIT:-16}"

echo "[lora-eval] submit one node via ${JOB_SCRIPT}"
echo "[lora-eval] tasks=${TASK_SPECS}"
echo "[lora-eval] out=${OUT_ROOT} limit=${LIMIT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "REPO_DIR=${REPO_DIR}" \
  "TASK_SPECS=${TASK_SPECS}" \
  "OUT_ROOT=${OUT_ROOT}" \
  "LIMIT=${LIMIT}" \
  "NUM_FRAMES=${NUM_FRAMES:-93}" \
  "STEPS=${STEPS:-30}" \
  "HEIGHT=${HEIGHT:-480}" \
  "WIDTH=${WIDTH:-768}" \
  "FPS=${FPS:-16}" \
  "SKIP_EXISTING=${SKIP_EXISTING:-0}"
