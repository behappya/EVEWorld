#!/usr/bin/env bash
# Submit one checkpoint's held-out generation on one eight-GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
for arg in "$@"; do
  [[ "${arg}" == *=* ]] || { echo "Bad argument: ${arg}" >&2; exit 1; }
  export "${arg}"
done

PIPELINE="${PIPELINE:?PIPELINE is required}"
METHOD="${METHOD:?METHOD is required}"
TRAINING_SEED="${TRAINING_SEED:?TRAINING_SEED is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?SPLIT_MANIFEST is required}"
EXPECTED_SPLIT="${EXPECTED_SPLIT:-test}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
LORA="${LORA:-NONE}"

python3 "${EVEWORLD_ROOT}/eveworld/data_curation/scripts/heldout_manifest_tool.py" validate-eval \
  --manifest "${SPLIT_MANIFEST}" \
  --expected-split "${EXPECTED_SPLIT}" \
  --data-path "${DATA_PATH}"

echo "[heldout-gen] submit ${METHOD}/train_seed_${TRAINING_SEED}"
echo "[heldout-gen] one node, GPUs=0,1,2,3,4,5,6,7"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] pipeline=${PIPELINE} lora=${LORA} out=${OUT_ROOT}"
  exit 0
fi

export JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/kjob_heldout_generate.sh}"
exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "REPO_DIR=${REPO_DIR}" \
  "PIPELINE=${PIPELINE}" \
  "METHOD=${METHOD}" \
  "TRAINING_SEED=${TRAINING_SEED}" \
  "GENERATION_SEEDS=${GENERATION_SEEDS:-6666 1234}" \
  "DATA_PATH=${DATA_PATH}" \
  "SPLIT_MANIFEST=${SPLIT_MANIFEST}" \
  "EXPECTED_SPLIT=${EXPECTED_SPLIT}" \
  "OUT_ROOT=${OUT_ROOT}" \
  "LORA=${LORA}" \
  "LIMIT=${LIMIT:-0}" \
  "STEPS=${STEPS:-30}" \
  "NUM_FRAMES=${NUM_FRAMES:-93}" \
  "HEIGHT=${HEIGHT:-480}" \
  "WIDTH=${WIDTH:-768}" \
  "FPS=${FPS:-16}" \
  "BLOCK_SIZE=${BLOCK_SIZE:-4}" \
  "SKIP_EXISTING=${SKIP_EXISTING:-0}"
