#!/usr/bin/env bash
set -euo pipefail

# Thin submitter for GR1 pack_data. The workspace host only invokes kjobctl;
# T5-11B is loaded inside the GPU kjob payload.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_pack_gr1_finetune_data.sh}"
export DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_finetune_data}"
export VIDEO_DIR="${VIDEO_DIR:-${DATA_ROOT}/raw_data}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
export TEXT_ENCODER_MODEL_PATH="${TEXT_ENCODER_MODEL_PATH:-/data/datasets/gagi/giga_world_0_video_pretrain/text_encoder}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_pack_gr1}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
export ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
export GPU_IDS="${GPU_IDS:-0}"

echo "Submitting GR1 pack_data kjob"
echo "Repo dir:        ${REPO_DIR}"
echo "Data root:       ${DATA_ROOT}"
echo "Video dir:       ${VIDEO_DIR}"
echo "Packed data dir: ${PACKED_DATA_DIR}"
echo "Text encoder:    ${TEXT_ENCODER_MODEL_PATH}"
echo "Run name:        ${RUN_NAME}"
echo "Run log:         ${RUN_LOG}"
echo "Job script:      ${JOB_SCRIPT}"
echo "GPU ids:         ${GPU_IDS}"
echo "Force repack:    ${FORCE_REPACK:-0}"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "DATA_ROOT=${DATA_ROOT}" \
  "VIDEO_DIR=${VIDEO_DIR}" \
  "PACKED_DATA_DIR=${PACKED_DATA_DIR}" \
  "TEXT_ENCODER_MODEL_PATH=${TEXT_ENCODER_MODEL_PATH}" \
  "LOG_DIR=${LOG_DIR}" \
  "RUN_LOG=${RUN_LOG}" \
  "ENV_FILE=${ENV_FILE}" \
  "FORCE_REPACK=${FORCE_REPACK:-0}" \
  "$@"
