#!/usr/bin/env bash
set -euo pipefail

# Qwen-IF + PA-I evaluation for the PhysLatent DreamGen full92 run.
# Defaults are pinned to the currently available Qwen3.6-35B-A3B vLLM endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-physlatent_adapter_step200_full92}"
VIDEO_DIR="${VIDEO_DIR:-${EVAL_ROOT}/dreamgenbench_video_dirs/${SOURCE_RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/eval_outputs}"
RUN_NAME="${RUN_NAME:-${SOURCE_RUN_NAME}_qwen_eval}"
EXPECTED_COUNT="${EXPECTED_COUNT:-92}"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-35B-A3B}"
METRICS="${METRICS:-qwen_if,pa_i}"

START_OFFSET="${START_OFFSET:-0}"
LIMIT="${LIMIT:-0}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"
FRAME_COUNT="${FRAME_COUNT:-49}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-0}"
JPEG_QUALITY="${JPEG_QUALITY:-85}"
MODEL_RETRIES="${MODEL_RETRIES:-3}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-600}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-32000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
RERUN_ERRORS="${RERUN_ERRORS:-0}"
DRY_RUN="${DRY_RUN:-0}"

mp4_count() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

require_video_dir() {
  local count
  count="$(mp4_count "${VIDEO_DIR}")"
  if [[ "${count}" != "${EXPECTED_COUNT}" ]]; then
    echo "Expected ${EXPECTED_COUNT} mp4 files in ${VIDEO_DIR}, found ${count}." >&2
    echo "Run eveworld/alternatives/physlatent/scripts/run_physlatent_full92_pa2.sh first if generated-only videos are missing." >&2
    exit 1
  fi
}

print_plan() {
  cat <<EOF
PhysLatent full92 Qwen eval
  repo:        ${REPO_DIR}
  video_dir:   ${VIDEO_DIR}
  output_root: ${OUTPUT_ROOT}
  run_name:    ${RUN_NAME}
  qwen_base:   ${QWEN_BASE}
  qwen_model:  ${QWEN_MODEL}
  metrics:     ${METRICS}
  expected:    ${EXPECTED_COUNT} videos

Expected outputs:
  ${OUTPUT_ROOT}/${RUN_NAME}_qwen_if.csv
  ${OUTPUT_ROOT}/${RUN_NAME}_pa_i.csv
  ${OUTPUT_ROOT}/${RUN_NAME}_summary.json
  ${OUTPUT_ROOT}/kjob_logs/${RUN_NAME}.log
EOF
}

main() {
  cd "${REPO_DIR}"
  require_video_dir
  print_plan

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, not launching eval."
    return
  fi

  QWEN_BASE="${QWEN_BASE}" \
  QWEN_MODEL="${QWEN_MODEL}" \
  VIDEO_DIR="${VIDEO_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_NAME="${RUN_NAME}" \
  METRICS="${METRICS}" \
  START_OFFSET="${START_OFFSET}" \
  LIMIT="${LIMIT}" \
  CONCURRENCY="${CONCURRENCY}" \
  MAX_INFLIGHT="${MAX_INFLIGHT}" \
  FRAME_COUNT="${FRAME_COUNT}" \
  MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE}" \
  JPEG_QUALITY="${JPEG_QUALITY}" \
  MODEL_RETRIES="${MODEL_RETRIES}" \
  MODEL_TIMEOUT="${MODEL_TIMEOUT}" \
  MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  DISABLE_THINKING="${DISABLE_THINKING}" \
  RERUN_ERRORS="${RERUN_ERRORS}" \
    "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.sh"
}

main "$@"
