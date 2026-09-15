#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"

VIDEO_DIR="${VIDEO_DIR:-/data/datasets/gagi/gr1_dreamgen_eval/dreamgenbench_video_dirs/gr1_dreamgen_8gpu_full_20260625_212933}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_dreamgen_gpt_if}"
RUN_LOG="${RUN_LOG:-${OUTPUT_ROOT}/kjob_logs/${RUN_NAME}_gpt_if.log}"

DIFROST_GENAI_BASE_URL="${DIFROST_GENAI_BASE_URL:-https://api-gateway.example.com/v1}"
DIFROST_HOST="${DIFROST_HOST:-api-gateway.example.com}"
DIFROST_MODEL="${DIFROST_MODEL:-gpt-5.5}"
START_OFFSET="${START_OFFSET:-0}"
LIMIT="${LIMIT:-0}"
CONCURRENCY="${CONCURRENCY:-10}"
FRAME_COUNT="${FRAME_COUNT:-8}"
SCALE_FACTOR="${SCALE_FACTOR:-0.5}"
MODEL_RETRIES="${MODEL_RETRIES:-6}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-32000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
THINKING_LEVEL="${THINKING_LEVEL:-low}"
INCLUDE_THOUGHTS="${INCLUDE_THOUGHTS:-0}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

mkdir -p "$(dirname "${RUN_LOG}")" "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

echo "============================================" | tee -a "${RUN_LOG}"
echo "DreamGenBench GPT-IF API eval launcher" | tee -a "${RUN_LOG}"
echo "Repo:                    ${REPO_DIR}" | tee -a "${RUN_LOG}"
echo "Video dir:               ${VIDEO_DIR}" | tee -a "${RUN_LOG}"
echo "Output root:             ${OUTPUT_ROOT}" | tee -a "${RUN_LOG}"
echo "Run name:                ${RUN_NAME}" | tee -a "${RUN_LOG}"
echo "Run log:                 ${RUN_LOG}" | tee -a "${RUN_LOG}"
echo "Difrost base:            ${DIFROST_GENAI_BASE_URL}" | tee -a "${RUN_LOG}"
echo "Difrost host:            ${DIFROST_HOST}" | tee -a "${RUN_LOG}"
echo "Difrost model:           ${DIFROST_MODEL}" | tee -a "${RUN_LOG}"
echo "Start/limit:             ${START_OFFSET}/${LIMIT}" | tee -a "${RUN_LOG}"
echo "Concurrency:             ${CONCURRENCY}" | tee -a "${RUN_LOG}"
echo "Frames/scale:            ${FRAME_COUNT}/${SCALE_FACTOR}" | tee -a "${RUN_LOG}"
echo "============================================" | tee -a "${RUN_LOG}"

THOUGHTS_FLAG=("--no-include-thoughts")
if [[ "${INCLUDE_THOUGHTS}" == "1" || "${INCLUDE_THOUGHTS}" == "true" ]]; then
  THOUGHTS_FLAG=("--include-thoughts")
fi

python benchmarks/dreamgenbench/eval_dreamgenbench_gpt_if_api.py \
  --video-dir "${VIDEO_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --base-url "${DIFROST_GENAI_BASE_URL}" \
  --host "${DIFROST_HOST}" \
  --api-token "${DIFROST_API_TOKEN:?set DIFROST_API_TOKEN}" \
  --model "${DIFROST_MODEL}" \
  --start-offset "${START_OFFSET}" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --frame-count "${FRAME_COUNT}" \
  --scale-factor "${SCALE_FACTOR}" \
  --model-retries "${MODEL_RETRIES}" \
  --model-max-tokens "${MODEL_MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --thinking-level "${THINKING_LEVEL}" \
  "${THOUGHTS_FLAG[@]}" 2>&1 | tee -a "${RUN_LOG}"

echo "summary: ${OUTPUT_ROOT}/${RUN_NAME}_gpt_if_summary.json" | tee -a "${RUN_LOG}"
