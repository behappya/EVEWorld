#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"

VIDEO_DIR="${VIDEO_DIR:-/data/datasets/gagi/gr1_dreamgen_eval/dreamgenbench_video_dirs/gr1_dreamgen_8gpu_full_20260625_212933}"
MANIFEST="${MANIFEST:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_dreamgen_qwen36vl_api}"
RUN_LOG="${RUN_LOG:-${OUTPUT_ROOT}/kjob_logs/${RUN_NAME}.log}"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-auto}"
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

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

if ! python - <<'PY' >/dev/null 2>&1
import openai  # noqa: F401
PY
then
  echo "Missing Python package: openai. Install it in ${CONDA_ENV} first." >&2
  exit 1
fi

mkdir -p "$(dirname "${RUN_LOG}")" "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

echo "============================================" | tee -a "${RUN_LOG}"
echo "DreamGenBench Qwen API eval launcher" | tee -a "${RUN_LOG}"
echo "Repo:                    ${REPO_DIR}" | tee -a "${RUN_LOG}"
echo "Video dir:               ${VIDEO_DIR}" | tee -a "${RUN_LOG}"
echo "Manifest:                ${MANIFEST:--}" | tee -a "${RUN_LOG}"
echo "Output root:             ${OUTPUT_ROOT}" | tee -a "${RUN_LOG}"
echo "Run name:                ${RUN_NAME}" | tee -a "${RUN_LOG}"
echo "Run log:                 ${RUN_LOG}" | tee -a "${RUN_LOG}"
echo "Qwen base:               ${QWEN_BASE}" | tee -a "${RUN_LOG}"
echo "Qwen model:              ${QWEN_MODEL}" | tee -a "${RUN_LOG}"
echo "Metrics:                 ${METRICS}" | tee -a "${RUN_LOG}"
echo "Max tokens:              ${MODEL_MAX_TOKENS}" | tee -a "${RUN_LOG}"
echo "Start/limit:             ${START_OFFSET}/${LIMIT}" | tee -a "${RUN_LOG}"
echo "Concurrency/inflight:    ${CONCURRENCY}/${MAX_INFLIGHT}" | tee -a "${RUN_LOG}"
echo "Frames/image side:       ${FRAME_COUNT}/${MAX_IMAGE_SIDE}" | tee -a "${RUN_LOG}"
echo "============================================" | tee -a "${RUN_LOG}"

echo "==> Qwen health/model endpoint check" | tee -a "${RUN_LOG}"
curl -sS --max-time 20 "${QWEN_BASE%/}/models" | head -c 2000 | tee -a "${RUN_LOG}" || true
echo | tee -a "${RUN_LOG}"

DISABLE_THINKING_FLAG=("--disable-thinking")
if [[ "${DISABLE_THINKING}" == "0" ]]; then
  DISABLE_THINKING_FLAG=("--no-disable-thinking")
fi
RERUN_ERRORS_FLAG=()
if [[ "${RERUN_ERRORS}" == "1" ]]; then
  RERUN_ERRORS_FLAG=("--rerun-errors")
fi
MANIFEST_FLAG=()
if [[ -n "${MANIFEST}" ]]; then
  MANIFEST_FLAG=("--manifest" "${MANIFEST}")
fi

python benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.py \
  --video-dir "${VIDEO_DIR}" \
  "${MANIFEST_FLAG[@]}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --qwen-base "${QWEN_BASE}" \
  --qwen-model "${QWEN_MODEL}" \
  --metrics "${METRICS}" \
  --start-offset "${START_OFFSET}" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --frame-count "${FRAME_COUNT}" \
  --max-image-side "${MAX_IMAGE_SIDE}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --model-retries "${MODEL_RETRIES}" \
  --model-timeout "${MODEL_TIMEOUT}" \
  --model-max-tokens "${MODEL_MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  "${DISABLE_THINKING_FLAG[@]}" \
  "${RERUN_ERRORS_FLAG[@]}" 2>&1 | tee -a "${RUN_LOG}"

echo "summary: ${OUTPUT_ROOT}/${RUN_NAME}_summary.json" | tee -a "${RUN_LOG}"
