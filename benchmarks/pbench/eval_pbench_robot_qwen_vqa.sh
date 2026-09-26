#!/usr/bin/env bash
set -euo pipefail

# Evaluate generated PBench Robot videos with an OpenAI-compatible Qwen VL
# endpoint. This script only sends HTTP requests to Qwen; it does not load
# judge model weights locally.
#
# Common overrides:
#   LIMIT=1 ./benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh
#   EVAL_DIR=/path/to/existing_eval ./benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh
#   QWEN_BASE=http://host:8000/v1 QWEN_MODEL=model_id ./benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"

METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_serving_full_20260612_145541}"
EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_DIR="${EVAL_DIR:-${EVAL_ROOT}/${TIMESTAMP}_qwen36vl}"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-auto}"

START_OFFSET="${START_OFFSET:-0}"
LIMIT="${LIMIT:-0}"
N_REPEATS="${N_REPEATS:-1}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"
FRAME_COUNT="${FRAME_COUNT:-8}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-512}"
JPEG_QUALITY="${JPEG_QUALITY:-85}"
CROP_MODE="${CROP_MODE:-right-half}"
MODEL_RETRIES="${MODEL_RETRIES:-3}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-300}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-256}"
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
  cat >&2 <<'EOF'
Missing Python package: openai

Install it in the conda env with:
  source ~/miniconda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-EVEWorld}"
  python -m pip install openai

Then rerun:
  ./benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh
EOF
  exit 1
fi

mkdir -p "${EVAL_DIR}"
RUN_LOG="${RUN_LOG:-${EVAL_DIR}/eval.log}"

cd "${REPO_DIR}"

echo "============================================" | tee -a "${RUN_LOG}"
echo "PBench Robot Qwen VQA eval launcher" | tee -a "${RUN_LOG}"
echo "Repo:                    ${REPO_DIR}" | tee -a "${RUN_LOG}"
echo "Metadata:                ${METADATA_JSONL}" | tee -a "${RUN_LOG}"
echo "Video dir:               ${VIDEO_DIR}" | tee -a "${RUN_LOG}"
echo "Eval dir:                ${EVAL_DIR}" | tee -a "${RUN_LOG}"
echo "Run log:                 ${RUN_LOG}" | tee -a "${RUN_LOG}"
echo "Qwen base:               ${QWEN_BASE}" | tee -a "${RUN_LOG}"
echo "Qwen model:              ${QWEN_MODEL}" | tee -a "${RUN_LOG}"
echo "Start/limit/repeats:     ${START_OFFSET}/${LIMIT}/${N_REPEATS}" | tee -a "${RUN_LOG}"
echo "Concurrency/inflight:    ${CONCURRENCY}/${MAX_INFLIGHT}" | tee -a "${RUN_LOG}"
echo "Frames/crop/image side:  ${FRAME_COUNT}/${CROP_MODE}/${MAX_IMAGE_SIDE}" | tee -a "${RUN_LOG}"
echo "============================================" | tee -a "${RUN_LOG}"

echo "==> Qwen health/model endpoint check" | tee -a "${RUN_LOG}"
curl -sS --max-time 20 "${QWEN_BASE%/}/models" | head -c 2000 | tee -a "${RUN_LOG}" || true
echo | tee -a "${RUN_LOG}"

echo "==> Current generated video count" | tee -a "${RUN_LOG}"
find "${VIDEO_DIR}" -maxdepth 1 -name 'robot_*.mp4' | wc -l | tee -a "${RUN_LOG}"

DISABLE_THINKING_FLAG=("--disable-thinking")
if [[ "${DISABLE_THINKING}" == "0" ]]; then
  DISABLE_THINKING_FLAG=("--no-disable-thinking")
fi
RERUN_ERRORS_FLAG=()
if [[ "${RERUN_ERRORS}" == "1" ]]; then
  RERUN_ERRORS_FLAG=("--rerun-errors")
fi

python benchmarks/pbench/eval_pbench_robot_qwen_vqa.py \
  --metadata-jsonl "${METADATA_JSONL}" \
  --video-dir "${VIDEO_DIR}" \
  --output-dir "${EVAL_DIR}" \
  --qwen-base "${QWEN_BASE}" \
  --qwen-model "${QWEN_MODEL}" \
  --start-offset "${START_OFFSET}" \
  --limit "${LIMIT}" \
  --n-repeats "${N_REPEATS}" \
  --concurrency "${CONCURRENCY}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --frame-count "${FRAME_COUNT}" \
  --max-image-side "${MAX_IMAGE_SIDE}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --crop-mode "${CROP_MODE}" \
  --model-retries "${MODEL_RETRIES}" \
  --model-timeout "${MODEL_TIMEOUT}" \
  --model-max-tokens "${MODEL_MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  "${DISABLE_THINKING_FLAG[@]}" \
  "${RERUN_ERRORS_FLAG[@]}" 2>&1 | tee -a "${RUN_LOG}"

echo "summary: ${EVAL_DIR}/qwen_vqa_summary.json" | tee -a "${RUN_LOG}"
echo "results: ${EVAL_DIR}/qwen_vqa_results.jsonl" | tee -a "${RUN_LOG}"
