#!/usr/bin/env bash
set -euo pipefail

# Submit official DreamGenBench Qwen-IF and PA-I evaluation to a GPU kjob.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${VIDEO_DIR:-}" ]]; then
  echo "Set VIDEO_DIR to generated-only videos, e.g.:" >&2
  echo "  VIDEO_DIR=/data/datasets/gagi/gr1_dreamgen_eval/dreamgenbench_video_dirs/<run> ./benchmarks/dreamgenbench/launch_dreamgenbench_qwen_eval_kjob.sh" >&2
  exit 1
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_dreamgenbench_qwen_eval.sh}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_dreamgen_qwen_eval}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
export ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
export GPU_IDS="${GPU_IDS:-0}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
export GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${OUTPUT_ROOT}/${RUN_NAME}_gpu_memory_samples.csv}"
export GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${OUTPUT_ROOT}/${RUN_NAME}_gpu_memory_peak.json}"
export DREAMGEN_EVAL_PYTHON="${DREAMGEN_EVAL_PYTHON:-/data/datasets/gagi/envs/dreamgenbench_eval_venv/bin/python}"

echo "Submitting DreamGenBench Qwen eval kjob"
echo "Repo dir:    ${REPO_DIR}"
echo "Video dir:   ${VIDEO_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run name:    ${RUN_NAME}"
echo "Run log:     ${RUN_LOG}"
echo "Job script:  ${JOB_SCRIPT}"
echo "GPU ids:     ${GPU_IDS}"
echo "GPU samples: ${GPU_MEMORY_SAMPLES}"
echo "GPU peak:    ${GPU_MEMORY_PEAK}"
echo "Eval python: ${DREAMGEN_EVAL_PYTHON}"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "DREAMGEN_REPO=${DREAMGEN_REPO:-/home/jovyan/gagibench/GR00T-Dreams}" \
  "DREAMGEN_EVAL_PYTHON=${DREAMGEN_EVAL_PYTHON}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "RUN_NAME=${RUN_NAME}" \
  "VIDEO_DIR=${VIDEO_DIR}" \
  "LOG_DIR=${LOG_DIR}" \
  "RUN_LOG=${RUN_LOG}" \
  "ENV_FILE=${ENV_FILE}" \
  "GPU_IDS=${GPU_IDS}" \
  "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}" \
  "GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}" \
  "GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}"
