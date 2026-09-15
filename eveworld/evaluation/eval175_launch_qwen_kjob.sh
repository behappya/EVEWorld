#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST="${MANIFEST:?Set MANIFEST to one audited model manifest}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
RUN_NAME="${RUN_NAME:-eval175_qwen_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/qwen.log}"
export JOB_SCRIPT="${SCRIPT_DIR}/eval175_kjob_qwen.sh"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "DREAMGEN_REPO=${DREAMGEN_REPO:-/home/jovyan/gagibench/GR00T-Dreams}" \
  "EVAL_PYTHON=${EVAL_PYTHON:-/data/datasets/gagi/envs/dreamgenbench_eval_venv/bin/python}" \
  "MANIFEST=${MANIFEST}" \
  "OUTPUT_DIR=${OUTPUT_DIR}" \
  "CHECKPOINT=${CHECKPOINT:-Qwen/Qwen2.5-VL-7B-Instruct}" \
  "METRICS=${METRICS:-qwen_if,pa_i}" \
  "SEED=${SEED:-0}" \
  "START_OFFSET=${START_OFFSET:-0}" \
  "LIMIT=${LIMIT:-0}" \
  "RUN_NAME=${RUN_NAME}" \
  "RUN_LOG=${RUN_LOG}" \
  "GPU_IDS=${GPU_IDS:-0}"
