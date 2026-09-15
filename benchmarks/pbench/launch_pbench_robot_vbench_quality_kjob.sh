#!/usr/bin/env bash
set -euo pipefail

# Thin submitter for PBench Robot VBench quality evaluation.
# Local side effects are limited to invoking kjobctl through submit_gigaworld0_kjob.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/pbench/kjob_pbench_robot_vbench_quality.sh}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_quality}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
export ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
export GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${LOG_DIR}/${RUN_NAME}_gpu_memory_samples.csv}"
export GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${LOG_DIR}/${RUN_NAME}_gpu_memory_peak.json}"
export GPU_IDS="${GPU_IDS:-0}"

METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_serving_full_20260612_145541}"
DOMAIN_SUMMARY="${DOMAIN_SUMMARY:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval/20260613_131107_qwen36vl/qwen_vqa_summary.json}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/data/datasets/gagi/vbench_cache}"

echo "Submitting PBench Robot VBench quality kjob"
echo "Repo dir:      ${REPO_DIR}"
echo "Output root:   ${OUTPUT_ROOT}"
echo "Run name:      ${RUN_NAME}"
echo "Run log:       ${RUN_LOG}"
echo "Job script:    ${JOB_SCRIPT}"
echo "GPU ids:       ${GPU_IDS}"
echo "GPU samples:   ${GPU_MEMORY_SAMPLES}"
echo "GPU peak:      ${GPU_MEMORY_PEAK}"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "METADATA_JSONL=${METADATA_JSONL}" \
  "SOURCE_VIDEO_DIR=${SOURCE_VIDEO_DIR}" \
  "DOMAIN_SUMMARY=${DOMAIN_SUMMARY}" \
  "VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR}" \
  "LOG_DIR=${LOG_DIR}" \
  "RUN_LOG=${RUN_LOG}" \
  "ENV_FILE=${ENV_FILE}" \
  "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}" \
  "GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}" \
  "GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}" \
  "GPU_IDS=${GPU_IDS}"
