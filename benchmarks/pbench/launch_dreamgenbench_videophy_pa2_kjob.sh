#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${VIDEO_DIR:-}" ]]; then
  echo "Set VIDEO_DIR to generated-only videos, e.g.:" >&2
  echo "  VIDEO_DIR=/data/datasets/gagi/gr1_dreamgen_eval/dreamgenbench_video_dirs/<run> ./benchmarks/pbench/launch_dreamgenbench_videophy_pa2_kjob.sh" >&2
  exit 1
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/pbench/kjob_dreamgenbench_videophy_pa2.sh}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_dreamgen_videophy_pa2}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
export GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${LOG_DIR}/${RUN_NAME}_gpu_memory_samples.csv}"
export GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${LOG_DIR}/${RUN_NAME}_gpu_memory_peak.json}"
export GPU_IDS="${GPU_IDS:-0}"
export VIDEOPHY_DIR="${VIDEOPHY_DIR:-/data/datasets/gagi/videophy}"
export VIDEOPHY_PYTHON="${VIDEOPHY_PYTHON:-/home/jovyan/miniconda/envs/EVEWorld/bin/python}"
export CHECKPOINT="${CHECKPOINT:-videophysics/videocon_physics}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export START_OFFSET="${START_OFFSET:-0}"
export LIMIT="${LIMIT:-0}"

echo "Submitting DreamGenBench VideoPhy PA-II kjob"
echo "Repo dir:        ${REPO_DIR}"
echo "Video dir:       ${VIDEO_DIR}"
echo "Output root:     ${OUTPUT_ROOT}"
echo "Run name:        ${RUN_NAME}"
echo "Run log:         ${RUN_LOG}"
echo "Job script:      ${JOB_SCRIPT}"
echo "GPU ids:         ${GPU_IDS}"
echo "GPU samples:     ${GPU_MEMORY_SAMPLES}"
echo "GPU peak:        ${GPU_MEMORY_PEAK}"
echo "VideoPhy dir:    ${VIDEOPHY_DIR}"
echo "VideoPhy python: ${VIDEOPHY_PYTHON}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Start/limit:     ${START_OFFSET}/${LIMIT}"

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
  "VIDEO_DIR=${VIDEO_DIR}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "RUN_NAME=${RUN_NAME}" \
  "LOG_DIR=${LOG_DIR}" \
  "RUN_LOG=${RUN_LOG}" \
  "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}" \
  "GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}" \
  "GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}" \
  "GPU_IDS=${GPU_IDS}" \
  "VIDEOPHY_DIR=${VIDEOPHY_DIR}" \
  "VIDEOPHY_PYTHON=${VIDEOPHY_PYTHON}" \
  "CHECKPOINT=${CHECKPOINT}" \
  "BATCH_SIZE=${BATCH_SIZE}" \
  "START_OFFSET=${START_OFFSET}" \
  "LIMIT=${LIMIT}"
