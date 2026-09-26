#!/usr/bin/env bash
#SBATCH --job-name=dreamgen_pa2
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
VIDEOPHY_DIR="${VIDEOPHY_DIR:-/data/datasets/gagi/videophy}"
VIDEOPHY_PYTHON="${VIDEOPHY_PYTHON:-/home/jovyan/miniconda/envs/EVEWorld/bin/python}"
VIDEO_DIR="${VIDEO_DIR:?Set VIDEO_DIR to generated-only DreamGenBench videos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs}"
RUN_NAME="${RUN_NAME:-dreamgen_videophy_pa2_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${LOG_DIR}/${RUN_NAME}_gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${LOG_DIR}/${RUN_NAME}_gpu_memory_peak.json}"
GPU_MONITOR_PID=""
GPU_IDS="${GPU_IDS:-0}"
CHECKPOINT="${CHECKPOINT:-videophysics/videocon_physics}"
BATCH_SIZE="${BATCH_SIZE:-16}"
START_OFFSET="${START_OFFSET:-0}"
LIMIT="${LIMIT:-0}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/datasets/gagi/.cache}"
IFS=' ,' read -r -a REQUESTED_GPU_IDS_ARGS <<< "${GPU_IDS}"
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${REQUESTED_GPU_IDS_ARGS[*]}")"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${HF_HOME}" "${HF_XET_CACHE}"
exec > >(tee -a "${RUN_LOG}") 2>&1

start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; GPU memory monitor disabled."
    return 0
  fi
  mkdir -p "$(dirname "${GPU_MEMORY_SAMPLES}")" "$(dirname "${GPU_MEMORY_PEAK}")"
  "${VIDEOPHY_PYTHON}" "${REPO_DIR}/scripts/gpu_memory_monitor.py" \
    --samples-csv "${GPU_MEMORY_SAMPLES}" \
    --peak-json "${GPU_MEMORY_PEAK}" \
    --interval-sec "${GPU_MONITOR_INTERVAL}" \
    --label "dreamgen_videophy_pa2_${RUN_NAME}" &
  GPU_MONITOR_PID="$!"
  echo "GPU memory monitor pid: ${GPU_MONITOR_PID}"
  echo "GPU memory samples:     ${GPU_MEMORY_SAMPLES}"
  echo "GPU memory peak:        ${GPU_MEMORY_PEAK}"
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    GPU_MONITOR_PID=""
  fi
}

echo "============================================"
echo "DreamGenBench VideoPhy PA-II"
echo "Host:                 $(hostname)"
echo "Repo:                 ${REPO_DIR}"
echo "VideoPhy dir:         ${VIDEOPHY_DIR}"
echo "VideoPhy python:      ${VIDEOPHY_PYTHON}"
echo "Video dir:            ${VIDEO_DIR}"
echo "Output root:          ${OUTPUT_ROOT}"
echo "Run name:             ${RUN_NAME}"
echo "Run log:              ${RUN_LOG}"
echo "GPU memory samples:   ${GPU_MEMORY_SAMPLES}"
echo "GPU memory peak:      ${GPU_MEMORY_PEAK}"
echo "Checkpoint:           ${CHECKPOINT}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Start/limit:          ${START_OFFSET}/${LIMIT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "============================================"
nvidia-smi || true

if [[ ! -x "${VIDEOPHY_PYTHON}" ]]; then
  echo "Missing VIDEOPHY_PYTHON: ${VIDEOPHY_PYTHON}" >&2
  echo "Run ./benchmarks/pbench/setup_videophy_env.sh first." >&2
  exit 1
fi
if [[ ! -d "${VIDEOPHY_DIR}" ]]; then
  echo "Missing VIDEOPHY_DIR: ${VIDEOPHY_DIR}" >&2
  echo "Run ./benchmarks/pbench/setup_videophy_env.sh first." >&2
  exit 1
fi

start_gpu_monitor
trap stop_gpu_monitor EXIT

VIDEOPHY_OUTPUT_ROOT="${OUTPUT_ROOT}/videophy"
PHYSICS_CSV="${VIDEOPHY_OUTPUT_ROOT}/${RUN_NAME}_physics_testing.csv"
PA_II_RAW_CSV="${OUTPUT_ROOT}/${RUN_NAME}_pa_ii_raw.csv"
PA_II_CSV="${OUTPUT_ROOT}/${RUN_NAME}_pa_ii.csv"

"${VIDEOPHY_PYTHON}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_dreamgenbench_videophy_input.py" \
  --video-dir "${VIDEO_DIR}" \
  --output-root "${VIDEOPHY_OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --start-offset "${START_OFFSET}" \
  --limit "${LIMIT}"

cd "${VIDEOPHY_DIR}/videocon/training/pipeline_video"
export PYTHONPATH="${VIDEOPHY_DIR}/videocon/training/pipeline_video:${VIDEOPHY_DIR}/VIDEOPHY2:${VIDEOPHY_DIR}:${PYTHONPATH:-}"

"${VIDEOPHY_PYTHON}" "${EVEWORLD_ROOT}/benchmarks/pbench/run_videophy_pa2.py" \
  --input_csv "${PHYSICS_CSV}" \
  --output_csv "${PA_II_RAW_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --batch_size "${BATCH_SIZE}"

"${VIDEOPHY_PYTHON}" "${EVEWORLD_ROOT}/benchmarks/pbench/convert_videophy_pa2_csv.py" \
  --raw-csv "${PA_II_RAW_CSV}" \
  --output-csv "${PA_II_CSV}" \
  --threshold 0.5

echo "VideoPhy PA-II finished."
echo "PA-II raw CSV: ${PA_II_RAW_CSV}"
echo "PA-II CSV:     ${PA_II_CSV}"
stop_gpu_monitor
