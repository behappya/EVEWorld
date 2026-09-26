#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for GigaWorld-0 on the PBench Robotics subset.
#
# Defaults to a cheap smoke run:
#   ./benchmarks/pbench/launch_pbench_robot_kjob.sh
#
# Full Robotics subset:
#   MODE=full ./benchmarks/pbench/launch_pbench_robot_kjob.sh
#
# Useful overrides:
#   DATA_LIMIT=5 NUM_INFERENCE_STEPS=5 ./benchmarks/pbench/launch_pbench_robot_kjob.sh
#   GPU_IDS=0 MODE=full ./benchmarks/pbench/launch_pbench_robot_kjob.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${MODE:-smoke}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

DATA_PATH="${DATA_PATH:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json}"
PARQUET_PATH="${PARQUET_PATH:-/home/jovyan/gagibench/pbench_raw/data/pbench.parquet}"
PBench_OUTPUT_ROOT="${PBENCH_OUTPUT_ROOT:-/home/jovyan/gagibench/pbench/giga_input}"
PREPARE_PBENCH_INPUT="${PREPARE_PBENCH_INPUT:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot}"
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_${MODE}}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/run.log}"
ENV_FILE="${ENV_FILE:-${SAVE_DIR}/run.env}"
SUBMIT_LOG="${SUBMIT_LOG:-${SAVE_DIR}/submit.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${SAVE_DIR}/generation_summary.json}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${SAVE_DIR}/gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${SAVE_DIR}/gpu_memory_peak.json}"

case "${MODE}" in
  smoke)
    DATA_LIMIT="${DATA_LIMIT:-1}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
    ;;
  full)
    DATA_LIMIT="${DATA_LIMIT:-0}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use MODE=smoke or MODE=full." >&2
    exit 1
    ;;
esac

mkdir -p "${SAVE_DIR}"

if [[ ! -f "${DATA_PATH}" ]]; then
  if [[ "${PREPARE_PBENCH_INPUT}" != "1" ]]; then
    echo "Missing PBench GigaWorld input JSON: ${DATA_PATH}" | tee -a "${SUBMIT_LOG}" >&2
    echo "Prepare it explicitly first, or rerun with PREPARE_PBENCH_INPUT=1." | tee -a "${SUBMIT_LOG}" >&2
    exit 1
  fi
  if [[ ! -f "${PARQUET_PATH}" ]]; then
    echo "Missing PBench parquet: ${PARQUET_PATH}" | tee -a "${SUBMIT_LOG}" >&2
    exit 1
  fi
  echo "Missing ${DATA_PATH}; preparing PBench Robotics input first." | tee -a "${SUBMIT_LOG}"
  CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
  CONDA_ENV="${CONDA_ENV:-EVEWorld}"
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  fi
  python "${SCRIPT_DIR}/prepare_pbench_it2v.py" \
    --parquet "${PARQUET_PATH}" \
    --output-root "${PBench_OUTPUT_ROOT}" \
    --subset robot
fi

cat > "${SAVE_DIR}/launcher.env" <<EOF
MODE=${MODE}
TIMESTAMP=${TIMESTAMP}
REPO_DIR=${REPO_DIR}
DATA_PATH=${DATA_PATH}
PARQUET_PATH=${PARQUET_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_NAME=${RUN_NAME}
SAVE_DIR=${SAVE_DIR}
LOG_FILE=${LOG_FILE}
ENV_FILE=${ENV_FILE}
SUBMIT_LOG=${SUBMIT_LOG}
SUMMARY_PATH=${SUMMARY_PATH}
DATA_LIMIT=${DATA_LIMIT}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}
GPU_IDS=${GPU_IDS:-}
ALLOW_MULTI_GPU_SMOKE=${ALLOW_MULTI_GPU_SMOKE:-}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}
GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}
GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}
MODEL_DIR=${MODEL_DIR:-}
FPS=${FPS:-16}
NUM_FRAMES=${NUM_FRAMES:-61}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-640}
SEED=${SEED:-6666}
PREPARE_PBENCH_INPUT=${PREPARE_PBENCH_INPUT}
EOF

{
  echo "============================================"
  echo "Launching PBench Robot GigaWorld-0 kjob"
  echo "Mode:          ${MODE}"
  echo "Repo:          ${REPO_DIR}"
  echo "Data path:     ${DATA_PATH}"
  echo "Save dir:      ${SAVE_DIR}"
  echo "Submit log:    ${SUBMIT_LOG}"
  echo "Run log:       ${LOG_FILE}"
  echo "Summary:       ${SUMMARY_PATH}"
  echo "Env file:      ${ENV_FILE}"
  echo "Launcher env:  ${SAVE_DIR}/launcher.env"
  echo "Data limit:    ${DATA_LIMIT}"
  echo "Steps:         ${NUM_INFERENCE_STEPS}"
  echo "Frames/FPS:    ${NUM_FRAMES:-61}/${FPS:-16}"
  echo "Model dir:     ${MODEL_DIR:-default}"
  echo "GPU ids:       ${GPU_IDS:-default}"
  echo "Multi-GPU smoke: ${ALLOW_MULTI_GPU_SMOKE:-default}"
  echo "GPU samples:   ${GPU_MEMORY_SAMPLES}"
  echo "GPU peak:      ${GPU_MEMORY_PEAK}"
  echo "Note:          smoke still loads the full 94G model; 1 step only reduces generation time."
  echo "============================================"
  echo
  echo "After the job starts, watch logs with:"
  echo "  tail -f ${LOG_FILE}"
  echo
  echo "If Kubernetes pod logs are needed:"
  echo "  kubectl get pods --sort-by=.metadata.creationTimestamp | tail -n 20"
  echo "  kubectl logs -f POD_NAME -c job-container"
  echo
} | tee -a "${SUBMIT_LOG}"

export REPO_DIR
export MODEL_DIR="${MODEL_DIR:-}"
export DATA_PATH
export OUTPUT_ROOT
export RUN_NAME
export SAVE_DIR
export LOG_FILE
export ENV_FILE
export SUMMARY_PATH
export DATA_LIMIT
export NUM_INFERENCE_STEPS
export GPU_IDS="${GPU_IDS:-}"
export ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE:-}"
export GPU_MONITOR_INTERVAL
export GPU_MEMORY_SAMPLES
export GPU_MEMORY_PEAK
export FPS="${FPS:-16}"
export NUM_FRAMES="${NUM_FRAMES:-61}"
export HEIGHT="${HEIGHT:-480}"
export WIDTH="${WIDTH:-640}"
export SEED="${SEED:-6666}"

"${SCRIPT_DIR}/submit_gigaworld0_kjob.sh" 2>&1 | tee -a "${SUBMIT_LOG}"
