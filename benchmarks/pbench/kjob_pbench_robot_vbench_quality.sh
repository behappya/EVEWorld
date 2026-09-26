#!/usr/bin/env bash
#SBATCH --job-name=pbench_vbench_quality
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  cat >&2 <<'EOF'
Refusing to run PBench VBench quality payload on the workspace host.

This script loads VBench/DreamSim/CLIP/DINO metric models and should run inside a GPU kjob.
Submit it with:
  ./benchmarks/pbench/launch_pbench_robot_vbench_quality_kjob.sh
EOF
  exit 2
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    echo "Pass overrides as KEY=VALUE, for example: LIMIT=1 GPU_IDS=0" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality}"
RUN_NAME="${RUN_NAME:-quality_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${LOG_DIR}/${RUN_NAME}_gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${LOG_DIR}/${RUN_NAME}_gpu_memory_peak.json}"
GPU_MONITOR_PID=""

METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_serving_full_20260612_145541}"
DOMAIN_SUMMARY="${DOMAIN_SUMMARY:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval/20260613_131107_qwen36vl/qwen_vqa_summary.json}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/data/datasets/gagi/vbench_cache}"
GPU_IDS_RAW="${GPU_IDS:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/datasets/gagi/.cache}"
export TORCH_HOME="${TORCH_HOME:-/data/datasets/gagi/torch_cache}"
export VBENCH_CACHE_DIR
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

IFS=' ,' read -r -a REQUESTED_GPU_IDS_ARGS <<< "${GPU_IDS_RAW}"
if [[ ${#REQUESTED_GPU_IDS_ARGS[@]} -eq 0 ]]; then
  echo "GPU_IDS resolved to an empty list." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${REQUESTED_GPU_IDS_ARGS[*]}")"
export CUDA_VISIBLE_DEVICES

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${VBENCH_CACHE_DIR}" "${TORCH_HOME}" "${HF_HOME}" "${HF_XET_CACHE}"
touch "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Persistent log: ${RUN_LOG}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; GPU memory monitor disabled."
    return 0
  fi
  mkdir -p "$(dirname "${GPU_MEMORY_SAMPLES}")" "$(dirname "${GPU_MEMORY_PEAK}")"
  python "${REPO_DIR}/scripts/gpu_memory_monitor.py" \
    --samples-csv "${GPU_MEMORY_SAMPLES}" \
    --peak-json "${GPU_MEMORY_PEAK}" \
    --interval-sec "${GPU_MONITOR_INTERVAL}" \
    --label "pbench_vbench_quality_${RUN_NAME}" &
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

LOCAL_IP="$(hostname -I | awk '{print $1}')"
cat > "${ENV_FILE}" <<EOF
REPO_DIR=${REPO_DIR}
CONDA_ENV=${CONDA_ENV}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_NAME=${RUN_NAME}
RUN_LOG=${RUN_LOG}
ENV_FILE=${ENV_FILE}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}
GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}
GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}
METADATA_JSONL=${METADATA_JSONL}
SOURCE_VIDEO_DIR=${SOURCE_VIDEO_DIR}
DOMAIN_SUMMARY=${DOMAIN_SUMMARY}
VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR}
TORCH_HOME=${TORCH_HOME}
HF_HOME=${HF_HOME}
LOCAL_IP=${LOCAL_IP}
GPU_IDS=${GPU_IDS_RAW}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EOF

echo "============================================"
echo "PBench Robot VBench quality kjob"
echo "Host:                 $(hostname)"
echo "Local IP:             ${LOCAL_IP}"
echo "Repo:                 ${REPO_DIR}"
echo "Output root:          ${OUTPUT_ROOT}"
echo "Metadata:             ${METADATA_JSONL}"
echo "Source videos:        ${SOURCE_VIDEO_DIR}"
echo "Domain summary:       ${DOMAIN_SUMMARY}"
echo "VBench cache:         ${VBENCH_CACHE_DIR}"
echo "Run log:              ${RUN_LOG}"
echo "Env file:             ${ENV_FILE}"
echo "GPU memory samples:   ${GPU_MEMORY_SAMPLES}"
echo "GPU memory peak:      ${GPU_MEMORY_PEAK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Limit:                ${LIMIT:-0}"
echo "Dimensions:           ${DIMENSIONS:-default}"
echo "============================================"

nvidia-smi || true
start_gpu_monitor
trap stop_gpu_monitor EXIT

QUALITY_RC=0
./benchmarks/pbench/eval_pbench_robot_vbench_quality.sh || QUALITY_RC=$?
stop_gpu_monitor
exit "${QUALITY_RC}"
