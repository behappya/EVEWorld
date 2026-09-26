#!/usr/bin/env bash
#SBATCH --job-name=gw0_video_serve
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  cat >&2 <<'EOF'
Refusing to run GigaWorld-0 serving payload on the workspace host.

This script loads the full model and is intended to run only inside a kjob/Slurm pod.
Submit it with:
  ./scripts/launch_gigaworld0_serve_kjob.sh

If you intentionally want to run it on this host, set ALLOW_LOCAL_RUN=1.
EOF
  exit 2
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    echo "Pass overrides as KEY=VALUE, for example: PORT=8000 GPU_IDS=0" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"

MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gigaworld0_serving}"
RUN_NAME="${RUN_NAME:-serve_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
SERVE_LOG="${SERVE_LOG:-${LOG_FILE:-${SAVE_DIR}/serve.log}}"
ENV_FILE="${ENV_FILE:-${SAVE_DIR}/serve.env}"
RESULTS_DIR="${RESULTS_DIR:-${SAVE_DIR}/results}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${SAVE_DIR}/gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${SAVE_DIR}/gpu_memory_peak.json}"
GPU_MONITOR_PID=""

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
GPU_IDS_RAW="${GPU_IDS:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.nv/ComputeCache}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

IFS=' ,' read -r -a REQUESTED_GPU_IDS_ARGS <<< "${GPU_IDS_RAW}"
if [[ ${#REQUESTED_GPU_IDS_ARGS[@]} -eq 0 ]]; then
  echo "GPU_IDS resolved to an empty list." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${REQUESTED_GPU_IDS_ARGS[*]}")"
export CUDA_VISIBLE_DEVICES

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_any_file() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [[ -f "${path}" ]]; then
      return 0
    fi
  done
  echo "Missing required file for ${label}. Checked:" >&2
  for path in "$@"; do
    echo "  ${path}" >&2
  done
  exit 1
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

log_mem() {
  echo "---- memory: $1 ----"
  if [[ -f /sys/fs/cgroup/memory.current ]]; then
    echo "cgroup memory.current: $(cat /sys/fs/cgroup/memory.current)"
    echo "cgroup memory.max:     $(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"
  fi
  free -h || true
  echo "---------------------"
}

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
    --label "gigaworld0_serve_${RUN_NAME}" &
  GPU_MONITOR_PID="$!"
  echo "GPU memory monitor pid: ${GPU_MONITOR_PID}"
  echo "GPU memory samples:     ${GPU_MEMORY_SAMPLES}"
  echo "GPU memory peak:        ${GPU_MEMORY_PEAK}"
}

require_dir "${REPO_DIR}"
require_dir "${MODEL_DIR}"
require_file "${MODEL_DIR}/transformer/config.json"
require_any_file "transformer weights" \
  "${MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors" \
  "${MODEL_DIR}/transformer/diffusion_pytorch_model.bin"
require_file "${MODEL_DIR}/text_encoder/config.json"
require_file "${MODEL_DIR}/text_encoder/pytorch_model.bin"
require_file "${MODEL_DIR}/vae/config.json"
require_any_file "VAE weights" \
  "${MODEL_DIR}/vae/diffusion_pytorch_model.safetensors" \
  "${MODEL_DIR}/vae/diffusion_pytorch_model.bin"

mkdir -p "${SAVE_DIR}" "${RESULTS_DIR}" "${CUDA_CACHE_PATH}"
touch "${SERVE_LOG}"
exec > >(tee -a "${SERVE_LOG}") 2>&1
echo "Persistent log: ${SERVE_LOG}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

LOCAL_IP="$(hostname -I | awk '{print $1}')"
cat > "${ENV_FILE}" <<EOF
REPO_DIR=${REPO_DIR}
GIGA_MODELS_DIR=${GIGA_MODELS_DIR}
CONDA_ENV=${CONDA_ENV}
MODEL_DIR=${MODEL_DIR}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_NAME=${RUN_NAME}
SAVE_DIR=${SAVE_DIR}
SERVE_LOG=${SERVE_LOG}
ENV_FILE=${ENV_FILE}
RESULTS_DIR=${RESULTS_DIR}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}
GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}
GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}
HOST=${HOST}
PORT=${PORT}
LOCAL_IP=${LOCAL_IP}
SERVING_URL=http://${LOCAL_IP}:${PORT}
GPU_IDS=${GPU_IDS_RAW}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
DEVICE=${DEVICE}
HF_HOME=${HF_HOME}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
DIFFUSERS_OFFLINE=${DIFFUSERS_OFFLINE}
EOF

echo "============================================"
echo "GigaWorld-0 video_pretrain HTTP serving"
echo "Host:                    $(hostname)"
echo "Local IP:                ${LOCAL_IP}"
echo "Serving URL:             http://${LOCAL_IP}:${PORT}"
echo "Repo:                    ${REPO_DIR}"
echo "Model dir:               ${MODEL_DIR}"
echo "Save dir:                ${SAVE_DIR}"
echo "Results dir:             ${RESULTS_DIR}"
echo "GPU memory samples:      ${GPU_MEMORY_SAMPLES}"
echo "GPU memory peak:         ${GPU_MEMORY_PEAK}"
echo "Env file:                ${ENV_FILE}"
echo "Serve log:               ${SERVE_LOG}"
echo "Requested GPU ids:       ${REQUESTED_GPU_IDS_ARGS[*]}"
echo "CUDA_VISIBLE_DEVICES:    ${CUDA_VISIBLE_DEVICES}"
echo "Device:                  ${DEVICE}"
echo "============================================"

log_mem "serve job start"
nvidia-smi || true
start_gpu_monitor
python scripts/check_cuda.py
log_mem "after cuda check"

exec python scripts/serve_gigaworld0.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --transformer-model-path "${MODEL_DIR}/transformer" \
  --text-encoder-model-path "${MODEL_DIR}/text_encoder" \
  --vae-model-path "${MODEL_DIR}/vae" \
  --default-save-dir "${RESULTS_DIR}"
