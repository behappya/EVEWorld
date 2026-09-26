#!/usr/bin/env bash
#SBATCH --job-name=gw0_gr1_train
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  cat >&2 <<'EOF'
Refusing to run GigaWorld-0 GR1 training payload on the workspace host.

This script loads the full video model and should run inside a GPU kjob.
Submit it with:
  ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh
EOF
  exit 2
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    echo "Pass overrides as KEY=VALUE, for example: MAX_STEPS=1 GPU_IDS=0" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
TRAIN_ACCELERATE="${TRAIN_ACCELERATE:-${TRAIN_VENV}/bin/accelerate}"

DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_finetune_data}"
PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
TRANSFORMER_MODEL_PATH="${TRANSFORMER_MODEL_PATH:-${MODEL_DIR}/transformer}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-${MODEL_DIR}/vae}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune}"
TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR:-${OUTPUT_ROOT}/experiments}"
RUN_NAME="${RUN_NAME:-train_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-${OUTPUT_ROOT}/runtime_configs}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${RUNTIME_CONFIG_DIR}/${RUN_NAME}.json}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${LOG_DIR}/${RUN_NAME}_gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${LOG_DIR}/${RUN_NAME}_gpu_memory_peak.json}"
GPU_MONITOR_PID=""

BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-configs.giga_world_0_video_gr1_finetune}"
MAX_STEPS="${MAX_STEPS:-200}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-5}"
NUM_WORKERS="${NUM_WORKERS:-6}"
NUM_FRAMES="${NUM_FRAMES:-93}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
FPS="${FPS:-16}"
SEED="${SEED:-6666}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WITH_EMA="${WITH_EMA:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
GPU_IDS_RAW="${GPU_IDS:-0 1 2 3 4 5 6 7}"
CONFIG_DRY_RUN="${CONFIG_DRY_RUN:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.nv/ComputeCache}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"
export DS_BUILD_FP_QUANTIZER="${DS_BUILD_FP_QUANTIZER:-0}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

IFS=' ,' read -r -a REQUESTED_GPU_IDS_ARGS <<< "${GPU_IDS_RAW}"
if [[ ${#REQUESTED_GPU_IDS_ARGS[@]} -eq 0 ]]; then
  echo "GPU_IDS resolved to an empty list." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${REQUESTED_GPU_IDS_ARGS[*]}")"
export CUDA_VISIBLE_DEVICES

PYTHON_GPU_IDS_ARGS=()
for idx in "${!REQUESTED_GPU_IDS_ARGS[@]}"; do
  PYTHON_GPU_IDS_ARGS+=("${idx}")
done

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
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
  "${TRAIN_PYTHON}" "${REPO_DIR}/scripts/gpu_memory_monitor.py" \
    --samples-csv "${GPU_MEMORY_SAMPLES}" \
    --peak-json "${GPU_MEMORY_PEAK}" \
    --interval-sec "${GPU_MONITOR_INTERVAL}" \
    --label "gr1_train_${RUN_NAME}" &
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

if [[ ! -x "${TRAIN_PYTHON}" ]]; then
  echo "Missing training Python: ${TRAIN_PYTHON}" >&2
  echo "Run first: ${REPO_DIR}/scripts/setup_gigaworld_train_venv.sh" >&2
  exit 1
fi
if [[ ! -x "${TRAIN_ACCELERATE}" ]]; then
  echo "Missing training accelerate executable: ${TRAIN_ACCELERATE}" >&2
  echo "Run first: ${REPO_DIR}/scripts/setup_gigaworld_train_venv.sh" >&2
  exit 1
fi

require_dir "${REPO_DIR}"
require_dir "${PACKED_DATA_DIR}"
require_file "${PACKED_DATA_DIR}/config.json"
require_dir "${TRANSFORMER_MODEL_PATH}"
require_file "${TRANSFORMER_MODEL_PATH}/config.json"
require_file "${TRANSFORMER_MODEL_PATH}/diffusion_pytorch_model.safetensors"
require_dir "${VAE_MODEL_PATH}"
require_file "${VAE_MODEL_PATH}/config.json"
require_file "${VAE_MODEL_PATH}/diffusion_pytorch_model.safetensors"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${TRAIN_PROJECT_DIR}" "${RUNTIME_CONFIG_DIR}" \
  "${HF_HOME}" "${HF_XET_CACHE}" "${CUDA_CACHE_PATH}"
touch "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Persistent log: ${RUN_LOG}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

# shellcheck disable=SC1090
source "${TRAIN_VENV}/bin/activate"

cd "${REPO_DIR}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in \
    /usr/local/cuda \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12 \
    "${CONDA_PREFIX:-}/targets/x86_64-linux" \
    "${CONDA_PREFIX:-}"; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    export CUDA_HOME
  fi
fi

cat > "${ENV_FILE}" <<EOF
REPO_DIR=${REPO_DIR}
GIGA_MODELS_DIR=${GIGA_MODELS_DIR}
CONDA_ENV=${CONDA_ENV}
TRAIN_VENV=${TRAIN_VENV}
TRAIN_PYTHON=${TRAIN_PYTHON}
TRAIN_ACCELERATE=${TRAIN_ACCELERATE}
PACKED_DATA_DIR=${PACKED_DATA_DIR}
MODEL_DIR=${MODEL_DIR}
TRANSFORMER_MODEL_PATH=${TRANSFORMER_MODEL_PATH}
VAE_MODEL_PATH=${VAE_MODEL_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT}
TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}
RUN_NAME=${RUN_NAME}
RUN_LOG=${RUN_LOG}
ENV_FILE=${ENV_FILE}
RUNTIME_CONFIG=${RUNTIME_CONFIG}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}
GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}
GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}
GPU_IDS=${GPU_IDS_RAW}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
PYTHON_GPU_IDS=${PYTHON_GPU_IDS_ARGS[*]}
MAX_STEPS=${MAX_STEPS}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}
CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}
NUM_FRAMES=${NUM_FRAMES}
HEIGHT=${HEIGHT}
WIDTH=${WIDTH}
FPS=${FPS}
SEED=${SEED}
MIXED_PRECISION=${MIXED_PRECISION}
WITH_EMA=${WITH_EMA}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
HF_HOME=${HF_HOME}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
DIFFUSERS_OFFLINE=${DIFFUSERS_OFFLINE}
CUDA_HOME=${CUDA_HOME:-}
DS_BUILD_OPS=${DS_BUILD_OPS}
DS_BUILD_FP_QUANTIZER=${DS_BUILD_FP_QUANTIZER}
DS_IGNORE_CUDA_DETECTION=${DS_IGNORE_CUDA_DETECTION}
EOF

EFFECTIVE_BATCH_SIZE=$((${#REQUESTED_GPU_IDS_ARGS[@]} * BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))

echo "============================================"
echo "GigaWorld-0 GR1 fine-tune kjob"
echo "Host:                    $(hostname)"
echo "Repo:                    ${REPO_DIR}"
echo "Train venv:              ${TRAIN_VENV}"
echo "Train python:            ${TRAIN_PYTHON}"
echo "Train accelerate:        ${TRAIN_ACCELERATE}"
echo "Packed data:             ${PACKED_DATA_DIR}"
echo "Transformer:             ${TRANSFORMER_MODEL_PATH}"
echo "VAE:                     ${VAE_MODEL_PATH}"
echo "Project dir:             ${TRAIN_PROJECT_DIR}"
echo "Run log:                 ${RUN_LOG}"
echo "Env file:                ${ENV_FILE}"
echo "Runtime config:          ${RUNTIME_CONFIG}"
echo "GPU memory samples:      ${GPU_MEMORY_SAMPLES}"
echo "GPU memory peak:         ${GPU_MEMORY_PEAK}"
echo "Requested GPU ids:       ${REQUESTED_GPU_IDS_ARGS[*]}"
echo "CUDA_VISIBLE_DEVICES:    ${CUDA_VISIBLE_DEVICES}"
echo "Python GPU ids:          ${PYTHON_GPU_IDS_ARGS[*]}"
echo "Steps:                   ${MAX_STEPS}"
echo "Batch/GPU x accum:       ${BATCH_SIZE_PER_GPU} x ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch size:    ${EFFECTIVE_BATCH_SIZE}"
echo "Frames/FPS/Size:         ${NUM_FRAMES}/${FPS}/${HEIGHT}x${WIDTH}"
echo "Seed:                    ${SEED}"
echo "CUDA_HOME:               ${CUDA_HOME:-<unset>}"
echo "DeepSpeed CUDA detect:   ${DS_IGNORE_CUDA_DETECTION}"
echo "============================================"

if [[ "${EFFECTIVE_BATCH_SIZE}" != "64" ]]; then
  echo "WARNING: effective batch size is ${EFFECTIVE_BATCH_SIZE}, not paper value 64."
fi

log_mem "job start"
nvidia-smi || true
start_gpu_monitor
trap stop_gpu_monitor EXIT

if [[ "${CONFIG_DRY_RUN}" == "1" ]]; then
  echo "CONFIG_DRY_RUN=1; skipping CUDA initialization check."
else
  "${TRAIN_PYTHON}" scripts/check_cuda.py
  log_mem "after cuda check"
fi

echo "Training dependency import check..."
"${TRAIN_PYTHON}" - <<'PY'
import sys
import torch
import transformers
from transformers import Dinov2WithRegistersConfig  # noqa: F401
from transformers.models import Qwen2_5_VLProcessor, Qwen2_5_VLTextModel  # noqa: F401
from giga_world_0 import GigaWorld0Trainer  # noqa: F401

print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("giga_world_0 trainer import: ok")
PY

echo "Writing runtime config..."
"${TRAIN_PYTHON}" scripts/write_train_runtime_config.py \
  "${RUNTIME_CONFIG}" \
  "${BASE_CONFIG_MODULE}" \
  "${TRAIN_PROJECT_DIR}" \
  "${PACKED_DATA_DIR}" \
  "${TRANSFORMER_MODEL_PATH}" \
  "${VAE_MODEL_PATH}" \
  "${MAX_STEPS}" \
  "${BATCH_SIZE_PER_GPU}" \
  "${GRADIENT_ACCUMULATION_STEPS}" \
  "${CHECKPOINT_INTERVAL}" \
  "${CHECKPOINT_TOTAL_LIMIT}" \
  "${NUM_WORKERS}" \
  "${NUM_FRAMES}" \
  "${HEIGHT}" \
  "${WIDTH}" \
  "${FPS}" \
  "${SEED}" \
  "${MIXED_PRECISION}" \
  "${WITH_EMA}" \
  "${ACTIVATION_CHECKPOINTING}" \
  "${TRAIN_ACCELERATE}" \
  "${PYTHON_GPU_IDS_ARGS[@]}"

echo "Runtime config preview:"
cat "${RUNTIME_CONFIG}"

if [[ "${CONFIG_DRY_RUN}" == "1" ]]; then
  echo
  echo "CONFIG_DRY_RUN=1; runtime config was written and training will not start."
  exit 0
fi

echo
echo "Starting training..."
TRAIN_RC=0
"${TRAIN_PYTHON}" scripts/train.py --config "${RUNTIME_CONFIG}" || TRAIN_RC=$?
if [[ "${TRAIN_RC}" != "0" ]]; then
  echo "Training command failed with exit code ${TRAIN_RC}" >&2
  exit "${TRAIN_RC}"
fi

if [[ ! -d "${TRAIN_PROJECT_DIR}/logs" ]]; then
  echo "Training did not create GigaTrain logs: ${TRAIN_PROJECT_DIR}/logs" >&2
  exit 1
fi

LATEST_TRAIN_LOG="$(find "${TRAIN_PROJECT_DIR}/logs" -maxdepth 1 -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "${LATEST_TRAIN_LOG}" ]]; then
  echo "No GigaTrain train_*.log found under ${TRAIN_PROJECT_DIR}/logs" >&2
  exit 1
fi

if ! grep -R "Step\\[${MAX_STEPS}/${MAX_STEPS}\\]" "${TRAIN_PROJECT_DIR}/logs" >/dev/null 2>&1; then
  echo "Training logs do not contain expected completion marker: Step[${MAX_STEPS}/${MAX_STEPS}]" >&2
  echo "Latest train log: ${LATEST_TRAIN_LOG}" >&2
  tail -120 "${LATEST_TRAIN_LOG}" >&2 || true
  exit 1
fi

if ! find "${TRAIN_PROJECT_DIR}/models" -maxdepth 1 -type d -name "*step_${MAX_STEPS}" | grep -q .; then
  echo "Training did not create expected checkpoint ending with step_${MAX_STEPS}" >&2
  find "${TRAIN_PROJECT_DIR}/models" -maxdepth 1 -type d -name 'checkpoint*' -printf '%p\n' 2>/dev/null >&2 || true
  exit 1
fi

echo "Training finished and verified."
echo "Project dir: ${TRAIN_PROJECT_DIR}"
echo "GigaTrain logs:"
find "${TRAIN_PROJECT_DIR}/logs" -maxdepth 1 -type f -name 'train_*.log' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -5 || true
echo "Checkpoints:"
find "${TRAIN_PROJECT_DIR}/models" -maxdepth 1 -type d -name 'checkpoint*' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -10 || true
stop_gpu_monitor
log_mem "job end"
