#!/usr/bin/env bash
#SBATCH --job-name=gw0_video_pretrain
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  cat >&2 <<'EOF'
Refusing to run GigaWorld-0 inference payload on the workspace host.

This script loads the full model and is intended to run only inside a kjob/Slurm pod.
Submit it with one of:
  ./scripts/submit_gigaworld0_kjob.sh
  ./benchmarks/pbench/launch_pbench_robot_kjob.sh

If you intentionally want to run it on this host, set ALLOW_LOCAL_RUN=1.
EOF
  exit 2
fi

# Accept KEY=VALUE overrides as script arguments. This avoids relying on
# kjobctl's slurm export flags, which are cluster-wrapper specific.
for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    echo "Pass overrides as KEY=VALUE, for example: DATA_LIMIT=0 NUM_INFERENCE_STEPS=30" >&2
    exit 1
  fi
  export "${arg}"
done

# Single-node GigaWorld-0 video_pretrain inference.
#
# Default mode is a cheap smoke test:
#   - uses assets/it2v.json
#   - keeps only the first sample
#   - runs 1 diffusion step
#
# Full demo:
#   ./scripts/kjob_gigaworld0_video_pretrain_infer.sh DATA_LIMIT=0 NUM_INFERENCE_STEPS=30

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"

MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/assets/it2v.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs}"
RUN_NAME="${RUN_NAME:-video_pretrain_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/run.log}"
ENV_FILE="${ENV_FILE:-${SAVE_DIR}/run.env}"
SUMMARY_PATH="${SUMMARY_PATH:-${SAVE_DIR}/generation_summary.json}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${SAVE_DIR}/gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${SAVE_DIR}/gpu_memory_peak.json}"
GPU_MONITOR_PID=""

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
DATA_LIMIT="${DATA_LIMIT:-1}"
ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE:-0}"
if [[ "${DATA_LIMIT}" != "0" && "${ALLOW_MULTI_GPU_SMOKE}" != "1" ]]; then
  GPU_IDS_RAW="0"
elif [[ -n "${GPU_IDS:-}" ]]; then
  GPU_IDS_RAW="${GPU_IDS}"
else
  GPU_IDS_RAW="0"
fi
FPS="${FPS:-16}"
NUM_FRAMES="${NUM_FRAMES:-61}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"
SEED="${SEED:-6666}"

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

PYTHON_GPU_IDS_ARGS=()
for idx in "${!REQUESTED_GPU_IDS_ARGS[@]}"; do
  PYTHON_GPU_IDS_ARGS+=("${idx}")
done

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
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
    --label "pbench_robot_generation_${RUN_NAME}" &
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

require_dir "${REPO_DIR}"
require_dir "${MODEL_DIR}"
require_file "${MODEL_DIR}/transformer/config.json"
if [[ ! -f "${MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors" && ! -f "${MODEL_DIR}/transformer/diffusion_pytorch_model.bin" ]]; then
  echo "Missing transformer weights: ${MODEL_DIR}/transformer/diffusion_pytorch_model.{safetensors,bin}" >&2
  exit 1
fi
require_file "${MODEL_DIR}/text_encoder/config.json"
require_file "${MODEL_DIR}/text_encoder/pytorch_model.bin"
require_file "${MODEL_DIR}/vae/config.json"
require_file "${MODEL_DIR}/vae/diffusion_pytorch_model.safetensors"
require_file "${DATA_PATH}"

mkdir -p "${SAVE_DIR}" "${CUDA_CACHE_PATH}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Persistent log: ${LOG_FILE}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

EFFECTIVE_DATA_PATH="${DATA_PATH}"
if [[ "${DATA_LIMIT}" != "0" ]]; then
  EFFECTIVE_DATA_PATH="${SAVE_DIR}/input_first_${DATA_LIMIT}.json"
  python -c 'import json, os, sys; src, dst, limit_s = sys.argv[1:4]; limit = int(limit_s); data = json.load(open(src, "r"))[:limit]; base = os.path.dirname(os.path.abspath(src)); [item.__setitem__("image", os.path.abspath(os.path.join(base, item["image"]))) for item in data if item.get("image") and not os.path.isabs(item["image"])]; json.dump(data, open(dst, "w"), indent=2)' \
    "${DATA_PATH}" "${EFFECTIVE_DATA_PATH}" "${DATA_LIMIT}"
fi

cat > "${ENV_FILE}" <<EOF
REPO_DIR=${REPO_DIR}
GIGA_MODELS_DIR=${GIGA_MODELS_DIR}
CONDA_ENV=${CONDA_ENV}
MODEL_DIR=${MODEL_DIR}
DATA_PATH=${DATA_PATH}
EFFECTIVE_DATA_PATH=${EFFECTIVE_DATA_PATH}
SAVE_DIR=${SAVE_DIR}
LOG_FILE=${LOG_FILE}
SUMMARY_PATH=${SUMMARY_PATH}
GPU_IDS=${GPU_IDS_RAW}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
PYTHON_GPU_IDS=${PYTHON_GPU_IDS_ARGS[*]}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}
GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}
GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}
DATA_LIMIT=${DATA_LIMIT}
FPS=${FPS}
NUM_FRAMES=${NUM_FRAMES}
HEIGHT=${HEIGHT}
WIDTH=${WIDTH}
SEED=${SEED}
HF_HOME=${HF_HOME}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
DIFFUSERS_OFFLINE=${DIFFUSERS_OFFLINE}
EOF

echo "============================================"
echo "GigaWorld-0 video_pretrain inference"
echo "Host:                    $(hostname)"
echo "Repo:                    ${REPO_DIR}"
echo "Model dir:               ${MODEL_DIR}"
echo "Data path:               ${EFFECTIVE_DATA_PATH}"
echo "Save dir:                ${SAVE_DIR}"
echo "Env file:                ${ENV_FILE}"
echo "Log file:                ${LOG_FILE}"
echo "Summary:                 ${SUMMARY_PATH}"
echo "Requested GPU ids:       ${REQUESTED_GPU_IDS_ARGS[*]}"
echo "CUDA_VISIBLE_DEVICES:    ${CUDA_VISIBLE_DEVICES}"
echo "Python GPU ids:          ${PYTHON_GPU_IDS_ARGS[*]}"
echo "GPU memory samples:      ${GPU_MEMORY_SAMPLES}"
echo "GPU memory peak:         ${GPU_MEMORY_PEAK}"
echo "Steps:                   ${NUM_INFERENCE_STEPS}"
echo "Frames/FPS/Size:         ${NUM_FRAMES}/${FPS}/${HEIGHT}x${WIDTH}"
echo "Seed:                    ${SEED}"
echo "============================================"

log_mem "job start"
nvidia-smi || true
start_gpu_monitor
trap stop_gpu_monitor EXIT

python scripts/check_cuda.py
log_mem "after cuda check"

GENERATION_RC=0
python scripts/inference.py \
  --data-path "${EFFECTIVE_DATA_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --transformer-model-path "${MODEL_DIR}/transformer" \
  --text-encoder-model-path "${MODEL_DIR}/text_encoder" \
  --vae-model-path "${MODEL_DIR}/vae" \
  --gpu-ids "${PYTHON_GPU_IDS_ARGS[@]}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --fps "${FPS}" \
  --num-frames "${NUM_FRAMES}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --seed "${SEED}" \
  --summary-path "${SUMMARY_PATH}" || GENERATION_RC=$?
stop_gpu_monitor
exit "${GENERATION_RC}"
