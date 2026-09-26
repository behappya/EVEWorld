#!/usr/bin/env bash
#SBATCH --job-name=gw0_gr1_dreamgen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  cat >&2 <<'EOF'
Refusing to run GR1 DreamGen generation on the workspace host.

This payload loads the full model and should run inside a GPU kjob.
Submit it with:
  ./benchmarks/dreamgenbench/launch_gr1_dreamgen_generation_kjob.sh
EOF
  exit 2
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    echo "Pass overrides as KEY=VALUE, for example: DATA_LIMIT=1 GPU_IDS='0 1 2 3 4 5 6 7'" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
USE_EMA="${USE_EMA:-1}"
PHYSICS_LATENT_MODEL_PATH="${PHYSICS_LATENT_MODEL_PATH:-}"
PHYSLATENT_UNCOND_MODE="${PHYSLATENT_UNCOND_MODE:-shared}"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
DATA_PATH="${DATA_PATH:-${EVAL_ROOT}/giga_input/gr1_dreamgen_it2v.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
RUN_NAME="${RUN_NAME:-gr1_dreamgen_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
MODEL_DIR="${MODEL_DIR:-${SAVE_DIR}/model}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/run.log}"
ENV_FILE="${ENV_FILE:-${SAVE_DIR}/run.env}"
SUMMARY_PATH="${SUMMARY_PATH:-${SAVE_DIR}/generation_summary.json}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${SAVE_DIR}/gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${SAVE_DIR}/gpu_memory_peak.json}"
GPU_MONITOR_PID=""

GPU_IDS_RAW="${GPU_IDS:-0 1 2 3 4 5 6 7}"
DATA_LIMIT="${DATA_LIMIT:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
NUM_FRAMES="${NUM_FRAMES:-93}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
SEED="${SEED:-6666}"

if [[ "${USE_EMA}" == "1" ]]; then
  TRANSFORMER_SRC="${CHECKPOINT_DIR}/transformer_ema"
else
  TRANSFORMER_SRC="${CHECKPOINT_DIR}/transformer"
fi

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
  "${PYTHON_BIN}" "${REPO_DIR}/scripts/gpu_memory_monitor.py" \
    --samples-csv "${GPU_MEMORY_SAMPLES}" \
    --peak-json "${GPU_MEMORY_PEAK}" \
    --interval-sec "${GPU_MONITOR_INTERVAL}" \
    --label "gr1_dreamgen_generation_${RUN_NAME}" &
  GPU_MONITOR_PID="$!"
  echo "GPU memory monitor pid: ${GPU_MONITOR_PID}"
  echo "GPU memory samples:     ${GPU_MEMORY_SAMPLES}"
  echo "GPU memory peak:        ${GPU_MEMORY_PEAK}"
}

require_dir "${REPO_DIR}"
require_dir "${CHECKPOINT_DIR}"
require_dir "${TRANSFORMER_SRC}"
require_dir "${PRETRAIN_DIR}/text_encoder"
require_dir "${PRETRAIN_DIR}/vae"
require_any_file "transformer weights" \
  "${TRANSFORMER_SRC}/diffusion_pytorch_model.safetensors" \
  "${TRANSFORMER_SRC}/diffusion_pytorch_model.bin"
require_file "${PRETRAIN_DIR}/text_encoder/config.json"
require_file "${PRETRAIN_DIR}/text_encoder/pytorch_model.bin"
require_file "${PRETRAIN_DIR}/vae/config.json"
require_any_file "VAE weights" \
  "${PRETRAIN_DIR}/vae/diffusion_pytorch_model.safetensors" \
  "${PRETRAIN_DIR}/vae/diffusion_pytorch_model.bin"
if [[ -n "${PHYSICS_LATENT_MODEL_PATH}" ]]; then
  if [[ -f "${PHYSICS_LATENT_MODEL_PATH}" ]]; then
    :
  elif [[ -f "${PHYSICS_LATENT_MODEL_PATH}/diffusion_pytorch_model.bin" ]]; then
    :
  elif [[ -f "${PHYSICS_LATENT_MODEL_PATH}/physics_latent_encoder/diffusion_pytorch_model.bin" ]]; then
    :
  else
    echo "Missing physics latent weights. Checked:" >&2
    echo "  ${PHYSICS_LATENT_MODEL_PATH}" >&2
    echo "  ${PHYSICS_LATENT_MODEL_PATH}/diffusion_pytorch_model.bin" >&2
    echo "  ${PHYSICS_LATENT_MODEL_PATH}/physics_latent_encoder/diffusion_pytorch_model.bin" >&2
    exit 1
  fi
fi

mkdir -p "${SAVE_DIR}" "${MODEL_DIR}" "${CUDA_CACHE_PATH}" "$(dirname "${LOG_FILE}")"
ln -sfn "${TRANSFORMER_SRC}" "${MODEL_DIR}/transformer"
ln -sfn "${PRETRAIN_DIR}/text_encoder" "${MODEL_DIR}/text_encoder"
ln -sfn "${PRETRAIN_DIR}/vae" "${MODEL_DIR}/vae"

touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Persistent log: ${LOG_FILE}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing ${DATA_PATH}; preparing GR1 DreamGen inputs first."
  "${PYTHON_BIN}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_gr1_dreamgen_inputs.py" \
    --output-root "${EVAL_ROOT}"
fi
require_file "${DATA_PATH}"

EFFECTIVE_DATA_PATH="${DATA_PATH}"
if [[ "${DATA_LIMIT}" != "0" ]]; then
  EFFECTIVE_DATA_PATH="${SAVE_DIR}/input_first_${DATA_LIMIT}.json"
  "${PYTHON_BIN}" -c 'import json, os, sys; src, dst, limit_s = sys.argv[1:4]; limit = int(limit_s); data = json.load(open(src, "r"))[:limit]; base = os.path.dirname(os.path.abspath(src)); [item.__setitem__("image", os.path.abspath(os.path.join(base, item["image"]))) for item in data if item.get("image") and not os.path.isabs(item["image"])]; json.dump(data, open(dst, "w"), ensure_ascii=False, indent=2)' \
    "${DATA_PATH}" "${EFFECTIVE_DATA_PATH}" "${DATA_LIMIT}"
fi

cat > "${ENV_FILE}" <<EOF
REPO_DIR=${REPO_DIR}
GIGA_MODELS_DIR=${GIGA_MODELS_DIR}
CONDA_ENV=${CONDA_ENV}
PYTHON_BIN=${PYTHON_BIN}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
PRETRAIN_DIR=${PRETRAIN_DIR}
USE_EMA=${USE_EMA}
TRANSFORMER_SRC=${TRANSFORMER_SRC}
PHYSICS_LATENT_MODEL_PATH=${PHYSICS_LATENT_MODEL_PATH}
PHYSLATENT_UNCOND_MODE=${PHYSLATENT_UNCOND_MODE}
MODEL_DIR=${MODEL_DIR}
DATA_PATH=${DATA_PATH}
EFFECTIVE_DATA_PATH=${EFFECTIVE_DATA_PATH}
SAVE_DIR=${SAVE_DIR}
LOG_FILE=${LOG_FILE}
ENV_FILE=${ENV_FILE}
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
echo "GigaWorld-0 GR1 DreamGen generation"
echo "Host:                    $(hostname)"
echo "Repo:                    ${REPO_DIR}"
echo "Checkpoint dir:          ${CHECKPOINT_DIR}"
echo "Transformer src:         ${TRANSFORMER_SRC}"
echo "Physics latent path:     ${PHYSICS_LATENT_MODEL_PATH:-<disabled>}"
echo "PhysLatent CFG mode:     ${PHYSLATENT_UNCOND_MODE}"
echo "Model dir:               ${MODEL_DIR}"
echo "Data path:               ${EFFECTIVE_DATA_PATH}"
echo "Save dir:                ${SAVE_DIR}"
echo "Summary:                 ${SUMMARY_PATH}"
echo "GPU memory samples:      ${GPU_MEMORY_SAMPLES}"
echo "GPU memory peak:         ${GPU_MEMORY_PEAK}"
echo "Requested GPU ids:       ${REQUESTED_GPU_IDS_ARGS[*]}"
echo "CUDA_VISIBLE_DEVICES:    ${CUDA_VISIBLE_DEVICES}"
echo "Python GPU ids:          ${PYTHON_GPU_IDS_ARGS[*]}"
echo "Steps:                   ${NUM_INFERENCE_STEPS}"
echo "Frames/FPS/Size:         ${NUM_FRAMES}/${FPS}/${HEIGHT}x${WIDTH}"
echo "Seed:                    ${SEED}"
echo "============================================"

log_mem "job start"
nvidia-smi || true
start_gpu_monitor

"${PYTHON_BIN}" scripts/check_cuda.py
log_mem "after cuda check"

status=0
EXTRA_INFERENCE_ARGS=()
if [[ -n "${PHYSICS_LATENT_MODEL_PATH}" ]]; then
  EXTRA_INFERENCE_ARGS+=(
    --physics-latent-model-path "${PHYSICS_LATENT_MODEL_PATH}"
    --physlatent-uncond-mode "${PHYSLATENT_UNCOND_MODE}"
  )
fi

"${PYTHON_BIN}" scripts/inference.py \
  --data-path "${EFFECTIVE_DATA_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --transformer-model-path "${MODEL_DIR}/transformer" \
  --text-encoder-model-path "${MODEL_DIR}/text_encoder" \
  --vae-model-path "${MODEL_DIR}/vae" \
  "${EXTRA_INFERENCE_ARGS[@]}" \
  --gpu-ids "${PYTHON_GPU_IDS_ARGS[@]}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --fps "${FPS}" \
  --num-frames "${NUM_FRAMES}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --seed "${SEED}" \
  --summary-path "${SUMMARY_PATH}" || status=$?

exit "${status}"
