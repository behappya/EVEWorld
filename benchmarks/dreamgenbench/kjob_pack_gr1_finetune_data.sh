#!/usr/bin/env bash
#SBATCH --job-name=gw0_gr1_pack
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  cat >&2 <<'EOF'
Refusing to run GR1 pack_data payload on the workspace host.

This script loads the T5-11B text encoder and should run inside a GPU kjob.
Submit it with:
  ./benchmarks/dreamgenbench/launch_gr1_pack_data_kjob.sh
EOF
  exit 2
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    echo "Pass overrides as KEY=VALUE, for example: FORCE_REPACK=1 GPU_IDS=0" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"

DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_finetune_data}"
VIDEO_DIR="${VIDEO_DIR:-${DATA_ROOT}/raw_data}"
PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
TEXT_ENCODER_MODEL_PATH="${TEXT_ENCODER_MODEL_PATH:-/data/datasets/gagi/giga_world_0_video_pretrain/text_encoder}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune}"
RUN_NAME="${RUN_NAME:-pack_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
WORK_PACKED_DATA_DIR="${WORK_PACKED_DATA_DIR:-${DATA_ROOT}/packed_data.work_${RUN_NAME}}"
FORCE_REPACK="${FORCE_REPACK:-0}"
GPU_IDS_RAW="${GPU_IDS:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.nv/ComputeCache}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

IFS=' ,' read -r -a REQUESTED_GPU_IDS_ARGS <<< "${GPU_IDS_RAW}"
if [[ ${#REQUESTED_GPU_IDS_ARGS[@]} -eq 0 ]]; then
  echo "GPU_IDS resolved to an empty list." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${REQUESTED_GPU_IDS_ARGS[*]}")"
export CUDA_VISIBLE_DEVICES

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

require_dir "${REPO_DIR}"
require_dir "${VIDEO_DIR}"
require_dir "${TEXT_ENCODER_MODEL_PATH}"
require_file "${TEXT_ENCODER_MODEL_PATH}/config.json"
require_file "${TEXT_ENCODER_MODEL_PATH}/pytorch_model.bin"

MP4_COUNT="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' | wc -l)"
TXT_COUNT="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.txt' | wc -l)"
if [[ "${MP4_COUNT}" -eq 0 ]]; then
  echo "No mp4 files found in ${VIDEO_DIR}" >&2
  exit 1
fi
if [[ "${MP4_COUNT}" != "${TXT_COUNT}" ]]; then
  echo "Video/text count mismatch in ${VIDEO_DIR}: mp4=${MP4_COUNT}, txt=${TXT_COUNT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${HF_HOME}" "${HF_XET_CACHE}" "${CUDA_CACHE_PATH}"
touch "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Persistent log: ${RUN_LOG}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

cat > "${ENV_FILE}" <<EOF
REPO_DIR=${REPO_DIR}
GIGA_MODELS_DIR=${GIGA_MODELS_DIR}
CONDA_ENV=${CONDA_ENV}
DATA_ROOT=${DATA_ROOT}
VIDEO_DIR=${VIDEO_DIR}
PACKED_DATA_DIR=${PACKED_DATA_DIR}
WORK_PACKED_DATA_DIR=${WORK_PACKED_DATA_DIR}
TEXT_ENCODER_MODEL_PATH=${TEXT_ENCODER_MODEL_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_NAME=${RUN_NAME}
RUN_LOG=${RUN_LOG}
ENV_FILE=${ENV_FILE}
GPU_IDS=${GPU_IDS_RAW}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
MP4_COUNT=${MP4_COUNT}
TXT_COUNT=${TXT_COUNT}
HF_HOME=${HF_HOME}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
EOF

echo "============================================"
echo "GigaWorld-0 GR1 pack_data kjob"
echo "Host:                    $(hostname)"
echo "Repo:                    ${REPO_DIR}"
echo "Video dir:               ${VIDEO_DIR}"
echo "Packed data dir:         ${PACKED_DATA_DIR}"
echo "Work packed data dir:    ${WORK_PACKED_DATA_DIR}"
echo "Text encoder:            ${TEXT_ENCODER_MODEL_PATH}"
echo "Run log:                 ${RUN_LOG}"
echo "Env file:                ${ENV_FILE}"
echo "CUDA_VISIBLE_DEVICES:    ${CUDA_VISIBLE_DEVICES}"
echo "Samples:                 ${MP4_COUNT}"
echo "Force repack:            ${FORCE_REPACK}"
echo "============================================"

log_mem "job start"
nvidia-smi || true

python scripts/check_cuda.py
log_mem "after cuda check"

if [[ -d "${PACKED_DATA_DIR}" ]]; then
  if [[ "${FORCE_REPACK}" != "1" ]]; then
    echo "Packed data dir already exists: ${PACKED_DATA_DIR}"
    echo "Set FORCE_REPACK=1 to replace it, or set PACKED_DATA_DIR to a new path."
    exit 0
  fi
  BACKUP_DIR="${PACKED_DATA_DIR}.backup_${RUN_NAME}"
  echo "Moving existing packed data to backup: ${BACKUP_DIR}"
  mv "${PACKED_DATA_DIR}" "${BACKUP_DIR}"
fi

if [[ -e "${WORK_PACKED_DATA_DIR}" ]]; then
  echo "Removing stale work dir: ${WORK_PACKED_DATA_DIR}"
  rm -rf "${WORK_PACKED_DATA_DIR}"
fi

echo "Starting pack_data..."
python scripts/pack_data.py \
  --video-dir "${VIDEO_DIR}" \
  --save-dir "${WORK_PACKED_DATA_DIR}" \
  --text-encoder-model-path "${TEXT_ENCODER_MODEL_PATH}"

echo "pack_data finished. Moving work dir into final path..."
mv "${WORK_PACKED_DATA_DIR}" "${PACKED_DATA_DIR}"

echo "Packed data ready: ${PACKED_DATA_DIR}"
du -sh "${PACKED_DATA_DIR}" 2>/dev/null || true
log_mem "job end"
