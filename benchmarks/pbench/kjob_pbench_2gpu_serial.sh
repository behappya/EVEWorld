#!/usr/bin/env bash
#SBATCH --job-name=pbench_2gpu_serial
#SBATCH --gpus-per-task=nvidia.com/gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail

# Serial PBench Robot generation on 2 GPUs. Models run one after another
# inside a single kjob so the total GPU footprint stays at 2.
# Pass overrides as KEY=VALUE args, e.g.:
#   MODELS="t4g_wmapA_pre_seed42_s250 round0"

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
GAGI="${GAGI:-/data/datasets/gagi}"

MODELS="${MODELS:-t4g_wmapA_pre_seed42_s250 round0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/pbench_gen}"
DATA_PATH="${DATA_PATH:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
NUM_FRAMES="${NUM_FRAMES:-61}"
FPS="${FPS:-16}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"
SEED="${SEED:-6666}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.nv/ComputeCache}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

model_dir_for() {
  case "$1" in
    pretrain) echo "${GAGI}/giga_world_0_video_pretrain" ;;
    round0) echo "${GAGI}/eve_v2_outputs/anchor_models/round0_ema_st" ;;
    t4g_wmapA_pre_seed42_s250) echo "${GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s250" ;;
    t4g_wmapA_pre_noaug_s50) echo "${GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s50" ;;
    t4g_wmaponly_s150) echo "${GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmaponly_s150" ;;
    *) echo "" ;;
  esac
}

mkdir -p "${OUTPUT_ROOT}"
JOB_LOG="${OUTPUT_ROOT}/pbench_2gpu_serial_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"

for model in ${MODELS}; do
  MODEL_DIR="$(model_dir_for "${model}")"
  if [[ -z "${MODEL_DIR}" ]]; then
    echo "Unknown model name: ${model}" | tee -a "${JOB_LOG}" >&2
    exit 1
  fi
  if [[ ! -f "${MODEL_DIR}/transformer/config.json" ]]; then
    echo "Missing transformer config for ${model}: ${MODEL_DIR}" | tee -a "${JOB_LOG}" >&2
    exit 1
  fi
  SAVE_DIR="${OUTPUT_ROOT}/pbench_robot_3p8s_${model}"
  SUMMARY_PATH="${SAVE_DIR}/generation_summary.json"
  if [[ -f "${SUMMARY_PATH}" ]]; then
    echo "SKIP ${model}: summary already exists ${SUMMARY_PATH}" | tee -a "${JOB_LOG}"
    continue
  fi
  mkdir -p "${SAVE_DIR}"
  echo "===== $(date '+%F %T') START ${model} -> ${SAVE_DIR}" | tee -a "${JOB_LOG}"
  python scripts/inference.py \
    --data-path "${DATA_PATH}" \
    --save-dir "${SAVE_DIR}" \
    --transformer-model-path "${MODEL_DIR}/transformer" \
    --text-encoder-model-path "${MODEL_DIR}/text_encoder" \
    --vae-model-path "${MODEL_DIR}/vae" \
    --gpu-ids 0 1 \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --fps "${FPS}" \
    --num-frames "${NUM_FRAMES}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --seed "${SEED}" \
    --summary-path "${SUMMARY_PATH}" >>"${SAVE_DIR}/run.log" 2>&1
  echo "===== $(date '+%F %T') DONE ${model}: mp4=$(find "${SAVE_DIR}" -maxdepth 1 -name 'robot_*.mp4' | wc -l)/174" | tee -a "${JOB_LOG}"
done

echo "===== $(date '+%F %T') ALL DONE: ${MODELS}" | tee -a "${JOB_LOG}"
