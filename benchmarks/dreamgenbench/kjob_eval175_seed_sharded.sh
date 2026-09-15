#!/usr/bin/env bash
#SBATCH --job-name=eve_eval175_seed_sharded
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run EVAL-175 generation on the workspace host." >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] || { echo "Bad argument: ${arg}" >&2; exit 2; }
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
GAGI="${GAGI:-/data/datasets/gagi}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI}/envs/giga_world_train_venv}"
PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

MODEL="${MODEL:-round0}"
SEEDS="${SEEDS:-40 4}"
DATA_ROOT="${DATA_ROOT:-${GAGI}/gr1_dreamgen_eval/giga_input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/eval175_matched_seeds}"
CHAIN_NAME="${CHAIN_NAME:-round0_seed040_004}"

export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_NVML_BASED_CUDA_CHECK=1

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
read -r -a SEED_ARGS <<< "${SEEDS}"

LOG_ROOT="${OUTPUT_ROOT}/${MODEL}/logs/${CHAIN_NAME}"
mkdir -p "${LOG_ROOT}"
RUN_LOG="${LOG_ROOT}/kjob.log"

{
  echo "== EVAL-175 matched-seed sharded generation =="
  echo "host=$(hostname) model=${MODEL} seeds=${SEED_ARGS[*]}"
  echo "execution=seeds serial; each seed sharded over 8 GPUs"
  echo "output=${OUTPUT_ROOT}/${MODEL}"
  nvidia-smi
  "${PYTHON}" eveworld/evaluation/eval175_seed_sharded_dispatch.py \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --chain-name "${CHAIN_NAME}" \
    --seeds "${SEED_ARGS[@]}" \
    --gpu-count 8 \
    --python "${PYTHON}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS:-30}" \
    --num-frames "${NUM_FRAMES:-93}" \
    --fps "${FPS:-16}" \
    --height "${HEIGHT:-480}" \
    --width "${WIDTH:-768}" \
    --skip-existing
} 2>&1 | tee -a "${RUN_LOG}"
