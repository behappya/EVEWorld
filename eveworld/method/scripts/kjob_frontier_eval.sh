#!/usr/bin/env bash
# Single-node checkpoint preference evaluation. One GPU is sufficient because
# checkpoints are evaluated sequentially; no multi-node fan-out is used.
#SBATCH --job-name=eve_frontier_eval
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
DATASET="${DATASET:-/data/datasets/gagi/gr1_finetune_data/packed_data}"
VAE_PATH="${VAE_PATH:-/data/datasets/gagi/giga_world_0_video_pretrain/vae}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-/data/datasets/gagi/giga_world_0_video_pretrain/transformer}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
CHECKPOINT_CONTROL="${CHECKPOINT_CONTROL:?CHECKPOINT_CONTROL is required}"
CHECKPOINT_ANTI="${CHECKPOINT_ANTI:?CHECKPOINT_ANTI is required}"
INDICES="${INDICES:-91,36,57,82,79,23,5,16}"
SIGMA="${SIGMA:-0.5}"
SEED="${SEED:-20260715}"
GPU_ID="${GPU_ID:-0}"
NEGATIVE_MODE="${NEGATIVE_MODE:-adjacent}"

for checkpoint in "${CHECKPOINT_CONTROL}" "${CHECKPOINT_ANTI}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint must be a weight file: ${checkpoint}" >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"
cd "${REPO_DIR}"
source "${TRAIN_VENV}/bin/activate"

"${TRAIN_PYTHON}" eveworld/method/scripts/evaluate_frontier_preference.py \
  --dataset "${DATASET}" \
  --vae "${VAE_PATH}" \
  --transformer "${TRANSFORMER_PATH}" \
  --checkpoint "control=${CHECKPOINT_CONTROL}" \
  --checkpoint "anti=${CHECKPOINT_ANTI}" \
  --output "${OUTPUT_JSON}" \
  --indices "${INDICES}" \
  --sigma "${SIGMA}" \
  --seed "${SEED}" \
  --negative-mode "${NEGATIVE_MODE}"

test -s "${OUTPUT_JSON}"
echo "Frontier preference evaluation finished: ${OUTPUT_JSON}"
