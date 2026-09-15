#!/usr/bin/env bash
#SBATCH --job-name=eve_frontier_gen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Bad argument: ${arg}"
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
MODEL_DIR="${MODEL_DIR:-${GAGI}/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${EVEWORLD_ROOT}/eveworld/data_curation/splits/frontier_20260715/train.jsonl}"
EXPECTED_SPLIT="${EXPECTED_SPLIT:-train}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
TASK_SPECS="${TASK_SPECS:?TASK_SPECS is required}"
LIMIT="${LIMIT:-8}"
STEPS="${STEPS:-30}"
NUM_FRAMES="${NUM_FRAMES:-93}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
FPS="${FPS:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
BOUNDARY_GUARD_VELOCITY_SCALE="${BOUNDARY_GUARD_VELOCITY_SCALE:-0.5}"
BOUNDARY_GUARD_DECAY="${BOUNDARY_GUARD_DECAY:-1.0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

source "${TRAIN_VENV}/bin/activate"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
cd "${REPO_DIR}"

read -r -a TASK_ARRAY <<< "${TASK_SPECS}"
SKIP_ARGS=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  SKIP_ARGS=(--skip-existing)
fi

nvidia-smi
"${TRAIN_PYTHON}" eveworld/method/scripts/frontier_generate_dispatch.py \
  --tasks "${TASK_ARRAY[@]}" \
  --data-path "${DATA_PATH}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-split "${EXPECTED_SPLIT}" \
  --out-root "${OUT_ROOT}" \
  --transformer "${MODEL_DIR}/transformer" \
  --text-encoder "${MODEL_DIR}/text_encoder" \
  --vae "${MODEL_DIR}/vae" \
  --gen-script "${EVEWORLD_ROOT}/eveworld/method/scripts/generate_frontier.py" \
  --python "${TRAIN_PYTHON}" \
  --limit "${LIMIT}" \
  --steps "${STEPS}" \
  --num-frames "${NUM_FRAMES}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --block-size "${BLOCK_SIZE}" \
  --boundary-guard-velocity-scale "${BOUNDARY_GUARD_VELOCITY_SCALE}" \
  --boundary-guard-decay "${BOUNDARY_GUARD_DECAY}" \
  "${SKIP_ARGS[@]}"
