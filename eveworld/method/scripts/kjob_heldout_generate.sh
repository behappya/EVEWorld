#!/usr/bin/env bash
#SBATCH --job-name=eve_heldout_gen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

for arg in "$@"; do
  [[ "${arg}" == *=* ]] || { echo "Bad argument: ${arg}"; exit 1; }
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
PYTHON="${TRAIN_VENV}/bin/python"
MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
PIPELINE="${PIPELINE:?PIPELINE is required}"
METHOD="${METHOD:?METHOD is required}"
TRAINING_SEED="${TRAINING_SEED:?TRAINING_SEED is required}"
GENERATION_SEEDS="${GENERATION_SEEDS:-6666 1234}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?SPLIT_MANIFEST is required}"
EXPECTED_SPLIT="${EXPECTED_SPLIT:-test}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
LORA="${LORA:-NONE}"

source "${TRAIN_VENV}/bin/activate"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
cd "${REPO_DIR}"

read -r -a SEED_ARRAY <<< "${GENERATION_SEEDS}"
SKIP_ARGS=()
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  SKIP_ARGS=(--skip-existing)
fi

echo "[heldout-gen] pipeline=${PIPELINE} method=${METHOD} train_seed=${TRAINING_SEED}"
echo "[heldout-gen] one node, 8 GPUs, generation_seeds=${GENERATION_SEEDS}"
nvidia-smi
"${PYTHON}" eveworld/method/scripts/heldout_generate_dispatch.py \
  --pipeline "${PIPELINE}" \
  --method "${METHOD}" \
  --training-seed "${TRAINING_SEED}" \
  --generation-seeds "${SEED_ARRAY[@]}" \
  --gpu-count 8 \
  --data-path "${DATA_PATH}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --expected-split "${EXPECTED_SPLIT}" \
  --out-root "${OUT_ROOT}" \
  --transformer "${MODEL_DIR}/transformer" \
  --text-encoder "${MODEL_DIR}/text_encoder" \
  --vae "${MODEL_DIR}/vae" \
  --lora "${LORA}" \
  --joint-script "${EVEWORLD_ROOT}/eveworld/method/scripts/generate_joint.py" \
  --frontier-script "${EVEWORLD_ROOT}/eveworld/method/scripts/generate_frontier.py" \
  --python "${PYTHON}" \
  --limit "${LIMIT:-0}" \
  --steps "${STEPS:-30}" \
  --num-frames "${NUM_FRAMES:-93}" \
  --height "${HEIGHT:-480}" \
  --width "${WIDTH:-768}" \
  --fps "${FPS:-16}" \
  --block-size "${BLOCK_SIZE:-4}" \
  "${SKIP_ARGS[@]}"
