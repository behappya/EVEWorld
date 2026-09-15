#!/usr/bin/env bash
# EVE-Frontier MVP: one-node, eight-GPU train-only overfit probe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

for _arg in "$@"; do
  [[ "${_arg}" == *=* ]] && export "${_arg}"
done

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh}"
export TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
export DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_finetune_data}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
export MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/eve}"

BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.method.configs.eve_frontier_mvp}"
SEED="${SEED:-20260715}"
FRONTIER_BLOCK_SIZE="${FRONTIER_BLOCK_SIZE:-4}"
# First eight rows of the frozen 20260715 train manifest; no val/test row is used.
FRONTIER_TRAIN_INDICES="${FRONTIER_TRAIN_INDICES:-91,36,57,82,79,23,5,16}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-frontier_mvp_overfit8_b${FRONTIER_BLOCK_SIZE}}"
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_${EXPERIMENT_TAG}_seed${SEED}}"

export TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_TAG}_seed${SEED}_${TIMESTAMP}}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export MAX_STEPS="${MAX_STEPS:-100}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
export CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:--1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export NUM_FRAMES="${NUM_FRAMES:-93}"
export HEIGHT="${HEIGHT:-480}"
export WIDTH="${WIDTH:-768}"
export FPS="${FPS:-16}"
export WITH_EMA=0
export SKIP_TRAIN_ENV_SETUP="${SKIP_TRAIN_ENV_SETUP:-1}"
export RUN_NAME

if [[ -d "${TRAIN_PROJECT_DIR}" ]] && find "${TRAIN_PROJECT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse non-empty TRAIN_PROJECT_DIR: ${TRAIN_PROJECT_DIR}" >&2
  exit 2
fi

PASS_ARGS=(
  "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}"
  "RUN_NAME=${RUN_NAME}"
  "SEED=${SEED}"
  "FRONTIER_BLOCK_SIZE=${FRONTIER_BLOCK_SIZE}"
  "FRONTIER_CONDITION_LATENTS=${FRONTIER_CONDITION_LATENTS:-1}"
  "FRONTIER_HISTORY_SIGMA=${FRONTIER_HISTORY_SIGMA:-0.0001}"
  "FRONTIER_SKIP_WEIGHT=${FRONTIER_SKIP_WEIGHT:-0.0}"
  "FRONTIER_SKIP_MARGIN=${FRONTIER_SKIP_MARGIN:-0.05}"
  "FRONTIER_SKIP_FUTURE_OFFSET=${FRONTIER_SKIP_FUTURE_OFFSET:-1}"
  "FRONTIER_SKIP_NEGATIVE_MODE=${FRONTIER_SKIP_NEGATIVE_MODE:-adjacent}"
  "FRONTIER_SKIP_DETACH_FUTURE=${FRONTIER_SKIP_DETACH_FUTURE:-0}"
  "FRONTIER_SKIP_DIAGNOSTIC=${FRONTIER_SKIP_DIAGNOSTIC:-0}"
  "FRONTIER_TRAIN_INDICES=${FRONTIER_TRAIN_INDICES}"
  "FRONTIER_LORA_RANK=${FRONTIER_LORA_RANK:-64}"
  "FRONTIER_LR=${FRONTIER_LR:-4.315837287515549e-05}"
  "FRONTIER_MAX_STEPS=${MAX_STEPS}"
  "FRONTIER_CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
  "FRONTIER_MAX_GRAD_NORM=${FRONTIER_MAX_GRAD_NORM:-1.0}"
  "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}"
  "WITH_EMA=0"
  "ACTIVATION_CHECKPOINTING=1"
  "RESUME=0"
)

echo "[Frontier] config=${BASE_CONFIG_MODULE} train_indices=${FRONTIER_TRAIN_INDICES}"
echo "[Frontier] one node / 8 GPUs, steps=${MAX_STEPS}, block=${FRONTIER_BLOCK_SIZE}, seed=${SEED}"
echo "[Frontier] project=${TRAIN_PROJECT_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] pass-args: ${PASS_ARGS[*]}"
  exit 0
fi

exec "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_train_kjob.sh" "${PASS_ARGS[@]}"
