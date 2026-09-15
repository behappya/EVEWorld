#!/usr/bin/env bash
# Submit one held-out main-table training run on one eight-GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
for arg in "$@"; do
  [[ "${arg}" == *=* ]] || { echo "Bad argument: ${arg}" >&2; exit 1; }
  export "${arg}"
done

METHOD="${METHOD:?METHOD must be joint_lora, frontier_only, or eve}"
TRAINING_SEED="${TRAINING_SEED:?TRAINING_SEED is required}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${EVEWORLD_ROOT}/eveworld/data_curation/splits/frontier_20260715/train.jsonl}"
EXPECTED_TRAIN_SPLIT="${EXPECTED_TRAIN_SPLIT:-train}"
MANIFEST_TOOL="${EVEWORLD_ROOT}/eveworld/data_curation/scripts/heldout_manifest_tool.py"
RUN_TAG="${RUN_TAG:-heldout_main_$(date +%Y%m%d)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/eve/heldout_main}"
PROJECT_DIR="${TRAIN_PROJECT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}/train/${METHOD}/seed_${TRAINING_SEED}}"

TRAIN_COUNT="$(python3 "${MANIFEST_TOOL}" count --manifest "${TRAIN_MANIFEST}" --expected-split "${EXPECTED_TRAIN_SPLIT}")"
TRAIN_INDICES="$(python3 "${MANIFEST_TOOL}" indices --manifest "${TRAIN_MANIFEST}" --expected-split "${EXPECTED_TRAIN_SPLIT}")"
STEPS_PER_EPOCH=$(( (TRAIN_COUNT + 7) / 8 ))
MAX_STEPS="${MAX_STEPS:-$((STEPS_PER_EPOCH * 50))}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-$(( (MAX_STEPS + 2) / 3 ))}"
RUN_NAME="${RUN_NAME:-${RUN_TAG}_${METHOD}_trainseed${TRAINING_SEED}}"

case "${METHOD}" in
  joint_lora)
    BASE_CONFIG_MODULE="eveworld.method.configs.eve_joint_lora"
    FRONTIER_SKIP_WEIGHT=0.0
    FRONTIER_SKIP_DETACH_FUTURE=0
    ;;
  frontier_only)
    BASE_CONFIG_MODULE="eveworld.method.configs.eve_frontier_mvp"
    FRONTIER_SKIP_WEIGHT=0.0
    FRONTIER_SKIP_DETACH_FUTURE=0
    ;;
  eve)
    BASE_CONFIG_MODULE="eveworld.method.configs.eve_frontier_mvp"
    FRONTIER_SKIP_WEIGHT="${FRONTIER_SKIP_WEIGHT:-0.5}"
    FRONTIER_SKIP_DETACH_FUTURE=1
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

if [[ -d "${PROJECT_DIR}" ]] && find "${PROJECT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse non-empty project: ${PROJECT_DIR}" >&2
  exit 2
fi

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh}"
export TRAIN_PROJECT_DIR="${PROJECT_DIR}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-/data/datasets/gagi/gr1_finetune_data/packed_data}"
export MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export GPU_IDS="0 1 2 3 4 5 6 7"
export MAX_STEPS
export BATCH_SIZE_PER_GPU=1
export GRADIENT_ACCUMULATION_STEPS=1
export CHECKPOINT_INTERVAL
export CHECKPOINT_TOTAL_LIMIT=-1
export NUM_WORKERS="${NUM_WORKERS:-4}"
export NUM_FRAMES="${NUM_FRAMES:-93}"
export HEIGHT="${HEIGHT:-480}"
export WIDTH="${WIDTH:-768}"
export FPS="${FPS:-16}"
export WITH_EMA=0
export SKIP_TRAIN_ENV_SETUP="${SKIP_TRAIN_ENV_SETUP:-1}"
export RUN_NAME

PASS_ARGS=(
  "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}"
  "RUN_NAME=${RUN_NAME}"
  "SEED=${TRAINING_SEED}"
  "HELDOUT_TRAIN_INDICES=${TRAIN_INDICES}"
  "FRONTIER_TRAIN_INDICES=${TRAIN_INDICES}"
  "FRONTIER_BLOCK_SIZE=${FRONTIER_BLOCK_SIZE:-4}"
  "FRONTIER_SKIP_WEIGHT=${FRONTIER_SKIP_WEIGHT}"
  "FRONTIER_SKIP_MARGIN=${FRONTIER_SKIP_MARGIN:-0.05}"
  "FRONTIER_SKIP_FUTURE_OFFSET=${FRONTIER_SKIP_FUTURE_OFFSET:-1}"
  "FRONTIER_SKIP_NEGATIVE_MODE=${FRONTIER_SKIP_NEGATIVE_MODE:-adjacent}"
  "FRONTIER_SKIP_DETACH_FUTURE=${FRONTIER_SKIP_DETACH_FUTURE}"
  "FRONTIER_SKIP_DIAGNOSTIC=0"
  "FRONTIER_LORA_RANK=${FRONTIER_LORA_RANK:-64}"
  "FRONTIER_LR=${FRONTIER_LR:-4.315837287515549e-05}"
  "FRONTIER_MAX_STEPS=${MAX_STEPS}"
  "FRONTIER_CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
  "FRONTIER_MAX_GRAD_NORM=${FRONTIER_MAX_GRAD_NORM:-1.0}"
  "CHECKPOINT_TOTAL_LIMIT=-1"
  "WITH_EMA=0"
  "ACTIVATION_CHECKPOINTING=1"
  "RESUME=0"
)

echo "[heldout-train] method=${METHOD} training_seed=${TRAINING_SEED}"
echo "[heldout-train] rows=${TRAIN_COUNT} steps=${MAX_STEPS} checkpoints=${CHECKPOINT_INTERVAL}"
echo "[heldout-train] one node, GPUs=0,1,2,3,4,5,6,7"
echo "[heldout-train] project=${PROJECT_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] ${PASS_ARGS[*]}"
  exit 0
fi

exec "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_train_kjob.sh" "${PASS_ARGS[@]}"
