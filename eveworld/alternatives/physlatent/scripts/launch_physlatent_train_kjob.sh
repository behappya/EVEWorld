#!/usr/bin/env bash
set -euo pipefail

# Submit PhysLatent-GigaWorld training through the existing GigaWorld kjob
# payload. The payload already handles runtime config materialization,
# GPU memory monitoring, and completion checks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR="${REPO_DIR}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh}"
export TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
export DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_finetune_data}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
export MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/physlatent_gigaworld}"
export BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.alternatives.physlatent.configs.gr1_physlatent_adapter}"
if [[ -z "${TRAIN_PROJECT_DIR:-}" ]]; then
  case "${BASE_CONFIG_MODULE}" in
    *phase_done_lora*) export TRAIN_PROJECT_DIR="${OUTPUT_ROOT}/experiments_phase_done_lora" ;;
    *phase_done_aux*) export TRAIN_PROJECT_DIR="${OUTPUT_ROOT}/experiments_phase_done_aux_warmup" ;;
    *aux_lora*) export TRAIN_PROJECT_DIR="${OUTPUT_ROOT}/experiments_aux_lora" ;;
    *aux*) export TRAIN_PROJECT_DIR="${OUTPUT_ROOT}/experiments_aux_warmup" ;;
    *lora*) export TRAIN_PROJECT_DIR="${OUTPUT_ROOT}/experiments_query_adapter_lora" ;;
    *) export TRAIN_PROJECT_DIR="${OUTPUT_ROOT}/experiments_query_adapter_warmup" ;;
  esac
else
  export TRAIN_PROJECT_DIR
fi
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_physlatent_adapter}"
export LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
export RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
export ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
export RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-${OUTPUT_ROOT}/runtime_configs}"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-${RUNTIME_CONFIG_DIR}/${RUN_NAME}.json}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
export GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${LOG_DIR}/${RUN_NAME}_gpu_memory_samples.csv}"
export GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${LOG_DIR}/${RUN_NAME}_gpu_memory_peak.json}"

export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
if [[ -z "${MAX_STEPS:-}" ]]; then
  case "${BASE_CONFIG_MODULE}" in
    *phase_done_lora*) export MAX_STEPS=800 ;;
    *phase_done_aux*) export MAX_STEPS=500 ;;
    *aux*) export MAX_STEPS=500 ;;
    *) export MAX_STEPS=200 ;;
  esac
else
  export MAX_STEPS
fi
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
if [[ -z "${CHECKPOINT_INTERVAL:-}" ]]; then
  case "${BASE_CONFIG_MODULE}" in
    *phase_done*) export CHECKPOINT_INTERVAL=100 ;;
    *aux*) export CHECKPOINT_INTERVAL=100 ;;
    *) export CHECKPOINT_INTERVAL=50 ;;
  esac
else
  export CHECKPOINT_INTERVAL
fi
export CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-5}"
export NUM_WORKERS="${NUM_WORKERS:-6}"
export NUM_FRAMES="${NUM_FRAMES:-93}"
export HEIGHT="${HEIGHT:-480}"
export WIDTH="${WIDTH:-768}"
export FPS="${FPS:-16}"
export SEED="${SEED:-6666}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export WITH_EMA="${WITH_EMA:-0}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export PHYS_LABELS_PATH="${PHYS_LABELS_PATH:-}"
export PHYS_RANDOM_CROP="${PHYS_RANDOM_CROP:-0}"
export PHYS_LOSS_STATE="${PHYS_LOSS_STATE:-}"
export PHYS_LOSS_GOAL="${PHYS_LOSS_GOAL:-}"
export PHYS_LOSS_CONTACT="${PHYS_LOSS_CONTACT:-}"
export PHYS_LOSS_TRAJECTORY="${PHYS_LOSS_TRAJECTORY:-}"
export PHYS_LOSS_PHASE="${PHYS_LOSS_PHASE:-}"
export PHYS_LOSS_DONE="${PHYS_LOSS_DONE:-}"
export PHYS_LOSS_GOAL_REACHED="${PHYS_LOSS_GOAL_REACHED:-}"
export PHYS_LOSS_RELEASE="${PHYS_LOSS_RELEASE:-}"
export PHYS_LOSS_OBJECT_MOTION="${PHYS_LOSS_OBJECT_MOTION:-}"
export PHYS_LOSS_TERMINAL_STABLE="${PHYS_LOSS_TERMINAL_STABLE:-}"

if [[ "${SKIP_TRAIN_ENV_SETUP:-0}" != "1" ]]; then
  echo "Preparing PhysLatent training venv..."
  "${REPO_DIR}/eveworld/alternatives/physlatent/scripts/setup_physlatent_train_env.sh"
else
  echo "Skipping training venv setup because SKIP_TRAIN_ENV_SETUP=1"
fi

if [[ ! -f "${PACKED_DATA_DIR}/config.json" ]]; then
  echo "Missing packed data: ${PACKED_DATA_DIR}/config.json" >&2
  echo "Run first: ./benchmarks/dreamgenbench/launch_gr1_pack_data_kjob.sh" >&2
  exit 1
fi

echo
echo "Submitting PhysLatent-GigaWorld training kjob"
echo "Repo dir:        ${REPO_DIR}"
echo "Base config:     ${BASE_CONFIG_MODULE}"
echo "Train venv:      ${TRAIN_VENV}"
echo "Packed data:     ${PACKED_DATA_DIR}"
echo "Model dir:       ${MODEL_DIR}"
echo "Project dir:     ${TRAIN_PROJECT_DIR}"
echo "Run name:        ${RUN_NAME}"
echo "Run log:         ${RUN_LOG}"
echo "Runtime config:  ${RUNTIME_CONFIG}"
echo "GPU samples:     ${GPU_MEMORY_SAMPLES}"
echo "GPU peak:        ${GPU_MEMORY_PEAK}"
echo "Job script:      ${JOB_SCRIPT}"
echo "GPU ids:         ${GPU_IDS}"
echo "Steps:           ${MAX_STEPS}"
echo "Batch/GPU:       ${BATCH_SIZE_PER_GPU}"
echo "Grad accum:      ${GRADIENT_ACCUMULATION_STEPS}"
echo "Checkpoint int:  ${CHECKPOINT_INTERVAL}"
echo "With EMA:        ${WITH_EMA}"

SUBMIT_CMD=(
  "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh"
  "TRAIN_VENV=${TRAIN_VENV}"
  "DATA_ROOT=${DATA_ROOT}"
  "PACKED_DATA_DIR=${PACKED_DATA_DIR}"
  "MODEL_DIR=${MODEL_DIR}"
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
  "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "RUN_LOG=${RUN_LOG}"
  "ENV_FILE=${ENV_FILE}"
  "RUNTIME_CONFIG_DIR=${RUNTIME_CONFIG_DIR}"
  "RUNTIME_CONFIG=${RUNTIME_CONFIG}"
  "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}"
  "GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}"
  "GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}"
  "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}"
  "GPU_IDS=${GPU_IDS}"
  "MAX_STEPS=${MAX_STEPS}"
  "BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU}"
  "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
  "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
  "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}"
  "NUM_WORKERS=${NUM_WORKERS}"
  "NUM_FRAMES=${NUM_FRAMES}"
  "HEIGHT=${HEIGHT}"
  "WIDTH=${WIDTH}"
  "FPS=${FPS}"
  "SEED=${SEED}"
  "MIXED_PRECISION=${MIXED_PRECISION}"
  "WITH_EMA=${WITH_EMA}"
  "ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}"
)

if [[ -n "${PHYS_LABELS_PATH}" ]]; then
  SUBMIT_CMD+=("PHYS_LABELS_PATH=${PHYS_LABELS_PATH}" "PHYS_RANDOM_CROP=${PHYS_RANDOM_CROP}")
fi
if [[ -n "${PHYS_LOSS_STATE}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_STATE=${PHYS_LOSS_STATE}")
fi
if [[ -n "${PHYS_LOSS_GOAL}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_GOAL=${PHYS_LOSS_GOAL}")
fi
if [[ -n "${PHYS_LOSS_CONTACT}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_CONTACT=${PHYS_LOSS_CONTACT}")
fi
if [[ -n "${PHYS_LOSS_TRAJECTORY}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_TRAJECTORY=${PHYS_LOSS_TRAJECTORY}")
fi
if [[ -n "${PHYS_LOSS_PHASE}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_PHASE=${PHYS_LOSS_PHASE}")
fi
if [[ -n "${PHYS_LOSS_DONE}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_DONE=${PHYS_LOSS_DONE}")
fi
if [[ -n "${PHYS_LOSS_GOAL_REACHED}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_GOAL_REACHED=${PHYS_LOSS_GOAL_REACHED}")
fi
if [[ -n "${PHYS_LOSS_RELEASE}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_RELEASE=${PHYS_LOSS_RELEASE}")
fi
if [[ -n "${PHYS_LOSS_OBJECT_MOTION}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_OBJECT_MOTION=${PHYS_LOSS_OBJECT_MOTION}")
fi
if [[ -n "${PHYS_LOSS_TERMINAL_STABLE}" ]]; then
  SUBMIT_CMD+=("PHYS_LOSS_TERMINAL_STABLE=${PHYS_LOSS_TERMINAL_STABLE}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "DRY_RUN=1; not submitting. Command would be:"
  printf ' %q' "${SUBMIT_CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${SUBMIT_CMD[@]}" "$@"
