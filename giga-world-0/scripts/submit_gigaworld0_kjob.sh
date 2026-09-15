#!/usr/bin/env bash
set -euo pipefail

# Submit the GigaWorld-0 inference script through kjobctl. This machine does
# not expose sbatch directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
RL_DIR="${RL_DIR:-/home/jovyan/new_rl/rl}"
JOB_SCRIPT="${JOB_SCRIPT:-${REPO_DIR}/scripts/kjob_gigaworld0_video_pretrain_infer.sh}"

if [[ ! -d "${RL_DIR}" ]]; then
  echo "Missing RL_DIR: ${RL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Missing JOB_SCRIPT: ${JOB_SCRIPT}" >&2
  exit 1
fi

SCRIPT_ARGS=("REPO_DIR=${REPO_DIR}")

maybe_export() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    SCRIPT_ARGS+=("${name}=${!name}")
  fi
}

for name in \
  GIGA_MODELS_DIR \
  GAGI \
  WA2_ROOT \
  CONDA_SH \
  CONDA_ENV \
  MODEL_DIR \
  DATA_PATH \
  OUTPUT_ROOT \
  FINAL_ROOT \
  MODELS \
  CHAIN_NAME \
  RUN_NAME \
  SAVE_DIR \
  LOG_FILE \
  ENV_FILE \
  SUMMARY_PATH \
  SERVE_LOG \
  RESULTS_DIR \
  HOST \
  PORT \
  DEVICE \
  DREAMGEN_REPO \
  DREAMGEN_EVAL_PYTHON \
  VIDEO_DIR \
  LOG_DIR \
  RUN_LOG \
  GPU_IDS \
  ALLOW_MULTI_GPU_SMOKE \
  GPU_MONITOR_INTERVAL \
  GPU_MEMORY_SAMPLES \
  GPU_MEMORY_PEAK \
  LIMIT \
  DIMENSIONS \
  RESOLUTION_NAME \
  LOCAL_CKPT \
  REQUIRE_CHECKPOINTS \
  SKIP_EXISTING \
  SAVE_GENERATED_ONLY \
  AUDIT_WORKERS \
  POSTPROCESS_WORKERS \
  EXPECTED_COUNT \
  NPROC_PER_NODE \
  WORLDARENA_ROOT \
  MANIFEST \
  VIDEO_ROOT \
  EVAL_ROOT \
  FRAME_ROOT \
  CORE_SUBDIR \
  CONFIG \
  METRICS \
  NUM_INFERENCE_STEPS \
  DATA_LIMIT \
  FPS \
  NUM_FRAMES \
  HEIGHT \
  WIDTH \
  SEED \
  HF_HOME \
  HF_HUB_OFFLINE \
  TRANSFORMERS_OFFLINE \
  DIFFUSERS_OFFLINE \
  CUDA_VISIBLE_DEVICES \
  NCCL_SOCKET_IFNAME \
  GLOO_SOCKET_IFNAME \
  NCCL_DEBUG; do
  maybe_export "${name}"
done

echo "Submitting GigaWorld-0 kjob"
echo "Repo dir:   ${REPO_DIR}"
echo "RL dir:     ${RL_DIR}"
echo "Job script: ${JOB_SCRIPT}"
if [[ -n "${RUN_NAME:-}" ]]; then
  EXPECTED_OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs}"
  EXPECTED_SAVE_DIR="${SAVE_DIR:-${EXPECTED_OUTPUT_ROOT}/${RUN_NAME}}"
  EXPECTED_LOG_FILE="${LOG_FILE:-${EXPECTED_SAVE_DIR}/run.log}"
  echo "Run name:   ${RUN_NAME}"
  echo "Save dir:   ${EXPECTED_SAVE_DIR}"
  echo "Run log:    ${EXPECTED_LOG_FILE}"
fi

if command -v kjobctl >/dev/null 2>&1; then
  KJOB_CMD=(kjobctl)
elif command -v pixi >/dev/null 2>&1; then
  cd "${RL_DIR}"
  KJOB_CMD=(pixi run kjobctl)
else
  echo "Neither kjobctl nor pixi is available." >&2
  exit 1
fi

exec "${KJOB_CMD[@]}" create slurm \
  --pod-template-label vllm-metrics=true \
  -- \
  --chdir "${REPO_DIR}" \
  "${JOB_SCRIPT}" \
  "${SCRIPT_ARGS[@]}" \
  "$@"
