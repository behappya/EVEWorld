#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
TRAIN_PYTHON="/data/datasets/gagi/envs/giga_world_train_venv/bin/python"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
PRETRAIN="/data/datasets/gagi/giga_world_0_video_pretrain/transformer"
OUTPUT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1/cic_transport_seed6666_s400"
PAYLOAD="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_seed6666_s400_kjob.sh"
CONFIG="eveworld.tia_transport.cic_transport_seed6666_s400_config"
CAMPAIGN="eveworld.tia_transport.cic_transport_seed6666_s400_campaign"

run_python() {
  cd "${REPO_DIR}"
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${TRAIN_PYTHON}" "$@"
}

static_check() {
  run_python -m eveworld.tia_transport.test_cic_transport
  run_python -m py_compile \
    eveworld/tia_transport/cic_transport_seed6666_s400_config.py \
    eveworld/tia_transport/cic_transport_seed6666_s400_campaign.py
  bash -n "${PAYLOAD}"
  bash -n "$0"
}

submit_job() {
  local resume="$1" submit_output rc job
  cd "${REPO_DIR}"
  set +e
  submit_output="$(
    SKIP_TRAIN_ENV_SETUP=1 \
    JOB_SCRIPT="${PAYLOAD}" \
    PACKED_DATA_DIR="${PACKED}" \
    OUTPUT_ROOT="${OUTPUT}" \
    TRAIN_PROJECT_DIR="${OUTPUT}/experiments" \
    RUN_NAME="cic_transport_seed6666_s400" \
    MAX_STEPS=400 \
    CHECKPOINT_INTERVAL=50 \
    CHECKPOINT_TOTAL_LIMIT=8 \
    BATCH_SIZE_PER_GPU=1 \
    GRADIENT_ACCUMULATION_STEPS=8 \
    GPU_IDS="0 1 2 3 4 5 6 7" \
      ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
        "CIC_TRANSPORT_ALLOW_RESUME=${resume}" \
        "BASE_CONFIG_MODULE=${CONFIG}" \
        "TRANSFORMER_MODEL_PATH=${PRETRAIN}" \
        "PACKED_DATA_DIR=${PACKED}" \
        "OUTPUT_ROOT=${OUTPUT}" \
        "TRAIN_PROJECT_DIR=${OUTPUT}/experiments" \
        "RUN_NAME=cic_transport_seed6666_s400" \
        "MAX_STEPS=400" \
        "CHECKPOINT_INTERVAL=50" \
        "CHECKPOINT_TOTAL_LIMIT=8" \
        "BATCH_SIZE_PER_GPU=1" \
        "GRADIENT_ACCUMULATION_STEPS=8" \
        "NUM_WORKERS=6" \
        "WITH_EMA=1" \
        "MIXED_PRECISION=bf16" \
        "ACTIVATION_CHECKPOINTING=1" \
        "SEED=6666" 2>&1
  )"
  rc=$?
  set -e
  printf '%s\n' "${submit_output}" | tee "${OUTPUT}/audit/submit.log"
  [[ "${rc}" == "0" ]] || return "${rc}"
  job="$(printf '%s\n' "${submit_output}" | grep -oE 'slurm-profile-slurm-[a-z0-9]+' | tail -1)"
  [[ -n "${job}" ]] || { echo "Could not parse job id" >&2; return 1; }
  printf '%s\n' "${job}" | tee "${OUTPUT}/audit/job_id.txt"
}

run_check() {
  static_check
  run_python -m "${CAMPAIGN}" preflight
}

run_submit() {
  static_check
  run_python -m "${CAMPAIGN}" preflight --prepare >/dev/null
  submit_job 0
}

run_resume() {
  static_check
  [[ -s "${OUTPUT}/audit/PREPARED" ]] || { echo "Missing PREPARED marker" >&2; exit 2; }
  submit_job 1
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  resume) run_resume ;;
  audit) run_python -m "${CAMPAIGN}" audit ;;
  *) echo "Usage: bash $(basename "$0") {check|submit|resume|audit}" >&2; exit 2 ;;
esac
