#!/usr/bin/env bash
# Paired 8-GPU launch entry for control reproduction and CIC-Transport.

set -euo pipefail

REPO_DIR="giga-world-0"
TRAIN_PYTHON="/data/datasets/gagi/envs/giga_world_train_venv/bin/python"
PACKED="/data/datasets/gagi/gr1_finetune_data/packed_data"
PRETRAIN="/data/datasets/gagi/giga_world_0_video_pretrain/transformer"
ROOT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1"
PAYLOAD="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_kjob_train.sh"
CAMPAIGN_MODULE="eveworld.tia_transport.cic_transport_campaign"

variant_output() {
  case "$1" in
    control) printf '%s\n' "${ROOT}/control_repro_seed42_s300" ;;
    transport) printf '%s\n' "${ROOT}/cic_transport_seed42_s300" ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

variant_config() {
  case "$1" in
    control) printf '%s\n' "eveworld.pipeline.t4g_joint_config" ;;
    transport) printf '%s\n' "eveworld.tia_transport.cic_transport_config" ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

run_python() {
  cd "${REPO_DIR}"
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${TRAIN_PYTHON}" "$@"
}

run_check() {
  run_python -m eveworld.tia_transport.test_cic_transport
  run_python -m "${CAMPAIGN_MODULE}" preflight --variant control
  run_python -m "${CAMPAIGN_MODULE}" preflight --variant transport
  run_python -m py_compile \
    eveworld/tia_transport/cic_transport_transformer.py \
    eveworld/tia_transport/cic_transport_trainer.py \
    eveworld/tia_transport/cic_transport_config.py \
    eveworld/tia_transport/cic_transport_pipeline.py \
    eveworld/tia_transport/cic_transport_campaign.py \
    eveworld/tia_transport/test_cic_transport.py
  bash -n eveworld/tia_transport/cic_transport_kjob_train.sh
  bash -n eveworld/tia_transport/cic_transport_launch.sh
}

submit_one() {
  local variant="$1"
  local output config run_name submit_output job
  output="$(variant_output "${variant}")"
  config="$(variant_config "${variant}")"
  run_name="cic_transport_${variant}_seed42_s300"

  run_python -m "${CAMPAIGN_MODULE}" preflight \
    --variant "${variant}" --prepare >/dev/null

  cd "${REPO_DIR}"
  set +e
  submit_output="$(
    SKIP_TRAIN_ENV_SETUP=1 \
    JOB_SCRIPT="${PAYLOAD}" \
    PACKED_DATA_DIR="${PACKED}" \
    OUTPUT_ROOT="${output}" \
    TRAIN_PROJECT_DIR="${output}/experiments" \
    RUN_NAME="${run_name}" \
    MAX_STEPS=300 \
    CHECKPOINT_INTERVAL=50 \
    CHECKPOINT_TOTAL_LIMIT=8 \
    BATCH_SIZE_PER_GPU=1 \
    GRADIENT_ACCUMULATION_STEPS=8 \
    GPU_IDS="0 1 2 3 4 5 6 7" \
      ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
        "CIC_TRANSPORT_VARIANT=${variant}" \
        "BASE_CONFIG_MODULE=${config}" \
        "TRANSFORMER_MODEL_PATH=${PRETRAIN}" \
        "PACKED_DATA_DIR=${PACKED}" \
        "OUTPUT_ROOT=${output}" \
        "TRAIN_PROJECT_DIR=${output}/experiments" \
        "RUN_NAME=${run_name}" \
        "MAX_STEPS=300" \
        "CHECKPOINT_INTERVAL=50" \
        "CHECKPOINT_TOTAL_LIMIT=8" \
        "BATCH_SIZE_PER_GPU=1" \
        "GRADIENT_ACCUMULATION_STEPS=8" \
        "NUM_WORKERS=6" \
        "WITH_EMA=1" \
        "MIXED_PRECISION=bf16" \
        "ACTIVATION_CHECKPOINTING=1" \
        "SEED=42" 2>&1
  )"
  local rc=$?
  set -e
  printf '%s\n' "${submit_output}" | tee "${output}/audit/submit.log"
  if [[ "${rc}" != "0" ]]; then
    echo "${variant} submission failed with rc=${rc}" >&2
    return "${rc}"
  fi
  job="$(printf '%s\n' "${submit_output}" | grep -oE 'slurm-profile-slurm-[a-z0-9]+' | tail -1)"
  if [[ -z "${job}" ]]; then
    echo "Could not parse ${variant} job id" >&2
    return 1
  fi
  printf '%s\n' "${job}" | tee "${output}/audit/job_id.txt"
}

run_audit() {
  run_python -m "${CAMPAIGN_MODULE}" audit --variant control
  run_python -m "${CAMPAIGN_MODULE}" audit --variant transport
}

case "${1:-}" in
  check) run_check ;;
  submit-control) submit_one control ;;
  submit-transport) submit_one transport ;;
  submit-pair)
    run_check
    submit_one control
    submit_one transport
    ;;
  audit) run_audit ;;
  *)
    echo "Usage: bash $(basename "$0") {check|submit-control|submit-transport|submit-pair|audit}" >&2
    exit 2
    ;;
esac
