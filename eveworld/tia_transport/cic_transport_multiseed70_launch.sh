#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${ROOT}/eval175_transport_raw_s150_s200_multiseed70"
PROBE_ROOT="${ROOT}/eval175_transport_raw_s150_s200_multiseed70_probes"
MANIFEST_ROOT="${ROOT}/eval175_transport_raw_s150_s200_multiseed70_eval"
PROJECT="${ROOT}/cic_transport_seed42_s300/experiments"
PAYLOAD="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_multiseed70_kjob.sh"
DISPATCHER="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_multiseed70_dispatch.py"
WORKER="${EVEWORLD_ROOT}/eveworld/tia_transport/eval175_transport_worker.py"
PYTHON="${GAGI}/envs/giga_world_train_venv/bin/python"

checkpoint_for_step() {
  local step="$1" checkpoint
  checkpoint="$(find "${PROJECT}/models" -maxdepth 1 -type d -name "checkpoint_epoch_*_step_${step}" -print -quit)"
  [[ -n "${checkpoint}" ]] || { echo "Missing step${step} checkpoint" >&2; return 1; }
  printf '%s\n' "${checkpoint}"
}

check_checkpoint() {
  local step="$1" checkpoint transformer
  checkpoint="$(checkpoint_for_step "${step}")"
  transformer="${checkpoint}/transformer"
  [[ -s "${transformer}/config.json" ]] || return 1
  [[ -s "${transformer}/cic_transport_config.json" ]] || return 1
  compgen -G "${transformer}/diffusion_pytorch_model.*" >/dev/null || return 1
}

run_check() {
  check_checkpoint 150
  check_checkpoint 200
  bash -n "${PAYLOAD}"
  bash -n "$0"
  "${PYTHON}" -m py_compile "${DISPATCHER}" "${WORKER}"
  echo "Transport raw s150/s200 and isolated multi-seed entrypoints are ready."
}

prepare_isolated_roots() {
  [[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing existing ${OUTPUT_ROOT}" >&2; exit 2; }
  [[ ! -e "${PROBE_ROOT}" ]] || { echo "Refusing existing ${PROBE_ROOT}" >&2; exit 2; }
  [[ ! -e "${MANIFEST_ROOT}" ]] || { echo "Refusing existing ${MANIFEST_ROOT}" >&2; exit 2; }

  mkdir -p "${OUTPUT_ROOT}/audit" "${PROBE_ROOT}" "${MANIFEST_ROOT}"
  for step in 150 200; do
    model="transport_raw_s${step}"
    checkpoint="$(checkpoint_for_step "${step}")"
    probe="${PROBE_ROOT}/${model}"
    mkdir -p "${probe}"
    ln -s "${checkpoint}/transformer" "${probe}/transformer"
    ln -s "${GAGI}/giga_world_0_video_pretrain/text_encoder" "${probe}/text_encoder"
    ln -s "${GAGI}/giga_world_0_video_pretrain/vae" "${probe}/vae"
  done
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/PREPARED"
}

dry_run_range() {
  local model="$1" start="$2" end="$3" chain="$4" probe
  probe="${PROBE_ROOT}/${model}"
  CIC_TRANSPORT_MODEL_NAME="${model}" CIC_TRANSPORT_MODEL_DIR="${probe}" \
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${PYTHON}" "${DISPATCHER}" \
      --model "${model}" \
      --data-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
      --output-root "${OUTPUT_ROOT}" \
      --chain-name "${chain}" \
      --seeds $(seq "${start}" "${end}") \
      --gpu-count 8 \
      --python "${PYTHON}" \
      --worker-script "${WORKER}" \
      --dry-run >/dev/null
}

submit_range() {
  local start="$1" end="$2" chain="$3" resume="$4" output rc job
  cd "${REPO_DIR}"
  set +e
  output="$(
    JOB_SCRIPT="${PAYLOAD}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_NAME="cic_transport_ms70_${chain}_s150_then_s200" \
      ./scripts/submit_gigaworld0_kjob.sh \
        "OUTPUT_ROOT=${OUTPUT_ROOT}" \
        "PROBE_ROOT=${PROBE_ROOT}" \
        "MANIFEST_ROOT=${MANIFEST_ROOT}" \
        "SEED_START=${start}" \
        "SEED_END=${end}" \
        "CHAIN_NAME=${chain}" \
        "RESUME=${resume}" 2>&1
  )"
  rc=$?
  set -e
  printf '%s\n' "${output}" | tee "${OUTPUT_ROOT}/audit/submit_${chain}.log"
  [[ "${rc}" == "0" ]] || return "${rc}"
  job="$(printf '%s\n' "${output}" | grep -oE 'slurm-profile-slurm-[a-z0-9]+' | tail -1)"
  [[ -n "${job}" ]] || { echo "Could not parse ${chain} job id" >&2; exit 1; }
  printf '%s\n' "${job}" | tee "${OUTPUT_ROOT}/audit/job_id_${chain}.txt"
}

run_submit() {
  run_check
  prepare_isolated_roots
  dry_run_range transport_raw_s150 1 35 seed001_035
  dry_run_range transport_raw_s200 36 70 seed036_070
  submit_range 1 35 seed001_035 0
  submit_range 36 70 seed036_070 0
}

run_resume() {
  local start="$1" end="$2" chain="$3"
  run_check
  [[ -s "${OUTPUT_ROOT}/audit/PREPARED" ]] || { echo "Missing PREPARED marker" >&2; exit 2; }
  submit_range "${start}" "${end}" "${chain}" 1
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  resume-001-035) run_resume 1 35 seed001_035 ;;
  resume-036-070) run_resume 36 70 seed036_070 ;;
  *) echo "Usage: bash $(basename "$0") {check|submit|resume-001-035|resume-036-070}" >&2; exit 2 ;;
esac
