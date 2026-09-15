#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250"
PROBE_ROOT="${ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250_probes"
MANIFEST_ROOT="${ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250_eval"
PROJECT="${ROOT}/cic_transport_seed6666_s400/experiments"
PAYLOAD="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_seed6666_eval250_kjob.sh"

check_checkpoint() {
  local step="$1" checkpoint
  checkpoint="$(find "${PROJECT}/models" -maxdepth 1 -type d -name "checkpoint_epoch_*_step_${step}" -print -quit)"
  [[ -n "${checkpoint}" ]] || { echo "Missing step${step}: ${PROJECT}" >&2; return 1; }
  [[ -s "${checkpoint}/transformer/config.json" ]] || return 1
  [[ -s "${checkpoint}/transformer/cic_transport_config.json" ]] || return 1
  compgen -G "${checkpoint}/transformer/diffusion_pytorch_model.*" >/dev/null || return 1
}

run_check() {
  local step
  for step in 150 200 250; do
    check_checkpoint "${step}"
  done
  bash -n "${PAYLOAD}"
  bash -n "$0"
  PYTHONWARNINGS=ignore \
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${GAGI}/envs/giga_world_train_venv/bin/python" \
    eveworld/tia_transport/eval175_transport_worker.py --help >/dev/null
  echo "Transport seed6666 raw s150/s200/s250 checkpoints and worker are ready."
}

submit_job() {
  local resume="$1" output rc job
  cd "${REPO_DIR}"
  set +e
  output="$(
    JOB_SCRIPT="${PAYLOAD}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_NAME="cic_transport_seed6666_eval175_seed004_raw_s150_s250" \
      ./scripts/submit_gigaworld0_kjob.sh \
        "OUTPUT_ROOT=${OUTPUT_ROOT}" \
        "PROBE_ROOT=${PROBE_ROOT}" \
        "MANIFEST_ROOT=${MANIFEST_ROOT}" \
        "SEED=4" \
        "RESUME=${resume}" 2>&1
  )"
  rc=$?
  set -e
  printf '%s\n' "${output}" | tee "${OUTPUT_ROOT}/audit/submit.log"
  [[ "${rc}" == "0" ]] || return "${rc}"
  job="$(printf '%s\n' "${output}" | grep -oE 'slurm-profile-slurm-[a-z0-9]+' | tail -1)"
  [[ -n "${job}" ]] || { echo "Could not parse job id" >&2; exit 1; }
  printf '%s\n' "${job}" | tee "${OUTPUT_ROOT}/audit/job_id.txt"
}

run_submit() {
  run_check
  [[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing existing ${OUTPUT_ROOT}" >&2; exit 2; }
  [[ ! -e "${PROBE_ROOT}" ]] || { echo "Refusing existing ${PROBE_ROOT}" >&2; exit 2; }
  [[ ! -e "${MANIFEST_ROOT}" ]] || { echo "Refusing existing ${MANIFEST_ROOT}" >&2; exit 2; }
  mkdir -p "${OUTPUT_ROOT}/audit"
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/PREPARED"
  submit_job 0
}

run_resume() {
  run_check
  [[ -s "${OUTPUT_ROOT}/audit/PREPARED" ]] || { echo "Missing PREPARED marker" >&2; exit 2; }
  submit_job 1
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  resume) run_resume ;;
  *) echo "Usage: bash $(basename "$0") {check|submit|resume}" >&2; exit 2 ;;
esac
