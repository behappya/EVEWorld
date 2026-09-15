#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${ROOT}/eval175_seed004_raw_s150_s250"
PROBE_ROOT="${ROOT}/eval175_seed004_raw_s150_s250_probes"
MANIFEST_ROOT="${ROOT}/eval175_seed004_raw_s150_s250_eval"
PAYLOAD="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_eval250_kjob.sh"
CONTROL_PROJECT="${ROOT}/control_repro_seed42_s300/experiments"
TRANSPORT_PROJECT="${ROOT}/cic_transport_seed42_s300/experiments"

check_checkpoint() {
  local project="$1" step="$2" transport="$3"
  local checkpoint
  checkpoint="$(find "${project}/models" -maxdepth 1 -type d -name "checkpoint_epoch_*_step_${step}" -print -quit)"
  [[ -n "${checkpoint}" ]] || { echo "Missing step${step}: ${project}" >&2; return 1; }
  [[ -s "${checkpoint}/transformer/config.json" ]] || return 1
  compgen -G "${checkpoint}/transformer/diffusion_pytorch_model.*" >/dev/null || return 1
  if [[ "${transport}" == "1" ]]; then
    [[ -s "${checkpoint}/transformer/cic_transport_config.json" ]] || return 1
  fi
}

run_check() {
  for step in 150 200 250; do
    check_checkpoint "${CONTROL_PROJECT}" "${step}" 0
    check_checkpoint "${TRANSPORT_PROJECT}" "${step}" 1
  done
  bash -n "${PAYLOAD}"
  bash -n "$0"
  PYTHONWARNINGS=ignore \
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${GAGI}/envs/giga_world_train_venv/bin/python" \
    eveworld/tia_transport/eval175_transport_worker.py --help >/dev/null
  echo "Six raw checkpoints and both workers are ready."
}

submit_variant() {
  local variant="$1" resume="$2"
  local output rc job
  cd "${REPO_DIR}"
  set +e
  output="$(
    JOB_SCRIPT="${PAYLOAD}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_NAME="cic_transport_eval175_seed004_${variant}_raw_s150_s250" \
      ./scripts/submit_gigaworld0_kjob.sh \
        "OUTPUT_ROOT=${OUTPUT_ROOT}" \
        "PROBE_ROOT=${PROBE_ROOT}" \
        "MANIFEST_ROOT=${MANIFEST_ROOT}" \
        "SEED=4" \
        "VARIANT=${variant}" \
        "RESUME=${resume}" 2>&1
  )"
  rc=$?
  set -e
  printf '%s\n' "${output}" | tee "${OUTPUT_ROOT}/audit/submit_${variant}.log"
  [[ "${rc}" == "0" ]] || return "${rc}"
  job="$(printf '%s\n' "${output}" | grep -oE 'slurm-profile-slurm-[a-z0-9]+' | tail -1)"
  [[ -n "${job}" ]] || { echo "Could not parse ${variant} job id" >&2; exit 1; }
  printf '%s\n' "${job}" | tee "${OUTPUT_ROOT}/audit/job_id_${variant}.txt"
}

run_submit_pair() {
  run_check
  [[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing existing ${OUTPUT_ROOT}" >&2; exit 2; }
  [[ ! -e "${PROBE_ROOT}" ]] || { echo "Refusing existing ${PROBE_ROOT}" >&2; exit 2; }
  [[ ! -e "${MANIFEST_ROOT}" ]] || { echo "Refusing existing ${MANIFEST_ROOT}" >&2; exit 2; }
  mkdir -p "${OUTPUT_ROOT}/audit"
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/PREPARED"
  submit_variant control 0
  submit_variant transport 0
}

run_resume() {
  local variant="$1"
  run_check
  [[ -s "${OUTPUT_ROOT}/audit/PREPARED" ]] || { echo "Missing PREPARED marker" >&2; exit 2; }
  submit_variant "${variant}" 1
}

case "${1:-}" in
  check) run_check ;;
  submit-pair) run_submit_pair ;;
  resume-control) run_resume control ;;
  resume-transport) run_resume transport ;;
  *) echo "Usage: bash $(basename "$0") {check|submit-pair|resume-control|resume-transport}" >&2; exit 2 ;;
esac
