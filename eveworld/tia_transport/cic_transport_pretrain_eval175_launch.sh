#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${ROOT}/eval175_seed004_pretrain_raw_s000"
PROBE_ROOT="${ROOT}/eval175_seed004_pretrain_raw_s000_probes"
MANIFEST_ROOT="${ROOT}/eval175_seed004_pretrain_raw_s000_eval"
PAYLOAD="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_pretrain_eval175_kjob.sh"
WEIGHT_DIR="${GAGI}/giga_world_0_video_pretrain/transformer"

run_check() {
  [[ -s "${WEIGHT_DIR}/config.json" ]] || { echo "Missing pretrained config" >&2; return 1; }
  compgen -G "${WEIGHT_DIR}/diffusion_pytorch_model.*" >/dev/null || {
    echo "Missing pretrained weights" >&2
    return 1
  }
  bash -n "${PAYLOAD}"
  bash -n "$0"
  PYTHONWARNINGS=ignore \
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${GAGI}/envs/giga_world_train_venv/bin/python" \
    eveworld/evaluation/eval175_multiseed_worker.py --help >/dev/null
  echo "Pretrained Transformer and standard EVAL-175 worker are ready."
}

run_submit() {
  run_check
  if [[ "${RESUME:-0}" != "1" ]]; then
    [[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing existing ${OUTPUT_ROOT}" >&2; exit 2; }
    [[ ! -e "${PROBE_ROOT}" ]] || { echo "Refusing existing ${PROBE_ROOT}" >&2; exit 2; }
    [[ ! -e "${MANIFEST_ROOT}" ]] || { echo "Refusing existing ${MANIFEST_ROOT}" >&2; exit 2; }
  fi
  mkdir -p "${OUTPUT_ROOT}/audit"
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/PREPARED"

  cd "${REPO_DIR}"
  set +e
  output="$(
    JOB_SCRIPT="${PAYLOAD}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_NAME="cic_transport_eval175_seed004_pretrain_raw_s000" \
      ./scripts/submit_gigaworld0_kjob.sh \
        "OUTPUT_ROOT=${OUTPUT_ROOT}" \
        "PROBE_ROOT=${PROBE_ROOT}" \
        "MANIFEST_ROOT=${MANIFEST_ROOT}" \
        "SEED=4" \
        "RESUME=${RESUME:-0}" 2>&1
  )"
  rc=$?
  set -e
  printf '%s\n' "${output}" | tee "${OUTPUT_ROOT}/audit/submit.log"
  [[ "${rc}" == "0" ]] || return "${rc}"
  job="$(printf '%s\n' "${output}" | grep -oE 'slurm-profile-slurm-[a-z0-9]+' | tail -1)"
  [[ -n "${job}" ]] || { echo "Could not parse job id" >&2; exit 1; }
  printf '%s\n' "${job}" | tee "${OUTPUT_ROOT}/audit/job_id.txt"
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  resume) RESUME=1 run_submit ;;
  *) echo "Usage: bash $(basename "$0") {check|submit|resume}" >&2; exit 2 ;;
esac
