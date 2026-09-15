#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062"
PROBE_ROOT="${ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062_probes"
MANIFEST_ROOT="${ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062_eval"
PAYLOAD="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_reference_seed049_062_eval175_kjob.sh"
PRETRAIN_WEIGHT_DIR="${GAGI}/giga_world_0_video_pretrain/transformer"
SFT_WEIGHT_DIR="${GAGI}/eve_v2_outputs/round0_fullft/experiments_round0/models/checkpoint_epoch_150_step_150/transformer"

run_check() {
  for weight_dir in "${PRETRAIN_WEIGHT_DIR}" "${SFT_WEIGHT_DIR}"; do
    [[ -s "${weight_dir}/config.json" ]] || { echo "Missing config: ${weight_dir}" >&2; return 1; }
    compgen -G "${weight_dir}/diffusion_pytorch_model.*" >/dev/null || return 1
  done
  bash -n "${PAYLOAD}"
  bash -n "$0"
  PYTHONWARNINGS=ignore \
  PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}" \
    "${GAGI}/envs/giga_world_train_venv/bin/python" \
    eveworld/evaluation/eval175_multiseed_worker.py --help >/dev/null
  echo "Pretrain s000, SFT raw s150, and seed049/062 worker are ready."
}

submit_job() {
  local resume="$1" output rc job
  cd "${REPO_DIR}"
  set +e
  output="$(
    JOB_SCRIPT="${PAYLOAD}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_NAME="reference_eval175_pretrain_s000_sft_s150_seed049_seed062" \
      ./scripts/submit_gigaworld0_kjob.sh \
        "OUTPUT_ROOT=${OUTPUT_ROOT}" \
        "PROBE_ROOT=${PROBE_ROOT}" \
        "MANIFEST_ROOT=${MANIFEST_ROOT}" \
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
