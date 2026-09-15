#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TRAIN_JOB="${TRAIN_JOB:-slurm-profile-slurm-5rgvd}"
CAPACITY_JOBS="${CAPACITY_JOBS:-slurm-profile-slurm-xbtd7 slurm-profile-slurm-cghw2}"
NAMESPACE="${NAMESPACE:-user-behappya}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/datasets/gagi/eve_v2_outputs/eve_ablation_strict_v1}"
RAW_OUTPUT_ROOT="${RAW_OUTPUT_ROOT:-/data/datasets/gagi/eve_v2_outputs/eve_ablation_strict_v1_eval175}"
EMA_OUTPUT_ROOT="${EMA_OUTPUT_ROOT:-/data/datasets/gagi/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema}"
RAW_PROBE_ROOT="${RAW_PROBE_ROOT:-/data/datasets/gagi/eve_v2_outputs/anchor_models/eve_ablation_strict_v1}"
EMA_PROBE_ROOT="${EMA_PROBE_ROOT:-/data/datasets/gagi/eve_v2_outputs/anchor_models/eve_ablation_strict_v1_ema}"
STATE_ROOT="${STATE_ROOT:-${TRAIN_ROOT}/controller_logs}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
RAW_RECEIPT="${STATE_ROOT}/cic_seed004_raw_submission.txt"
EMA_RECEIPT="${STATE_ROOT}/cic_seed004_ema_submission.txt"
LOCK_DIR="${STATE_ROOT}/cic_seed004_dual_submission.lock"

mkdir -p "${STATE_ROOT}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[$(timestamp)] another CIC submission watcher owns ${LOCK_DIR}"
  exit 0
fi

cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

job_state() {
  local job="$1"
  local succeeded failed
  succeeded="$(kubectl get job -n "${NAMESPACE}" "${job}" \
    -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$(kubectl get job -n "${NAMESPACE}" "${job}" \
    -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  if [[ "${succeeded}" == "1" ]]; then
    printf '%s' succeeded
  elif [[ -n "${failed}" && "${failed}" != "0" ]]; then
    printf '%s' failed
  else
    printf '%s' running
  fi
}

while true; do
  train_state="$(job_state "${TRAIN_JOB}")"
  if [[ "${train_state}" == "failed" ]]; then
    echo "[$(timestamp)] training job ${TRAIN_JOB} failed; refusing submission" >&2
    exit 1
  fi

  capacity_ready=1
  capacity_status=()
  read -r -a capacity_jobs <<< "${CAPACITY_JOBS}"
  for job in "${capacity_jobs[@]}"; do
    state="$(job_state "${job}")"
    capacity_status+=("${job}=${state}")
    if [[ "${state}" == "failed" ]]; then
      echo "[$(timestamp)] prerequisite EMA job ${job} failed; refusing CIC submission" >&2
      exit 1
    fi
    [[ "${state}" == "succeeded" ]] || capacity_ready=0
  done

  if [[ "${train_state}" == "succeeded" && "${capacity_ready}" == "1" ]]; then
    echo "[$(timestamp)] training and capacity jobs completed"
    break
  fi
  echo "[$(timestamp)] waiting: ${TRAIN_JOB}=${train_state} ${capacity_status[*]}"
  sleep "${POLL_SECONDS}"
done

model_root="${TRAIN_ROOT}/cic_only_seed42_s250"
models_root="${model_root}/experiments/models"
for step in 50 100 150 200 250; do
  checkpoint="$(find "${models_root}" -maxdepth 1 -type d \
    -name "checkpoint*_step_${step}" -print -quit)"
  if [[ -z "${checkpoint}" || \
        ! -s "${checkpoint}/transformer/diffusion_pytorch_model.bin" || \
        ! -s "${checkpoint}/transformer/config.json" ]]; then
    echo "[$(timestamp)] incomplete CIC raw checkpoint at step ${step}" >&2
    exit 1
  fi
done

checkpoint250="$(find "${models_root}" -maxdepth 1 -type d \
  -name 'checkpoint*_step_250' -print -quit)"
for weight_kind in transformer transformer_ema; do
  if [[ ! -s "${checkpoint250}/${weight_kind}/diffusion_pytorch_model.bin" || \
        ! -s "${checkpoint250}/${weight_kind}/config.json" ]]; then
    echo "[$(timestamp)] incomplete CIC step250 ${weight_kind}" >&2
    exit 1
  fi
done

checksum_file="${model_root}/step250_transformer.sha256"
[[ -s "${checksum_file}" ]] || {
  echo "[$(timestamp)] missing ${checksum_file}" >&2
  exit 1
}
sha256sum --check "${checksum_file}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

submit_cic() {
  local label="$1"
  local weight_kind="$2"
  local suffix="$3"
  local output_root="$4"
  local probe_root="$5"
  local receipt="$6"
  local model="cic_only_seed42_s250${suffix}"

  if [[ -s "${output_root}/${model}/seed004/_COMPLETE.json" ]]; then
    echo "[$(timestamp)] ${label} CIC output is already complete"
    return 0
  fi
  if [[ -s "${receipt}" ]]; then
    echo "[$(timestamp)] ${label} CIC receipt already exists: ${receipt}"
    return 0
  fi

  echo "[$(timestamp)] submitting CIC-only ${label} seed004 generation"
  submission="$({
    JOB_SCRIPT="${EVEWORLD_ROOT}/eveworld/pipeline/kjob_eve_ablation_generate_chain.sh" \
      bash scripts/submit_gigaworld0_kjob.sh \
        "CHAIN_NAME=strict_seed004_cic_only_${label}" \
        'VARIANTS=cic_only' \
        "WEIGHT_KIND=${weight_kind}" \
        "OUTPUT_MODEL_SUFFIX=${suffix}" \
        "OUTPUT_ROOT=${output_root}" \
        "PROBE_ROOT=${probe_root}"
  } 2>&1)"
  printf '%s\n' "${submission}"

  job_name="$(printf '%s\n' "${submission}" | \
    sed -n 's#^job.batch/\([^ ]*\) created$#\1#p' | tail -n 1)"
  [[ -n "${job_name}" ]] || {
    echo "[$(timestamp)] could not parse ${label} job name" >&2
    exit 1
  }

  temporary="${receipt}.tmp.$$"
  {
    echo "submitted_at=$(timestamp)"
    echo "job=${job_name}"
    echo "train_job=${TRAIN_JOB}"
    echo "capacity_jobs=${CAPACITY_JOBS}"
    echo "variant=cic_only"
    echo "seed=004"
    echo "weight_kind=${weight_kind}"
    echo "output_model=${model}"
    echo "output_root=${output_root}"
    echo "probe_root=${probe_root}"
  } > "${temporary}"
  mv "${temporary}" "${receipt}"
  echo "[$(timestamp)] submitted ${job_name}; receipt=${receipt}"
}

# Separate receipts make reruns safe if the first submission succeeds and the
# second one is interrupted before its receipt is written.
submit_cic raw transformer '' "${RAW_OUTPUT_ROOT}" "${RAW_PROBE_ROOT}" "${RAW_RECEIPT}"
submit_cic ema transformer_ema '_ema' "${EMA_OUTPUT_ROOT}" "${EMA_PROBE_ROOT}" "${EMA_RECEIPT}"
