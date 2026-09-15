#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GAGI="${GAGI:-/data/datasets/gagi}"
GENERATION_ROOT="${GENERATION_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema_ckpt_sweep}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema_ckpt_sweep_eval}"
GEMINI_ROOT="${GEMINI_ROOT:-${GAGI}/eve_v2_outputs/gemini_eval}"
INPUT_ROOT="${INPUT_ROOT:-${GAGI}/gr1_dreamgen_eval/giga_input}"
STATE_ROOT="${STATE_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1/controller_logs}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CONCURRENCY="${CONCURRENCY:-100}"
NAMESPACE="${NAMESPACE:-user-behappya}"
JOB_NAMES="${JOB_NAMES:?JOB_NAMES is required}"
MODELS="${MODELS:-cic_only_seed42_s050_ema cic_only_seed42_s100_ema cic_only_seed42_s150_ema cic_only_seed42_s200_ema}"
RUN_PREFIX="${RUN_PREFIX:-dreamgen_eve_cic_ema_checkpoint_sweep_seed004_qwen_protocol}"
SUMMARY_ROOT="${GEMINI_ROOT}/${RUN_PREFIX}_3run_summary"
MLR_MODELS="${MLR_MODELS:-standard_sft copy_paste_only cwm_only cic_only igt eveworld}"
MLR_INPUT_ROOT="${MLR_INPUT_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_component_table_mlr_inputs}"
MLR_OUTPUT="${MLR_OUTPUT:-${GAGI}/eve_v2_outputs/track4gen_probe/eve_ablation_component_table_seed004_mlr_v2}"
MLR_JOB_SCRIPT="${MLR_JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/pipeline/kjob_eval_t4g_exam_v2.sh}"
MLR_SUBMISSION_RECEIPT="${STATE_ROOT}/eve_ablation_component_table_mlr_submission.txt"
RECEIPT="${STATE_ROOT}/cic_ema_checkpoint_sweep_complete.txt"
LOCK_DIR="${STATE_ROOT}/cic_ema_checkpoint_sweep.lock"

mkdir -p "${STATE_ROOT}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

if [[ -s "${RECEIPT}" && -s "${SUMMARY_ROOT}/summary.json" ]]; then
  echo "[$(timestamp)] checkpoint sweep already complete: ${SUMMARY_ROOT}"
  exit 0
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[$(timestamp)] another watcher owns ${LOCK_DIR}"
  exit 0
fi
cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

job_failed() {
  local job="$1"
  local failed
  failed="$(kubectl get job -n "${NAMESPACE}" "${job}" \
    -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  [[ -n "${failed}" && "${failed}" != "0" ]]
}

job_succeeded() {
  local job="$1"
  local succeeded
  succeeded="$(kubectl get job -n "${NAMESPACE}" "${job}" \
    -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  [[ -n "${succeeded}" && "${succeeded}" != "0" ]]
}

mlr_ready() {
  local summary="${MLR_OUTPUT}/exam_v2_summary.json"
  [[ -s "${summary}" ]] || return 1
  [[ "$(jq '.records | length' "${summary}")" -eq 756 ]] || return 1
  local model
  read -r -a mlr_model_args <<< "${MLR_MODELS}"
  for model in "${mlr_model_args[@]}"; do
    jq -e --arg model "${model}" '.summary[$model].n == 126' "${summary}" >/dev/null || return 1
  done
}

read -r -a model_args <<< "${MODELS}"
read -r -a job_args <<< "${JOB_NAMES}"
while true; do
  missing=()
  for model in "${model_args[@]}"; do
    [[ -s "${GENERATION_ROOT}/${model}/seed004/_COMPLETE.json" ]] || missing+=("${model}")
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "[$(timestamp)] all checkpoint generations complete"
    break
  fi
  for job in "${job_args[@]}"; do
    if job_failed "${job}"; then
      echo "[$(timestamp)] generation job failed: ${job}" >&2
      exit 1
    fi
  done
  echo "[$(timestamp)] waiting for generation: ${missing[*]}"
  sleep "${POLL_SECONDS}"
done

# A completion marker is written just before its generation payload exits. Wait for both
# GPU jobs to release their nodes before submitting MLR, preserving the two-node cap.
while true; do
  pending_jobs=()
  for job in "${job_args[@]}"; do
    if job_failed "${job}"; then
      echo "[$(timestamp)] generation job failed after writing outputs: ${job}" >&2
      exit 1
    fi
    job_succeeded "${job}" || pending_jobs+=("${job}")
  done
  if [[ "${#pending_jobs[@]}" -eq 0 ]]; then
    echo "[$(timestamp)] generation jobs released both GPU nodes"
    break
  fi
  echo "[$(timestamp)] waiting for generation jobs to exit: ${pending_jobs[*]}"
  sleep "${POLL_SECONDS}"
done

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

# Assemble the exact generated videos represented by the mixed-source component table.
# Standard SFT and EVEWorld come from doc65; the four strict ablations use EMA step250.
mlr_labels=(standard_sft copy_paste_only cwm_only cic_only igt eveworld)
mlr_sources=(
  "${GAGI}/eve_v2_outputs/eval175_matched_seeds/round0"
  "${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema/cp_only_seed42_s250_ema"
  "${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema/cwm_only_seed42_s250_ema"
  "${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema/cic_only_seed42_s250_ema"
  "${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema/igt_seed42_s250_ema"
  "${GAGI}/eve_v2_outputs/eval175_multiseed/t4g_wmapA_pre_seed42_s250"
)
mkdir -p "${MLR_INPUT_ROOT}"
for index in "${!mlr_labels[@]}"; do
  label="${mlr_labels[$index]}"
  source_root="${mlr_sources[$index]}"
  [[ -s "${source_root}/seed004/_COMPLETE.json" ]] || {
    echo "Missing component-table generation for ${label}: ${source_root}" >&2
    exit 1
  }
  ln -sfn "${source_root}" "${MLR_INPUT_ROOT}/${label}"
done

mlr_job=""
if mlr_ready; then
  echo "[$(timestamp)] MLR-v2 already complete: ${MLR_OUTPUT}/exam_v2_summary.json"
elif [[ -s "${MLR_SUBMISSION_RECEIPT}" ]]; then
  mlr_job="$(sed -n 's/^job=//p' "${MLR_SUBMISSION_RECEIPT}" | tail -n 1)"
  [[ -n "${mlr_job}" ]] || { echo "Invalid MLR submission receipt" >&2; exit 1; }
  echo "[$(timestamp)] resuming MLR watcher for ${mlr_job}"
else
  echo "[$(timestamp)] submitting six-method component-table MLR-v2 job"
  submit_output="$(
    JOB_SCRIPT="${MLR_JOB_SCRIPT}" bash scripts/submit_gigaworld0_kjob.sh \
      "EXAM_GEN=${MLR_INPUT_ROOT}" \
      'EXAM_SEED=4' \
      "EXAM_ARMS=${MLR_MODELS}" \
      "OUT_DIR=${MLR_OUTPUT}"
  )"
  printf '%s\n' "${submit_output}"
  mlr_job="$(printf '%s\n' "${submit_output}" | \
    sed -n 's|^job.batch/\([^ ]*\) created$|\1|p' | tail -n 1)"
  [[ -n "${mlr_job}" ]] || { echo "Could not parse submitted MLR job" >&2; exit 1; }
  temporary="${MLR_SUBMISSION_RECEIPT}.tmp.$$"
  {
    echo "submitted_at=$(timestamp)"
    echo "job=${mlr_job}"
    echo "models=${MLR_MODELS}"
    echo "generation_root=${MLR_INPUT_ROOT}"
    echo "output=${MLR_OUTPUT}/exam_v2_summary.json"
  } > "${temporary}"
  mv "${temporary}" "${MLR_SUBMISSION_RECEIPT}"
fi

for model in "${model_args[@]}"; do
  echo "[$(timestamp)] auditing ${model}"
  python eveworld/evaluation/eval175_multiseed_prepare.py \
    --generation-root "${GENERATION_ROOT}/${model}" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${MANIFEST_ROOT}/${model}" \
    --model-name "${model}" \
    --seeds 4
  jq -e '.ready == true and .manifest_count == 126 and (.errors | length) == 0' \
    "${MANIFEST_ROOT}/${model}/prepare_report.json" >/dev/null
done

echo "[$(timestamp)] launching three Gemini judge repeats"
MODELS="${MODELS}" MANIFEST_ROOT="${MANIFEST_ROOT}" OUTPUT_ROOT="${GEMINI_ROOT}" \
  RUN_PREFIX="${RUN_PREFIX}" CONCURRENCY="${CONCURRENCY}" \
  bash eveworld/pipeline/run_eve_ablation_gemini_repeats.sh

# Retry only rows that the first pass recorded as errors. Successful rows are resumed untouched.
MODELS="${MODELS}" MANIFEST_ROOT="${MANIFEST_ROOT}" OUTPUT_ROOT="${GEMINI_ROOT}" \
  RUN_PREFIX="${RUN_PREFIX}" CONCURRENCY="${CONCURRENCY}" RERUN_ERRORS=1 \
  bash eveworld/pipeline/run_eve_ablation_gemini_repeats.sh

python eveworld/evaluation/eval175_ablation_repeats_summarize.py \
  --run-dir "${GEMINI_ROOT}/${RUN_PREFIX}_repeat01" \
  --run-dir "${GEMINI_ROOT}/${RUN_PREFIX}_repeat02" \
  --run-dir "${GEMINI_ROOT}/${RUN_PREFIX}_repeat03" \
  --output-dir "${SUMMARY_ROOT}" \
  --expected-models "${model_args[@]}"

while ! mlr_ready; do
  if [[ -z "${mlr_job}" ]]; then
    echo "MLR result is incomplete and no MLR job is recorded" >&2
    exit 1
  fi
  if job_failed "${mlr_job}"; then
    echo "[$(timestamp)] MLR-v2 job failed: ${mlr_job}" >&2
    exit 1
  fi
  if job_succeeded "${mlr_job}"; then
    echo "[$(timestamp)] MLR-v2 job completed without a valid 6x126 summary" >&2
    exit 1
  fi
  echo "[$(timestamp)] waiting for MLR-v2 job ${mlr_job}"
  sleep "${POLL_SECONDS}"
done
echo "[$(timestamp)] MLR-v2 six-method component-table summary complete"

temporary="${RECEIPT}.tmp.$$"
{
  echo "completed_at=$(timestamp)"
  echo "jobs=${JOB_NAMES}"
  echo "generation_root=${GENERATION_ROOT}"
  echo "manifest_root=${MANIFEST_ROOT}"
  echo "summary=${SUMMARY_ROOT}/summary.json"
  echo "table=${SUMMARY_ROOT}/table.csv"
  echo "mlr_job=${mlr_job}"
  echo "mlr_summary=${MLR_OUTPUT}/exam_v2_summary.json"
} > "${temporary}"
mv "${temporary}" "${RECEIPT}"
echo "[$(timestamp)] checkpoint sweep complete: ${SUMMARY_ROOT}"
