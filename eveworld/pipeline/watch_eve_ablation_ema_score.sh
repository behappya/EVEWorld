#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GAGI="${GAGI:-/data/datasets/gagi}"
RAW_GENERATION_ROOT="${RAW_GENERATION_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175}"
EMA_GENERATION_ROOT="${EMA_GENERATION_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema}"
RAW_MANIFEST_ROOT="${RAW_MANIFEST_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_eval}"
EMA_MANIFEST_ROOT="${EMA_MANIFEST_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema_eval}"
GEMINI_ROOT="${GEMINI_ROOT:-${GAGI}/eve_v2_outputs/gemini_eval}"
INPUT_ROOT="${INPUT_ROOT:-${GAGI}/gr1_dreamgen_eval/giga_input}"
STATE_ROOT="${STATE_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1/controller_logs}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CONCURRENCY="${CONCURRENCY:-100}"
NAMESPACE="${NAMESPACE:-user-behappya}"
INITIAL_EMA_JOBS="${INITIAL_EMA_JOBS:-slurm-profile-slurm-xbtd7 slurm-profile-slurm-cghw2}"
RAW_PREFIX="dreamgen_eve_ablation_strict_v1_seed004_qwen_protocol"
EMA_PREFIX="dreamgen_eve_ablation_strict_v1_seed004_ema_qwen_protocol"
RAW_MODELS="eveworld_seed42_s250 matched_sft_seed42_s250 cp_only_seed42_s250 cwm_only_seed42_s250 igt_seed42_s250 cic_only_seed42_s250"
EMA_MODELS="eveworld_seed42_s250_ema matched_sft_seed42_s250_ema cp_only_seed42_s250_ema cwm_only_seed42_s250_ema igt_seed42_s250_ema cic_only_seed42_s250_ema"
COMPARISON_ROOT="${GEMINI_ROOT}/${EMA_PREFIX}_raw_comparison"
RECEIPT="${STATE_ROOT}/ema_raw_comparison_complete.txt"
LOCK_DIR="${STATE_ROOT}/ema_score_pipeline.lock"

mkdir -p "${STATE_ROOT}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

if [[ -s "${RECEIPT}" && -s "${COMPARISON_ROOT}/comparison.json" ]]; then
  echo "[$(timestamp)] comparison already complete: ${COMPARISON_ROOT}"
  exit 0
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[$(timestamp)] another scoring watcher owns ${LOCK_DIR}"
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

read -r -a initial_jobs <<< "${INITIAL_EMA_JOBS}"
read -r -a ema_models <<< "${EMA_MODELS}"
while true; do
  missing=()
  [[ -s "${RAW_GENERATION_ROOT}/cic_only_seed42_s250/seed004/_COMPLETE.json" ]] || \
    missing+=("raw/cic_only_seed42_s250")
  for model in "${ema_models[@]}"; do
    [[ -s "${EMA_GENERATION_ROOT}/${model}/seed004/_COMPLETE.json" ]] || \
      missing+=("ema/${model}")
  done
  if [[ "${#missing[@]}" == "0" ]]; then
    echo "[$(timestamp)] all raw/EMA generation completion markers found"
    break
  fi

  for job in "${initial_jobs[@]}"; do
    if job_failed "${job}"; then
      echo "[$(timestamp)] generation job ${job} failed" >&2
      exit 1
    fi
  done
  for receipt in \
    "${STATE_ROOT}/cic_seed004_raw_submission.txt" \
    "${STATE_ROOT}/cic_seed004_ema_submission.txt"; do
    if [[ -s "${receipt}" ]]; then
      job="$(sed -n 's/^job=//p' "${receipt}" | tail -n 1)"
      if [[ -n "${job}" ]] && job_failed "${job}"; then
        echo "[$(timestamp)] CIC generation job ${job} failed" >&2
        exit 1
      fi
    fi
  done
  echo "[$(timestamp)] waiting for generation: ${missing[*]}"
  sleep "${POLL_SECONDS}"
done

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

prepare_manifest() {
  local model="$1"
  local generation_root="$2"
  local manifest_root="$3"
  python eveworld/evaluation/eval175_multiseed_prepare.py \
    --generation-root "${generation_root}/${model}" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${manifest_root}/${model}" \
    --model-name "${model}" \
    --seeds 4
  jq -e '.ready == true and .manifest_count == 126 and (.errors | length) == 0' \
    "${manifest_root}/${model}/prepare_report.json" >/dev/null
}

echo "[$(timestamp)] auditing raw CIC videos"
prepare_manifest cic_only_seed42_s250 "${RAW_GENERATION_ROOT}" "${RAW_MANIFEST_ROOT}"
echo "[$(timestamp)] auditing six EMA models"
for model in "${ema_models[@]}"; do
  prepare_manifest "${model}" "${EMA_GENERATION_ROOT}" "${EMA_MANIFEST_ROOT}"
done

score_repeats() {
  local models="$1"
  local manifest_root="$2"
  local prefix="$3"
  echo "[$(timestamp)] scoring ${prefix}"
  MODELS="${models}" MANIFEST_ROOT="${manifest_root}" OUTPUT_ROOT="${GEMINI_ROOT}" \
    RUN_PREFIX="${prefix}" \
    CONCURRENCY="${CONCURRENCY}" bash eveworld/pipeline/run_eve_ablation_gemini_repeats.sh
  # Retry only rows recorded as errors; completed rows remain untouched.
  MODELS="${models}" MANIFEST_ROOT="${manifest_root}" OUTPUT_ROOT="${GEMINI_ROOT}" \
    RUN_PREFIX="${prefix}" \
    CONCURRENCY="${CONCURRENCY}" RERUN_ERRORS=1 \
    bash eveworld/pipeline/run_eve_ablation_gemini_repeats.sh
}

summarize_repeats() {
  local models="$1"
  local prefix="$2"
  local output_dir="${GEMINI_ROOT}/${prefix}_3run_summary"
  read -r -a model_args <<< "${models}"
  python eveworld/evaluation/eval175_ablation_repeats_summarize.py \
    --run-dir "${GEMINI_ROOT}/${prefix}_repeat01" \
    --run-dir "${GEMINI_ROOT}/${prefix}_repeat02" \
    --run-dir "${GEMINI_ROOT}/${prefix}_repeat03" \
    --output-dir "${output_dir}" \
    --expected-models "${model_args[@]}"
}

# Complete the existing raw runs with CIC first, then score the independent EMA set.
score_repeats "${RAW_MODELS}" "${RAW_MANIFEST_ROOT}" "${RAW_PREFIX}"
summarize_repeats "${RAW_MODELS}" "${RAW_PREFIX}"
score_repeats "${EMA_MODELS}" "${EMA_MANIFEST_ROOT}" "${EMA_PREFIX}"
summarize_repeats "${EMA_MODELS}" "${EMA_PREFIX}"

python eveworld/evaluation/eval175_raw_ema_compare.py \
  --raw-summary "${GEMINI_ROOT}/${RAW_PREFIX}_3run_summary/summary.json" \
  --ema-summary "${GEMINI_ROOT}/${EMA_PREFIX}_3run_summary/summary.json" \
  --output-dir "${COMPARISON_ROOT}"

temporary="${RECEIPT}.tmp.$$"
{
  echo "completed_at=$(timestamp)"
  echo "raw_summary=${GEMINI_ROOT}/${RAW_PREFIX}_3run_summary/summary.json"
  echo "ema_summary=${GEMINI_ROOT}/${EMA_PREFIX}_3run_summary/summary.json"
  echo "comparison=${COMPARISON_ROOT}/comparison.json"
} > "${temporary}"
mv "${temporary}" "${RECEIPT}"
echo "[$(timestamp)] raw/EMA comparison complete: ${COMPARISON_ROOT}"
