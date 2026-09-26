#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
CONDA_SH="/home/jovyan/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
ROOT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${ROOT}/qwen_seed004_references_thinking_off"
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
EXTRA_CONCURRENCY="${EXTRA_CONCURRENCY:-204}"
RERUN_ERRORS="${RERUN_ERRORS:-0}"
labels=(pretrain_raw_s000 standard_sft_seed004)
manifests=(
  "${ROOT}/eval175_seed004_pretrain_raw_s000_eval/pretrain_raw_s000/manifests/seed004.jsonl"
  "/data/datasets/gagi/eve_v2_outputs/eval175_matched_seeds_eval/round0_seed040_004/manifests/seed004.jsonl"
)

for manifest in "${manifests[@]}"; do
  [[ -s "${manifest}" ]] || { echo "Missing manifest ${manifest}" >&2; exit 2; }
  [[ "$(wc -l <"${manifest}")" == "126" ]] || { echo "Manifest is not 126 rows: ${manifest}" >&2; exit 2; }
done

models_json="$(curl -fsS --max-time 20 "${QWEN_BASE%/}/models")" || {
  echo "Qwen endpoint is not ready: ${QWEN_BASE}" >&2
  exit 3
}
QWEN_MODEL="${QWEN_MODEL:-$(jq -r '.data[0].id // empty' <<<"${models_json}")}"
[[ -n "${QWEN_MODEL}" ]] || { echo "Could not resolve Qwen model id" >&2; exit 3; }

process_count=6
per_process=$((EXTRA_CONCURRENCY / process_count))
((per_process > 0)) || per_process=1
actual_concurrency=$((per_process * process_count))
printf '[inputs] labels=%s endpoint=%s qwen_model=%s processes=%d per_process=%d total=%d thinking=off\n' \
  "${labels[*]}" "${QWEN_BASE}" "${QWEN_MODEL}" "${process_count}" "${per_process}" "${actual_concurrency}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"
pids=()
run_labels=()

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  manifest="${manifests[$index]}"
  for repeat in 01 02 03; do
    run_name="${label}_repeat${repeat}"
    output_dir="${OUTPUT_ROOT}/${run_name}"
    log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
    command=(
      python benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.py
      --manifest "${manifest}"
      --output-root "${output_dir}"
      --run-name "${run_name}"
      --qwen-base "${QWEN_BASE}"
      --qwen-model "${QWEN_MODEL}"
      --metrics qwen_if
      --concurrency "${per_process}"
      --max-inflight "${per_process}"
      --frame-count 49
      --jpeg-quality 85
      --temperature 0
      --disable-thinking
      --model-timeout 1200
      --model-max-tokens 32000
    )
    if [[ "${RERUN_ERRORS}" == "1" ]]; then
      command+=(--rerun-errors)
    fi
    printf '[launch] %s label=%s repeat=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${label}" "${repeat}" | tee -a "${log_path}"
    "${command[@]}" >>"${log_path}" 2>&1 &
    pids+=("$!")
    run_labels+=("${run_name}")
  done
done

result=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[complete] ${run_labels[$index]}"
  else
    echo "[failed] ${run_labels[$index]}" >&2
    result=1
  fi
done
exit "${result}"
