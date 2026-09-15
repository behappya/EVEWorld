#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
CONDA_SH="/home/jovyan/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="giga_models"
ROOT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1"
MANIFEST_ROOT="${ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250_eval"
OUTPUT_ROOT="${ROOT}/qwen_seed004_transport_seed6666_raw_s150_s250_thinking_off"
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-100}"
RERUN_ERRORS="${RERUN_ERRORS:-0}"
models=(
  transport_seed6666_raw_s150
  transport_seed6666_raw_s200
  transport_seed6666_raw_s250
)

for model in "${models[@]}"; do
  report="${MANIFEST_ROOT}/${model}/prepare_report.json"
  manifest="${MANIFEST_ROOT}/${model}/manifests/seed004.jsonl"
  [[ -s "${report}" && -s "${manifest}" ]] || {
    echo "Missing ${model} manifest" >&2
    exit 2
  }
  jq -e '.ready == true and .manifest_count == 126 and (.errors | length) == 0' \
    "${report}" >/dev/null
  [[ "$(wc -l <"${manifest}")" -eq 126 ]] || {
    echo "${model}: manifest line count is not 126" >&2
    exit 2
  }
done

models_json="$(curl -fsS --max-time 20 "${QWEN_BASE%/}/models")" || {
  echo "Qwen endpoint is not ready: ${QWEN_BASE}" >&2
  exit 3
}
QWEN_MODEL="${QWEN_MODEL:-$(jq -r '.data[0].id // empty' <<<"${models_json}")}"
[[ -n "${QWEN_MODEL}" ]] || { echo "Could not resolve Qwen model id" >&2; exit 3; }

process_count=$((${#models[@]} * 3))
per_process=$((GLOBAL_CONCURRENCY / process_count))
((per_process > 0)) || per_process=1
actual_concurrency=$((per_process * process_count))
printf '[inputs] models=%s endpoint=%s qwen_model=%s processes=%d per_process=%d total=%d thinking=off\n' \
  "${models[*]}" "${QWEN_BASE}" "${QWEN_MODEL}" "${process_count}" \
  "${per_process}" "${actual_concurrency}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"
pids=()
labels=()

for model in "${models[@]}"; do
  manifest="${MANIFEST_ROOT}/${model}/manifests/seed004.jsonl"
  for repeat in 01 02 03; do
    run_name="${model}_repeat${repeat}"
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
      --parallel-frame-decode
      --temperature 0
      --disable-thinking
      --model-timeout 1200
      --model-max-tokens 32000
    )
    if [[ "${RERUN_ERRORS}" == "1" ]]; then
      command+=(--rerun-errors)
    fi
    printf '[launch] %s model=%s repeat=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${model}" "${repeat}" \
      | tee -a "${log_path}"
    "${command[@]}" >>"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${run_name}")
  done
done

result=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[complete] ${labels[$index]}"
  else
    echo "[failed] ${labels[$index]}" >&2
    result=1
  fi
done
exit "${result}"
