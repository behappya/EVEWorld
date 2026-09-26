#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
CONDA_SH="/home/jovyan/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
ROOT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1"
MANIFEST_BASE="${ROOT}/eval175_transport_raw_s150_s200_multiseed70_eval"
S150_MANIFEST_ROOT="${MANIFEST_BASE}/transport_raw_s150_qwen_ready_late06"
S200_MANIFEST_ROOT="${MANIFEST_BASE}/transport_raw_s200_qwen_ready_all70"
S150_BATCH01_AUDIT="${ROOT}/qwen_transport_raw_s150_multiseed70_thinking_off/batch01_three_repeat_audit.json"
OUTPUT_ROOT="${ROOT}/qwen_transport_s150_late06_s200_all70_thinking_off"
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-600}"
WAIT_SEC="${WAIT_SEC:-60}"

[[ "${GLOBAL_CONCURRENCY}" =~ ^[0-9]+$ ]] || { echo "Bad concurrency" >&2; exit 2; }
per_process=$((GLOBAL_CONCURRENCY / 3))
((per_process > 0)) || per_process=1
actual_concurrency=$((per_process * 3))

check_report() {
  local report="$1" expected="$2"
  jq -e --argjson expected "${expected}" \
    '.ready == true and .manifest_count == $expected and (.errors | length) == 0' \
    "${report}" >/dev/null 2>&1
}

wait_for_report() {
  local report="$1" expected="$2" label="$3"
  while ! check_report "${report}" "${expected}"; do
    printf '[wait] %s %s manifest is not ready\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${label}"
    sleep "${WAIT_SEC}"
  done
}

wait_for_endpoint() {
  until models_json="$(curl -fsS --max-time 20 "${QWEN_BASE%/}/models" 2>/dev/null)"; do
    printf '[wait] %s Qwen endpoint is not ready\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    sleep "${WAIT_SEC}"
  done
  QWEN_MODEL="${QWEN_MODEL:-$(jq -r '.data[0].id // empty' <<<"${models_json}")}" 
  [[ -n "${QWEN_MODEL}" ]] || { echo "Could not resolve Qwen model" >&2; exit 3; }
  export QWEN_MODEL
}

run_batch() {
  local prefix="$1" manifest="$2" expected="$3"
  [[ -s "${manifest}" && "$(wc -l <"${manifest}")" -eq "${expected}" ]] || {
    echo "Bad ${prefix} manifest" >&2
    return 2
  }
  printf '[batch] %s prefix=%s items=%s processes=3 per_process=%s total=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${prefix}" "${expected}" \
    "${per_process}" "${actual_concurrency}"
  local pids=() labels=() repeat run_name output_dir log_path
  for repeat in 01 02 03; do
    run_name="${prefix}_repeat${repeat}"
    output_dir="${OUTPUT_ROOT}/${run_name}"
    log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
    printf '[launch] %s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${run_name}" \
      | tee -a "${log_path}"
    python benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.py \
      --manifest "${manifest}" \
      --output-root "${output_dir}" \
      --run-name "${run_name}" \
      --qwen-base "${QWEN_BASE}" \
      --qwen-model "${QWEN_MODEL}" \
      --metrics qwen_if \
      --concurrency "${per_process}" \
      --max-inflight "${per_process}" \
      --frame-count 49 \
      --jpeg-quality 85 \
      --parallel-frame-decode \
      --temperature 0 \
      --disable-thinking \
      --model-timeout 1200 \
      --model-max-tokens 32000 \
      >>"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${run_name}")
  done
  local result=0 index
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "[complete] ${labels[$index]}"
    else
      echo "[failed] ${labels[$index]}" >&2
      result=1
    fi
  done
  return "${result}"
}

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"
wait_for_endpoint
printf '[inputs] endpoint=%s judge=%s thinking=off\n' "${QWEN_BASE}" "${QWEN_MODEL}"

wait_for_report "${S150_MANIFEST_ROOT}/prepare_report.json" 756 s150_late06
run_batch s150_late06 "${S150_MANIFEST_ROOT}/manifests/all_seeds.jsonl" 756

wait_for_report "${S200_MANIFEST_ROOT}/prepare_report.json" 8820 s200_all70
run_batch s200_all70 "${S200_MANIFEST_ROOT}/manifests/all_seeds.jsonl" 8820

python eveworld/tia_transport/cic_transport_remaining_qwen_results.py \
  --qwen-root "${OUTPUT_ROOT}" \
  --s150-batch01-audit "${S150_BATCH01_AUDIT}" \
  --output "${OUTPUT_ROOT}/three_repeat_audit.json"
