#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
CONDA_SH="/home/jovyan/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
ROOT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1"
GEN_ROOT="${ROOT}/eval175_seed004_raw_s150_s250"
MANIFEST_ROOT="${ROOT}/eval175_seed004_raw_s150_s250_eval"
OUTPUT_ROOT="${ROOT}/gemini_seed004_raw_s150_s250"
RUN_PREFIX="cic_transport_seed004_raw_s150_s250_qwen_if"
CONCURRENCY="${CONCURRENCY:-100}"
RERUN_ERRORS="${RERUN_ERRORS:-0}"
PARTIAL="${PARTIAL:-0}"
models=(
  control_raw_s150 control_raw_s200 control_raw_s250
  transport_raw_s150 transport_raw_s200 transport_raw_s250
)

if [[ "${PARTIAL}" != "1" ]]; then
  [[ -s "${GEN_ROOT}/audit/CONTROL_COMPLETE" ]] || { echo "Control generation is incomplete" >&2; exit 2; }
  [[ -s "${GEN_ROOT}/audit/TRANSPORT_COMPLETE" ]] || { echo "Transport generation is incomplete" >&2; exit 2; }
fi

manifest_args=()
ready_models=()
for model in "${models[@]}"; do
  report="${MANIFEST_ROOT}/${model}/prepare_report.json"
  manifest="${MANIFEST_ROOT}/${model}/manifests/seed004.jsonl"
  if [[ ! -s "${report}" || ! -s "${manifest}" ]] || \
     ! jq -e '.ready == true and .manifest_count == 126 and (.errors | length) == 0' \
       "${report}" >/dev/null 2>&1; then
    if [[ "${PARTIAL}" == "1" ]]; then
      echo "[skip-not-ready] ${model}"
      continue
    fi
    echo "Missing or invalid ${model} manifest" >&2
    exit 2
  fi
  manifest_args+=(--manifest "${manifest}")
  ready_models+=("${model}")
done
[[ "${#ready_models[@]}" -gt 0 ]] || { echo "No ready models" >&2; exit 2; }
printf '[inputs] partial=%s models=%s\n' "${PARTIAL}" "${ready_models[*]}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"
pids=()

for repeat in 01 02 03; do
  output_dir="${OUTPUT_ROOT}/${RUN_PREFIX}_repeat${repeat}"
  log_path="${OUTPUT_ROOT}/logs/repeat${repeat}.log"
  command=(
    python eveworld/evaluation/eval_gemini_dreamgen_qwen_protocol.py
    "${manifest_args[@]}"
    --output-dir "${output_dir}"
    --model gemini-3.5-flash
    --metrics qwen_if
    --concurrency "${CONCURRENCY}"
    --frame-count 49
    --jpeg-quality 85
    --temperature 0
    --thinking-level low
    --no-include-thoughts
    --model-timeout 1200
    --model-max-tokens 32000
  )
  if [[ "${RERUN_ERRORS}" == "1" ]]; then
    command+=(--rerun-errors)
  fi
  printf '[launch] %s repeat=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${repeat}" | tee -a "${log_path}"
  "${command[@]}" >>"${log_path}" 2>&1 &
  pids+=("$!")
done

result=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[complete] repeat=$((index + 1))"
  else
    echo "[failed] repeat=$((index + 1))" >&2
    result=1
  fi
done
[[ "${result}" == "0" ]] || exit "${result}"

if [[ "${PARTIAL}" == "1" ]]; then
  echo "[partial-complete] Final 756-row audit deferred until all six manifests are ready."
  exit 0
fi

python eveworld/tia_transport/cic_transport_eval250_results.py \
  --gemini-root "${OUTPUT_ROOT}" \
  --run-prefix "${RUN_PREFIX}" \
  --output "${OUTPUT_ROOT}/three_repeat_audit.json"
