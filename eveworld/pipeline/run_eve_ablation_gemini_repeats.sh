#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/data/datasets/gagi/eve_v2_outputs/eve_ablation_strict_v1_eval175_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/eve_v2_outputs/gemini_eval}"
RUN_PREFIX="${RUN_PREFIX:-dreamgen_eve_ablation_strict_v1_seed004_qwen_protocol}"
MODELS="${MODELS:-eveworld_seed42_s250 matched_sft_seed42_s250 cp_only_seed42_s250}"
CONCURRENCY="${CONCURRENCY:-100}"
RERUN_ERRORS="${RERUN_ERRORS:-0}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

read -r -a model_args <<< "${MODELS}"
manifest_args=()
for model in "${model_args[@]}"; do
  report="${MANIFEST_ROOT}/${model}/prepare_report.json"
  manifest="${MANIFEST_ROOT}/${model}/manifests/seed004.jsonl"
  [[ -s "${report}" && -s "${manifest}" ]] || {
    echo "Missing audited manifest for ${model}" >&2
    exit 1
  }
  jq -e '.ready == true and .manifest_count == 126 and (.errors | length) == 0' \
    "${report}" >/dev/null
  manifest_args+=(--manifest "${manifest}")
done

log_root="${OUTPUT_ROOT}/${RUN_PREFIX}_logs"
mkdir -p "${log_root}"
pids=()
repeats=()

for repeat in 01 02 03; do
  output_dir="${OUTPUT_ROOT}/${RUN_PREFIX}_repeat${repeat}"
  log_path="${log_root}/repeat${repeat}.log"
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
  echo "[launch] repeat=${repeat} models=${MODELS} output=${output_dir} log=${log_path}"
  printf '\n[resume-launch] %s models=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "${MODELS}" >>"${log_path}"
  "${command[@]}" >>"${log_path}" 2>&1 &
  pids+=("$!")
  repeats+=("${repeat}")
done

result=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[complete] repeat=${repeats[$index]}"
  else
    echo "[failed] repeat=${repeats[$index]}" >&2
    result=1
  fi
done
exit "${result}"
