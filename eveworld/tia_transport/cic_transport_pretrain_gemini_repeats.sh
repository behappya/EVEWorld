#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
CONDA_SH="/home/jovyan/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="giga_models"
ROOT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1"
GEN_ROOT="${ROOT}/eval175_seed004_pretrain_raw_s000"
MANIFEST_ROOT="${ROOT}/eval175_seed004_pretrain_raw_s000_eval"
OUTPUT_ROOT="${ROOT}/gemini_seed004_pretrain_raw_s000"
RUN_PREFIX="cic_transport_seed004_pretrain_raw_s000_qwen_if"
MODEL="pretrain_raw_s000"
CONCURRENCY="${CONCURRENCY:-100}"
RERUN_ERRORS="${RERUN_ERRORS:-0}"

[[ -s "${GEN_ROOT}/audit/GENERATION_COMPLETE" ]] || {
  echo "Pretrained generation campaign is not complete" >&2
  exit 2
}
report="${MANIFEST_ROOT}/${MODEL}/prepare_report.json"
manifest="${MANIFEST_ROOT}/${MODEL}/manifests/seed004.jsonl"
[[ -s "${report}" && -s "${manifest}" ]] || { echo "Missing pretrained manifest" >&2; exit 2; }
jq -e '.ready == true and .manifest_count == 126 and (.errors | length) == 0' \
  "${report}" >/dev/null

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
    --manifest "${manifest}"
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

python eveworld/tia_transport/cic_transport_pretrain_results.py \
  --gemini-root "${OUTPUT_ROOT}" \
  --run-prefix "${RUN_PREFIX}" \
  --output "${OUTPUT_ROOT}/three_repeat_audit.json"
