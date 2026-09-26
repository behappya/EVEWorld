#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
CONDA_SH="/home/jovyan/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
GAGI="/data/datasets/gagi"
ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
GENERATION_ROOT="${ROOT}/eval175_transport_raw_s150_s200_multiseed70/transport_raw_s150"
MANIFEST_BASE="${ROOT}/eval175_transport_raw_s150_s200_multiseed70_eval"
QWEN_ROOT="${ROOT}/qwen_transport_raw_s150_multiseed70_thinking_off"
BATCH="${BATCH:-batch01}"
QWEN_BASE="${QWEN_BASE:?Set QWEN_BASE to the ready Qwen /v1 endpoint}"
GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-600}"
RERUN_ERRORS="${RERUN_ERRORS:-0}"
MODEL_NAME="transport_raw_s150"
MANIFEST_ROOT="${MANIFEST_BASE}/${MODEL_NAME}_qwen_ready_${BATCH}"
SEED_FILE="${MANIFEST_ROOT}/frozen_seeds.txt"

[[ "${BATCH}" =~ ^batch[0-9][0-9]$ ]] || { echo "Bad BATCH=${BATCH}" >&2; exit 2; }
[[ "${GLOBAL_CONCURRENCY}" =~ ^[0-9]+$ ]] || { echo "Bad GLOBAL_CONCURRENCY" >&2; exit 2; }

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

if [[ ! -e "${MANIFEST_ROOT}" ]]; then
  mapfile -t seeds < <(
    find "${GENERATION_ROOT}" -mindepth 2 -maxdepth 2 -type f -name _COMPLETE.json -printf '%h\n' \
      | sed -nE 's#^.*/seed([0-9]{3})$#\1#p' \
      | sort -n \
      | sed -E 's/^0*([0-9]+)$/\1/'
  )
  [[ "${#seeds[@]}" -gt 0 ]] || { echo "No complete seeds found" >&2; exit 2; }
  [[ " ${seeds[*]} " == *" 4 "* ]] || { echo "Frozen batch must contain seed004" >&2; exit 2; }
  mkdir -p "${MANIFEST_ROOT}"
  printf '%03d\n' "${seeds[@]}" >"${SEED_FILE}"
  printf '[freeze] batch=%s count=%d seeds=%s\n' "${BATCH}" "${#seeds[@]}" "${seeds[*]}"
  python eveworld/evaluation/eval175_multiseed_prepare.py \
    --generation-root "${GENERATION_ROOT}" \
    --input-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${MANIFEST_ROOT}" \
    --model-name "${MODEL_NAME}" \
    --seeds "${seeds[@]}" \
    >"${MANIFEST_ROOT}/prepare_stdout.log"
else
  [[ -s "${SEED_FILE}" ]] || { echo "Missing frozen seed file: ${SEED_FILE}" >&2; exit 2; }
  mapfile -t seeds < <(sed -nE 's/^0*([0-9]+)$/\1/p' "${SEED_FILE}")
fi

report="${MANIFEST_ROOT}/prepare_report.json"
manifest="${MANIFEST_ROOT}/manifests/all_seeds.jsonl"
expected=$((${#seeds[@]} * 126))
jq -e --argjson expected "${expected}" \
  '.ready == true and .manifest_count == $expected and (.errors | length) == 0' \
  "${report}" >/dev/null
[[ "$(wc -l <"${manifest}")" -eq "${expected}" ]] || {
  echo "Manifest line count does not match ${expected}" >&2
  exit 2
}

models_json="$(curl -fsS --max-time 20 "${QWEN_BASE%/}/models")" || {
  echo "Qwen endpoint is not ready: ${QWEN_BASE}" >&2
  exit 3
}
QWEN_MODEL="${QWEN_MODEL:-$(jq -r '.data[0].id // empty' <<<"${models_json}")}"
[[ -n "${QWEN_MODEL}" ]] || { echo "Could not resolve Qwen model id" >&2; exit 3; }

per_process=$((GLOBAL_CONCURRENCY / 3))
((per_process > 0)) || per_process=1
actual_concurrency=$((per_process * 3))
mkdir -p "${QWEN_ROOT}/logs"
printf '[inputs] batch=%s seeds=%d items=%d endpoint=%s model=%s processes=3 per_process=%d total=%d thinking=off\n' \
  "${BATCH}" "${#seeds[@]}" "${expected}" "${QWEN_BASE}" "${QWEN_MODEL}" \
  "${per_process}" "${actual_concurrency}"

pids=()
labels=()
for repeat in 01 02 03; do
  run_name="${BATCH}_repeat${repeat}"
  output_dir="${QWEN_ROOT}/${run_name}"
  log_path="${QWEN_ROOT}/logs/${run_name}.log"
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
  printf '[launch] %s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${run_name}" | tee -a "${log_path}"
  "${command[@]}" >>"${log_path}" 2>&1 &
  pids+=("$!")
  labels+=("${run_name}")
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
