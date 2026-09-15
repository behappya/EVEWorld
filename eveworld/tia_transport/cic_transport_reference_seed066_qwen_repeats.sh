#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="giga-world-0"
CONDA_SH="/home/jovyan/miniconda/etc/profile.d/conda.sh"
CONDA_ENV="giga_models"
ROOT="/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1"
MANIFEST_ROOT="${ROOT}/eval175_references_pretrain_s000_sft_s200_seed066_eval"
OUTPUT_ROOT="${ROOT}/qwen_references_pretrain_s000_sft_s200_seed066_thinking_off"
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-600}"
WAIT_SEC="${WAIT_SEC:-60}"
models=(pretrain_raw_s000 standard_sft_raw_s200)
seeds=(66)

while ! {
  ready=1
  for model in "${models[@]}"; do
    jq -e '.ready == true and .manifest_count == 126 and (.errors | length) == 0' \
      "${MANIFEST_ROOT}/${model}/prepare_report.json" >/dev/null 2>&1 || ready=0
  done
  [[ "${ready}" == "1" ]]
}; do
  printf '[wait] %s generation audit is not ready\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  sleep "${WAIT_SEC}"
done

for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    manifest="${MANIFEST_ROOT}/${model}/manifests/seed$(printf '%03d' "${seed}").jsonl"
    [[ -s "${manifest}" && "$(wc -l <"${manifest}")" -eq 126 ]] || {
      echo "Bad manifest for ${model} seed ${seed}" >&2
      exit 2
    }
  done
done

until models_json="$(curl -fsS --max-time 20 "${QWEN_BASE%/}/models" 2>/dev/null)"; do
  printf '[wait] %s Qwen endpoint is not ready\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  sleep "${WAIT_SEC}"
done
QWEN_MODEL="${QWEN_MODEL:-$(jq -r '.data[0].id // empty' <<<"${models_json}")}"
[[ -n "${QWEN_MODEL}" ]] || { echo "Could not resolve Qwen model id" >&2; exit 3; }

process_count=$((${#models[@]} * ${#seeds[@]} * 3))
per_process=$((GLOBAL_CONCURRENCY / process_count))
((per_process > 0)) || per_process=1
actual_concurrency=$((per_process * process_count))

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"
printf '[inputs] references=%s seeds=%s endpoint=%s judge=%s processes=%d per_process=%d total=%d thinking=off\n' \
  "${models[*]}" "${seeds[*]}" "${QWEN_BASE}" "${QWEN_MODEL}" "${process_count}" \
  "${per_process}" "${actual_concurrency}"

pids=()
labels=()
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    seed_name="seed$(printf '%03d' "${seed}")"
    manifest="${MANIFEST_ROOT}/${model}/manifests/${seed_name}.jsonl"
    for repeat in 01 02 03; do
      run_name="${model}_${seed_name}_repeat${repeat}"
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
      printf '[launch] %s model=%s seed=%s repeat=%s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${model}" "${seed_name}" "${repeat}" \
        | tee -a "${log_path}"
      "${command[@]}" >>"${log_path}" 2>&1 &
      pids+=("$!")
      labels+=("${run_name}")
    done
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
if [[ "${result}" == "0" ]]; then
  python eveworld/tia_transport/cic_transport_reference_seed066_qwen_results.py \
    --qwen-root "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/three_repeat_audit.json"
fi
exit "${result}"
