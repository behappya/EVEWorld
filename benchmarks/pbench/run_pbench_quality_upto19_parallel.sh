#!/usr/bin/env bash
set -euo pipefail

# Submit the remaining PBench Quality/VBench rows concurrently.
# Each kjob requests one GPU. On an 8-GPU node/cluster this is much faster than
# the sequential todo script, while still keeping each VBench process simple.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/giga_world_0_outputs}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"
EXPECTED_COUNT="${EXPECTED_COUNT:-174}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/data/datasets/gagi/vbench_cache}"
GPU_IDS="${GPU_IDS:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
SKIP_IN_PROGRESS_BY_LOG="${SKIP_IN_PROGRESS_BY_LOG:-1}"
ALLOW_WITH_SEQUENTIAL="${ALLOW_WITH_SEQUENTIAL:-0}"

QUALITY_DIMENSIONS="${QUALITY_DIMENSIONS:-i2v_subject i2v_background aesthetic_quality imaging_quality background_consistency motion_smoothness subject_consistency overall_consistency}"

TASKS=(
  "Pretrain|5.8s|pbench_robot_5p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_length_sweep/pbench_robot_5p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_qwen_vqa_eval/pbench_robot_5p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/pbench_robot_vbench_quality_length_sweep/pbench_robot_5p8s_full_pbench_len_sweep_full_8gpu"
  "Pretrain|9.8s|pbench_robot_9p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_length_sweep/pbench_robot_9p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_qwen_vqa_eval/pbench_robot_9p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/pbench_robot_vbench_quality_length_sweep/pbench_robot_9p8s_full_pbench_len_sweep_full_8gpu"
  "Pretrain|15.8s|pbench_robot_15p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_length_sweep/pbench_robot_15p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_qwen_vqa_eval/pbench_robot_15p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/pbench_robot_vbench_quality_length_sweep/pbench_robot_15p8s_full_pbench_len_sweep_full_8gpu"
  "Pretrain|19.8s|pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_length_sweep/pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/pbench_robot_qwen_vqa_eval/pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/pbench_robot_vbench_quality_length_sweep/pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu"
  "GR1/SFT|3.8s|gr1_pbench_robot_3p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_length_sweep/gr1_pbench_robot_3p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_3p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_3p8s_full_gr1_pbench_len_sweep_full_8gpu"
  "GR1/SFT|5.8s|gr1_pbench_robot_5p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_length_sweep/gr1_pbench_robot_5p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_5p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_5p8s_full_gr1_pbench_len_sweep_full_8gpu"
  "GR1/SFT|9.8s|gr1_pbench_robot_9p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_length_sweep/gr1_pbench_robot_9p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_9p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_9p8s_full_gr1_pbench_len_sweep_full_8gpu"
  "GR1/SFT|15.8s|gr1_pbench_robot_15p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_length_sweep/gr1_pbench_robot_15p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_15p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_15p8s_full_gr1_pbench_len_sweep_full_8gpu"
  "GR1/SFT|19.8s|gr1_pbench_robot_19p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_length_sweep/gr1_pbench_robot_19p8s_full_gr1_pbench_len_sweep_full_8gpu|${EVAL_ROOT}/gr1_pbench_robot_qwen_vqa_eval/gr1_pbench_robot_19p8s_full_gr1_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000/qwen_vqa_summary.json|${EVAL_ROOT}/gr1_pbench_robot_vbench_quality_length_sweep/gr1_pbench_robot_19p8s_full_gr1_pbench_len_sweep_full_8gpu"
)

BASE_DREAMSIM_MODELS="${EVAL_ROOT}/pbench_robot_vbench_quality/vbench_work/models"

if [[ "${ALLOW_WITH_SEQUENTIAL}" != "1" ]]; then
  if pgrep -af "run_pbench_quality_upto19_todo.sh" | rg -v "rg|parallel" >/dev/null; then
    cat >&2 <<EOF
The sequential quality script is still running. Stop it first to avoid duplicate submissions.

Recommended:
  Ctrl-C in the terminal running ./benchmarks/pbench/run_pbench_quality_upto19_todo.sh

Or, from another shell:
  pkill -f run_pbench_quality_upto19_todo.sh

The currently running kjob will continue; this parallel script will skip it by log.
EOF
    exit 2
  fi
fi

mp4_count() {
  find "$1" -maxdepth 1 -type f -name 'robot_*.mp4' 2>/dev/null | wc -l
}

quality_summary() {
  echo "$1/quality_eval/pbench_robot_quality_overall_summary.json"
}

run_log() {
  local run_name="$1"
  local output_root="$2"
  echo "${output_root}/kjob_logs/${run_name}_quality.log"
}

quality_done() {
  [[ -f "$(quality_summary "$1")" ]]
}

ensure_dreamsim_models_link() {
  local output_root="$1"
  [[ -d "${BASE_DREAMSIM_MODELS}" ]] || {
    echo "Missing base model cache: ${BASE_DREAMSIM_MODELS}" >&2
    echo "Run ./benchmarks/pbench/download_vbench_quality_checkpoints.sh first." >&2
    exit 1
  }
  mkdir -p "${output_root}/vbench_work"
  if [[ -e "${output_root}/vbench_work/models" && ! -L "${output_root}/vbench_work/models" ]]; then
    echo "Using existing models dir: ${output_root}/vbench_work/models"
  else
    ln -sfn "${BASE_DREAMSIM_MODELS}" "${output_root}/vbench_work/models"
  fi
}

submit_task() {
  local model="$1" target="$2" run_name="$3" source_video_dir="$4" domain_summary="$5" output_root="$6"
  local log_path
  log_path="$(run_log "${run_name}" "${output_root}")"

  if quality_done "${output_root}"; then
    echo "DONE ${model} ${target}"
    return
  fi
  if [[ "${SKIP_IN_PROGRESS_BY_LOG}" == "1" && -f "${log_path}" ]]; then
    echo "SKIP-IN-PROGRESS ${model} ${target}: ${log_path}"
    return
  fi
  local count
  count="$(mp4_count "${source_video_dir}")"
  if [[ "${count}" != "${EXPECTED_COUNT}" ]]; then
    echo "SKIP-MISSING-VIDEOS ${model} ${target}: ${count}/${EXPECTED_COUNT} ${source_video_dir}" >&2
    return
  fi

  ensure_dreamsim_models_link "${output_root}"
  mkdir -p "${output_root}/kjob_logs"

  echo "SUBMIT ${model} ${target}: ${run_name}"
  RUN_NAME="${run_name}_quality" \
  OUTPUT_ROOT="${output_root}" \
  SOURCE_VIDEO_DIR="${source_video_dir}" \
  DOMAIN_SUMMARY="${domain_summary}" \
  METADATA_JSONL="${METADATA_JSONL}" \
  VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR}" \
  GPU_IDS="${GPU_IDS}" \
  GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL}" \
  RUN_LOG="${log_path}" \
  ENV_FILE="${output_root}/kjob_logs/${run_name}_quality.env" \
  GPU_MEMORY_SAMPLES="${output_root}/kjob_logs/${run_name}_quality_gpu_memory_samples.csv" \
  GPU_MEMORY_PEAK="${output_root}/kjob_logs/${run_name}_quality_gpu_memory_peak.json" \
  DIMENSIONS="${QUALITY_DIMENSIONS}" \
  "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_vbench_quality_kjob.sh"
}

echo "Parallel PBench Quality/VBench submitter"
echo "Each task uses one GPU kjob. GPU_IDS inside each pod: ${GPU_IDS}"
echo "Task count: ${#TASKS[@]}"
echo

for item in "${TASKS[@]}"; do
  IFS='|' read -r model target run_name source_video_dir domain_summary output_root <<< "${item}"
  submit_task "${model}" "${target}" "${run_name}" "${source_video_dir}" "${domain_summary}" "${output_root}"
done

echo
echo "Monitoring summaries every ${POLL_INTERVAL_SEC}s..."
while true; do
  done_count=0
  total=0
  for item in "${TASKS[@]}"; do
    IFS='|' read -r _model _target _run_name _source_video_dir _domain_summary output_root <<< "${item}"
    total=$((total + 1))
    if quality_done "${output_root}"; then
      done_count=$((done_count + 1))
    fi
  done
  remaining=$((total - done_count))
  pct="$(python3 - <<PY
total=${total}
done=${done_count}
print(f"{done / total * 100:.1f}")
PY
)"
  echo "$(date '+%F %T') done=${done_count}/${total} (${pct}%), remaining=${remaining}"
  [[ "${done_count}" == "${total}" ]] && break
  sleep "${POLL_INTERVAL_SEC}"
done

echo "All quality summaries are complete."
python3 "${EVEWORLD_ROOT}/benchmarks/pbench/summarize_pbench_quality_upto19.py"
