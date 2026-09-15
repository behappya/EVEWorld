#!/usr/bin/env bash
set -euo pipefail

# Quality-only sweep for PBench rows that are still marked as "待补" in the
# reproduction README. This does not regenerate videos and does not rerun
# PBench Domain/VQA; it only submits VBench/VBench2 quality kjobs one by one.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/giga_world_0_outputs}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"
QUALITY_GPU_IDS="${QUALITY_GPU_IDS:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
EXPECTED_COUNT="${EXPECTED_COUNT:-174}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/data/datasets/gagi/vbench_cache}"

# Keep this aligned with the paper table columns:
# i2v-s, i2v-bg, aes, img, bg-con, mot, sub-con, o-con.
QUALITY_DIMENSIONS="${QUALITY_DIMENSIONS:-i2v_subject i2v_background aesthetic_quality imaging_quality background_consistency motion_smoothness subject_consistency overall_consistency}"

# Already done separately:
#   Pretrain 3.8s -> /data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality
# This script fills the remaining rows in the paper-style table.
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

mp4_count() {
  find "$1" -maxdepth 1 -type f -name 'robot_*.mp4' 2>/dev/null | wc -l
}

quality_done() {
  [[ -f "$1/quality_eval/pbench_robot_quality_overall_summary.json" ]]
}

ensure_dreamsim_models_link() {
  local output_root="$1"
  if [[ ! -d "${BASE_DREAMSIM_MODELS}" ]]; then
    cat >&2 <<EOF
Missing base DreamSim/VBench model cache:
  ${BASE_DREAMSIM_MODELS}

Run this first if needed:
  cd ${REPO_DIR}
  ./benchmarks/pbench/download_vbench_quality_checkpoints.sh
EOF
    exit 1
  fi
  mkdir -p "${output_root}/vbench_work"
  if [[ -e "${output_root}/vbench_work/models" && ! -L "${output_root}/vbench_work/models" ]]; then
    echo "Using existing non-symlink models dir: ${output_root}/vbench_work/models"
  else
    ln -sfn "${BASE_DREAMSIM_MODELS}" "${output_root}/vbench_work/models"
  fi
}

wait_for_quality() {
  local run_name="$1"
  local output_root="$2"
  local run_log="${output_root}/kjob_logs/${run_name}.log"
  echo "Waiting for Quality/VBench: ${run_name}"
  while true; do
    if quality_done "${output_root}"; then
      echo "Quality complete: ${output_root}/quality_eval/pbench_robot_quality_overall_summary.json"
      break
    fi
    if [[ -f "${run_log}" ]] && rg -q "Traceback|RuntimeError|CUDA out|Killed|Checkpoint check failed|Error:" "${run_log}"; then
      echo "Quality appears to have failed. Check log: ${run_log}" >&2
      rg -n "Traceback|RuntimeError|CUDA out|Killed|Checkpoint check failed|Error:" "${run_log}" -S | tail -n 120 >&2
      exit 1
    fi
    echo "  summary not ready; sleeping 60s"
    sleep 60
  done
}

echo "PBench Quality/VBench todo sweep"
echo "Repo:             ${REPO_DIR}"
echo "GPU ids:          ${QUALITY_GPU_IDS}"
echo "Expected videos:  ${EXPECTED_COUNT}"
echo "Dimensions:       ${QUALITY_DIMENSIONS}"
echo "VBench cache:     ${VBENCH_CACHE_DIR}"
echo

for item in "${TASKS[@]}"; do
  IFS='|' read -r model target run_name source_video_dir domain_summary output_root <<< "${item}"
  echo "============================================================"
  echo "${model} ${target}: ${run_name}"
  echo "source: ${source_video_dir}"
  echo "domain: ${domain_summary}"
  echo "output: ${output_root}"

  if quality_done "${output_root}"; then
    echo "Skip: quality already complete"
    continue
  fi

  count="$(mp4_count "${source_video_dir}")"
  if [[ "${count}" != "${EXPECTED_COUNT}" ]]; then
    echo "Skip: expected ${EXPECTED_COUNT} videos, found ${count}: ${source_video_dir}" >&2
    continue
  fi

  if [[ ! -f "${domain_summary}" ]]; then
    echo "Warn: missing domain summary; Quality will run, but Overall Score will be empty until Domain is supplied."
    echo "      ${domain_summary}"
  fi

  ensure_dreamsim_models_link "${output_root}"
  mkdir -p "${output_root}/kjob_logs"

  RUN_NAME="${run_name}_quality" \
  OUTPUT_ROOT="${output_root}" \
  SOURCE_VIDEO_DIR="${source_video_dir}" \
  DOMAIN_SUMMARY="${domain_summary}" \
  METADATA_JSONL="${METADATA_JSONL}" \
  VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR}" \
  GPU_IDS="${QUALITY_GPU_IDS}" \
  GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL}" \
  RUN_LOG="${output_root}/kjob_logs/${run_name}_quality.log" \
  ENV_FILE="${output_root}/kjob_logs/${run_name}_quality.env" \
  GPU_MEMORY_SAMPLES="${output_root}/kjob_logs/${run_name}_quality_gpu_memory_samples.csv" \
  GPU_MEMORY_PEAK="${output_root}/kjob_logs/${run_name}_quality_gpu_memory_peak.json" \
  DIMENSIONS="${QUALITY_DIMENSIONS}" \
  "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_vbench_quality_kjob.sh"

  wait_for_quality "${run_name}_quality" "${output_root}"
done

echo "============================================================"
echo "Quality todo sweep finished. Generate the combined CSV with:"
echo "  python3 ${EVEWORLD_ROOT}/benchmarks/pbench/summarize_pbench_quality_upto19.py"
