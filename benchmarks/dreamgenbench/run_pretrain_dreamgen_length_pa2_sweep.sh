#!/usr/bin/env bash
set -euo pipefail

# Sequentially evaluate the pretrain transformer on the same local GR1
# DreamGen-style inputs used by the SFT length sweep.
#
# For each length:
#   1. submit 8-GPU generation with the pretrain transformer
#   2. wait until generation_summary.json reports all samples generated
#   3. crop side-by-side videos to generated-only videos
#   4. submit VideoPhy PA-II
#   5. wait until PA-II CSV is produced

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${EVAL_ROOT}/eval_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/miniconda/envs/EVEWorld/bin/python}"
VIDEOPHY_CHECKPOINT="${VIDEOPHY_CHECKPOINT:-/data/datasets}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
PA2_GPU_IDS="${PA2_GPU_IDS:-0}"
PA2_BATCH_SIZE="${PA2_BATCH_SIZE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

LENGTH_LABELS=("5p8s" "9p8s" "15p8s")
LENGTH_FRAMES=("93" "157" "253")

summary_value() {
  local summary_path="$1"
  local key="$2"
  python3 - "$summary_path" "$key" <<'PY'
import json
import sys
path, key = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get(key, ""))
PY
}

wait_for_generation() {
  local run_name="$1"
  local save_dir="$2"
  local summary_path="${save_dir}/generation_summary.json"
  local log_path="${save_dir}/run.log"

  echo "Waiting for generation: ${run_name}"
  while true; do
    if [[ -f "${log_path}" ]] && rg -q "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}"; then
      echo "Generation appears to have failed. Check log: ${log_path}" >&2
      rg -n "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}" -S | tail -n 80 >&2
      exit 1
    fi

    if [[ -f "${summary_path}" ]]; then
      local request_count generated_count
      request_count="$(summary_value "${summary_path}" request_count)"
      generated_count="$(summary_value "${summary_path}" generated_count)"
      if [[ "${request_count}" != "" && "${generated_count}" == "${request_count}" && "${generated_count}" != "0" ]]; then
        echo "Generation complete: ${generated_count}/${request_count}"
        break
      fi
    fi

    local count
    count="$(find "${save_dir}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l || true)"
    echo "  generated mp4 count so far: ${count}; sleeping 60s"
    sleep 60
  done
}

wait_for_pa2() {
  local run_name="$1"
  local pa_csv="${EVAL_OUTPUT_ROOT}/${run_name}_pa_ii.csv"
  local log_path="${EVAL_OUTPUT_ROOT}/kjob_logs/${run_name}.log"

  echo "Waiting for PA-II: ${run_name}"
  while true; do
    if [[ -f "${log_path}" ]] && rg -q "Traceback|Error|failed|Killed|CUDA out" "${log_path}"; then
      echo "PA-II appears to have failed. Check log: ${log_path}" >&2
      rg -n "Traceback|Error|failed|Killed|CUDA out" "${log_path}" -S | tail -n 80 >&2
      exit 1
    fi

    if [[ -s "${pa_csv}" ]]; then
      local rows
      rows="$(python3 - "$pa_csv" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)"
      if [[ "${rows}" == "92" ]]; then
        echo "PA-II complete: ${pa_csv}"
        break
      fi
      echo "  PA-II rows so far: ${rows}; sleeping 30s"
    else
      echo "  PA-II CSV not ready; sleeping 30s"
    fi
    sleep 30
  done
}

summarize_pa2() {
  local pa_csv="$1"
  python3 - "$pa_csv" <<'PY'
import csv
import statistics
import sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
pred = [int(float(r["prediction"])) for r in rows]
raw = [float(r["raw_score"]) for r in rows]
print("count:", len(rows))
print("positive:", sum(pred))
print("score:", sum(pred) / len(pred) if pred else "nan")
print("mean_raw:", statistics.mean(raw) if raw else "nan")
print("median_raw:", statistics.median(raw) if raw else "nan")
print("max_raw:", max(raw) if raw else "nan")
PY
}

summarize_peak_memory() {
  local title="$1"
  local peak_json="$2"
  if [[ ! -s "${peak_json}" ]]; then
    echo "${title} GPU peak: missing (${peak_json})"
    return
  fi
  python3 - "$title" "$peak_json" <<'PY'
import json
import sys
title, path = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
used_mib = data.get("peak_memory_used_mib")
used_gib = data.get("peak_memory_used_gib")
gpu_index = data.get("peak_gpu_index")
gpu_name = data.get("peak_gpu_name")
total_mib = None
for row in data.get("latest") or []:
    if str(row.get("gpu_index")) == str(gpu_index):
        total_mib = row.get("memory_total_mib")
        break
total = f", total={total_mib} MiB" if total_mib is not None else ""
print(f"{title} GPU peak: {used_mib} MiB / {used_gib} GiB on GPU {gpu_index} ({gpu_name}{total})")
PY
}

cd "${REPO_DIR}"

echo "Pretrain DreamGen length PA-II sweep"
echo "Repo:              ${REPO_DIR}"
echo "Pretrain dir:      ${PRETRAIN_DIR}"
echo "EVAL_ROOT:         ${EVAL_ROOT}"
echo "Timestamp:         ${TIMESTAMP}"
echo "GPU ids:           ${GPU_IDS}"
echo "PA-II GPU ids:     ${PA2_GPU_IDS}"
echo "PA-II batch size:  ${PA2_BATCH_SIZE}"
echo

for idx in "${!LENGTH_LABELS[@]}"; do
  label="${LENGTH_LABELS[$idx]}"
  frames="${LENGTH_FRAMES[$idx]}"
  run_name="pretrain_dreamgen_8gpu_${label}_full_${TIMESTAMP}"
  save_dir="${OUTPUT_ROOT}/${run_name}"
  video_dir="${EVAL_ROOT}/dreamgenbench_video_dirs/${run_name}"
  pa_run_name="${run_name}_videophy_pa2"
  pa_csv="${EVAL_OUTPUT_ROOT}/${pa_run_name}_pa_ii.csv"

  echo "============================================================"
  echo "Length ${label}: NUM_FRAMES=${frames}, FPS=${FPS}"
  echo "Generation run: ${run_name}"
  echo "============================================================"

  CHECKPOINT_DIR="${PRETRAIN_DIR}" \
  PRETRAIN_DIR="${PRETRAIN_DIR}" \
  USE_EMA=0 \
  RUN_NAME="${run_name}" \
  GPU_IDS="${GPU_IDS}" \
  NUM_FRAMES="${frames}" \
  FPS="${FPS}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_dreamgen_generation_kjob.sh"

  wait_for_generation "${run_name}" "${save_dir}"
  summarize_peak_memory "Generation ${label}" "${save_dir}/gpu_memory_peak.json"

  echo "Preparing generated-only videos: ${video_dir}"
  SOURCE_VIDEO_DIR="${save_dir}" \
    "${PYTHON_BIN}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py" --overwrite

  echo "Submitting PA-II: ${pa_run_name}"
  VIDEO_DIR="${video_dir}" \
  RUN_NAME="${pa_run_name}" \
  CHECKPOINT="${VIDEOPHY_CHECKPOINT}" \
  GPU_IDS="${PA2_GPU_IDS}" \
  BATCH_SIZE="${PA2_BATCH_SIZE}" \
  "${EVEWORLD_ROOT}/benchmarks/pbench/launch_dreamgenbench_videophy_pa2_kjob.sh"

  wait_for_pa2 "${pa_run_name}"

  echo "PA-II summary for ${label}:"
  summarize_pa2 "${pa_csv}"
  summarize_peak_memory "PA-II ${label}" "${EVAL_OUTPUT_ROOT}/kjob_logs/${pa_run_name}_gpu_memory_peak.json"
  echo
done

echo "All pretrain length PA-II runs finished."
