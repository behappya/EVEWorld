#!/usr/bin/env bash
set -euo pipefail

# Resumable long-video PA-II sweep for the local 92 GR1 DreamGen-style inputs.
#
# This evaluates both:
#   - SFT checkpoint transformer_ema
#   - Pretrain transformer
#
# For each model and length:
#   1. submit or wait for 8-GPU generation
#   2. crop side-by-side videos to generated-only videos
#   3. submit or wait for VideoPhy PA-II
#   4. print generation/PA-II memory and PA-II score
#
# The frame counts are chosen for 8-GPU sequence parallel. Do not replace them
# with arbitrary 20s/25s/30s frame counts unless the temporal latent dimension
# still divides evenly by 8.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${EVAL_ROOT}/eval_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/miniconda/envs/giga_models/bin/python}"
VIDEOPHY_CHECKPOINT="${VIDEOPHY_CHECKPOINT:-/data/datasets}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
PA2_GPU_IDS="${PA2_GPU_IDS:-0}"
PA2_BATCH_SIZE="${PA2_BATCH_SIZE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
EXPECTED_COUNT="${EXPECTED_COUNT:-92}"
SWEEP_ID="${SWEEP_ID:-long_20_25_30}"
FORCE_RERUN="${FORCE_RERUN:-0}"

LENGTH_LABELS=("19p8s" "25p8s" "29p8s")
LENGTH_DISPLAY=("about 20s" "about 25s" "about 30s")
LENGTH_FRAMES=("317" "413" "477")

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

csv_row_count() {
  local csv_path="$1"
  if [[ ! -s "${csv_path}" ]]; then
    echo 0
    return
  fi
  python3 - "$csv_path" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

mp4_count() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

generation_done() {
  local summary_path="$1"
  [[ -f "${summary_path}" ]] || return 1
  local request_count generated_count
  request_count="$(summary_value "${summary_path}" request_count)"
  generated_count="$(summary_value "${summary_path}" generated_count)"
  [[ "${request_count}" == "${EXPECTED_COUNT}" && "${generated_count}" == "${request_count}" ]]
}

pa2_done() {
  local pa_csv="$1"
  [[ "$(csv_row_count "${pa_csv}")" == "${EXPECTED_COUNT}" ]]
}

check_generation_failure() {
  local log_path="$1"
  if [[ -f "${log_path}" ]] && rg -q "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}"; then
    echo "Generation appears to have failed. Check log: ${log_path}" >&2
    rg -n "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}" -S | tail -n 80 >&2
    exit 1
  fi
}

check_pa2_failure() {
  local log_path="$1"
  if [[ -f "${log_path}" ]] && rg -q "Traceback|Error|failed|Killed|CUDA out" "${log_path}"; then
    echo "PA-II appears to have failed. Check log: ${log_path}" >&2
    rg -n "Traceback|Error|failed|Killed|CUDA out" "${log_path}" -S | tail -n 80 >&2
    exit 1
  fi
}

wait_for_generation() {
  local run_name="$1"
  local save_dir="$2"
  local summary_path="${save_dir}/generation_summary.json"
  local log_path="${save_dir}/run.log"

  echo "Waiting for generation: ${run_name}"
  while true; do
    check_generation_failure "${log_path}"
    if generation_done "${summary_path}"; then
      echo "Generation complete: ${EXPECTED_COUNT}/${EXPECTED_COUNT}"
      break
    fi
    echo "  generated mp4 count so far: $(mp4_count "${save_dir}"); sleeping 60s"
    sleep 60
  done
}

wait_for_pa2() {
  local pa_run_name="$1"
  local pa_csv="${EVAL_OUTPUT_ROOT}/${pa_run_name}_pa_ii.csv"
  local log_path="${EVAL_OUTPUT_ROOT}/kjob_logs/${pa_run_name}.log"

  echo "Waiting for PA-II: ${pa_run_name}"
  while true; do
    check_pa2_failure "${log_path}"
    if pa2_done "${pa_csv}"; then
      echo "PA-II complete: ${pa_csv}"
      break
    fi
    echo "  PA-II rows so far: $(csv_row_count "${pa_csv}"); sleeping 30s"
    sleep 30
  done
}

prepare_generated_only() {
  local save_dir="$1"
  local video_dir="$2"
  if [[ "${FORCE_RERUN}" != "1" && "$(mp4_count "${video_dir}")" == "${EXPECTED_COUNT}" ]]; then
    echo "Generated-only videos already prepared: ${video_dir}"
    return
  fi
  echo "Preparing generated-only videos: ${video_dir}"
  SOURCE_VIDEO_DIR="${save_dir}" \
    "${PYTHON_BIN}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py" --overwrite
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
print("min_raw:", min(raw) if raw else "nan")
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

print_final_summary() {
  PYTHONPATH="" python3 - "${EVAL_ROOT}" "${OUTPUT_ROOT}" "${EVAL_OUTPUT_ROOT}" "${SWEEP_ID}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
eval_output_root = Path(sys.argv[3])
sweep_id = sys.argv[4]

items = [
    ("SFT", "about 20s", "317 / 16", "19.8125s", f"sft_dreamgen_8gpu_19p8s_full_{sweep_id}"),
    ("SFT", "about 25s", "413 / 16", "25.8125s", f"sft_dreamgen_8gpu_25p8s_full_{sweep_id}"),
    ("SFT", "about 30s", "477 / 16", "29.8125s", f"sft_dreamgen_8gpu_29p8s_full_{sweep_id}"),
    ("Pretrain", "about 20s", "317 / 16", "19.8125s", f"pretrain_dreamgen_8gpu_19p8s_full_{sweep_id}"),
    ("Pretrain", "about 25s", "413 / 16", "25.8125s", f"pretrain_dreamgen_8gpu_25p8s_full_{sweep_id}"),
    ("Pretrain", "about 30s", "477 / 16", "29.8125s", f"pretrain_dreamgen_8gpu_29p8s_full_{sweep_id}"),
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Final generation summary:")
print("| Model | Target | NUM_FRAMES/FPS | Actual | Generated | Wall time | Mean/sample | Median/sample | Generation GPU peak |")
print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for model, target, frames, actual, run in items:
    summary = load_json(output_root / run / "generation_summary.json")
    peak = load_json(output_root / run / "gpu_memory_peak.json")
    generated = f"{summary.get('generated_count')} / {summary.get('request_count')}"
    peak_text = f"{peak.get('peak_memory_used_mib')} MiB / {peak.get('peak_memory_used_gib')} GiB"
    print(
        f"| {model} | {target} | {frames} | {actual} | {generated} | "
        f"{summary.get('wall_time_sec'):.3f}s | {summary.get('model_elapsed_mean_sec'):.3f}s | "
        f"{summary.get('model_elapsed_median_sec'):.3f}s | {peak_text} |"
    )

print()
print("Final PA-II summary:")
print("| Model | Target | PA-II positive | PA-II score | Mean raw | Median raw | Min raw | Max raw | PA-II GPU peak |")
print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for model, target, _frames, _actual, run in items:
    pa_run = f"{run}_videophy_pa2"
    pa_csv = eval_output_root / f"{pa_run}_pa_ii.csv"
    rows = list(csv.DictReader(open(pa_csv, newline="", encoding="utf-8")))
    pred = [int(float(row["prediction"])) for row in rows]
    raw = [float(row["raw_score"]) for row in rows]
    peak = load_json(eval_output_root / "kjob_logs" / f"{pa_run}_gpu_memory_peak.json")
    peak_text = f"{peak.get('peak_memory_used_mib')} MiB / {peak.get('peak_memory_used_gib')} GiB"
    print(
        f"| {model} | {target} | {sum(pred)} / {len(pred)} | {sum(pred) / len(pred):.6f} | "
        f"{statistics.mean(raw):.6f} | {statistics.median(raw):.6f} | {min(raw):.6f} | {max(raw):.6f} | {peak_text} |"
    )
PY
}

run_one() {
  local model_key="$1"
  local model_display="$2"
  local checkpoint_dir="$3"
  local use_ema="$4"
  local label="$5"
  local display="$6"
  local frames="$7"

  local run_name="${model_key}_dreamgen_8gpu_${label}_full_${SWEEP_ID}"
  local save_dir="${OUTPUT_ROOT}/${run_name}"
  local summary_path="${save_dir}/generation_summary.json"
  local video_dir="${EVAL_ROOT}/dreamgenbench_video_dirs/${run_name}"
  local pa_run_name="${run_name}_videophy_pa2"
  local pa_csv="${EVAL_OUTPUT_ROOT}/${pa_run_name}_pa_ii.csv"

  echo "============================================================"
  echo "${model_display} ${display}: NUM_FRAMES=${frames}, FPS=${FPS}, actual=$(python3 - <<PY
print(round(${frames} / ${FPS}, 4))
PY
)s"
  echo "Generation run: ${run_name}"
  echo "============================================================"

  if [[ "${FORCE_RERUN}" != "1" && -f "${save_dir}/submit.log" && ! -f "${save_dir}/run.log" && ! -f "${summary_path}" ]]; then
    if rg -q "Error|failed|No such file" "${save_dir}/submit.log"; then
      echo "Generation submission appears to have failed. Check: ${save_dir}/submit.log" >&2
      tail -n 80 "${save_dir}/submit.log" >&2
      exit 1
    fi
  fi

  if [[ "${FORCE_RERUN}" != "1" ]] && generation_done "${summary_path}"; then
    echo "Generation already complete: ${run_name}"
  elif [[ "${FORCE_RERUN}" != "1" && -d "${save_dir}" ]] && [[ -f "${save_dir}/run.log" || -f "${save_dir}/submit.log" ]]; then
    echo "Found existing generation state; waiting instead of submitting duplicate: ${save_dir}"
    wait_for_generation "${run_name}" "${save_dir}"
  else
    CHECKPOINT_DIR="${checkpoint_dir}" \
    PRETRAIN_DIR="${PRETRAIN_DIR}" \
    USE_EMA="${use_ema}" \
    RUN_NAME="${run_name}" \
    SAVE_DIR="${save_dir}" \
    SUMMARY_PATH="${summary_path}" \
    GPU_IDS="${GPU_IDS}" \
    NUM_FRAMES="${frames}" \
    FPS="${FPS}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_dreamgen_generation_kjob.sh"
    wait_for_generation "${run_name}" "${save_dir}"
  fi
  summarize_peak_memory "Generation ${model_display} ${display}" "${save_dir}/gpu_memory_peak.json"

  prepare_generated_only "${save_dir}" "${video_dir}"

  if [[ "${FORCE_RERUN}" != "1" ]] && pa2_done "${pa_csv}"; then
    echo "PA-II already complete: ${pa_csv}"
  elif [[ "${FORCE_RERUN}" != "1" && -f "${EVAL_OUTPUT_ROOT}/kjob_logs/${pa_run_name}.log" ]]; then
    echo "Found existing PA-II log; waiting instead of submitting duplicate: ${pa_run_name}"
    wait_for_pa2 "${pa_run_name}"
  else
    echo "Submitting PA-II: ${pa_run_name}"
    VIDEO_DIR="${video_dir}" \
    RUN_NAME="${pa_run_name}" \
    CHECKPOINT="${VIDEOPHY_CHECKPOINT}" \
    GPU_IDS="${PA2_GPU_IDS}" \
    BATCH_SIZE="${PA2_BATCH_SIZE}" \
    "${EVEWORLD_ROOT}/benchmarks/pbench/launch_dreamgenbench_videophy_pa2_kjob.sh"
    wait_for_pa2 "${pa_run_name}"
  fi

  echo "PA-II summary for ${model_display} ${display}:"
  summarize_pa2 "${pa_csv}"
  summarize_peak_memory "PA-II ${model_display} ${display}" "${EVAL_OUTPUT_ROOT}/kjob_logs/${pa_run_name}_gpu_memory_peak.json"
  echo
}

cd "${REPO_DIR}"

echo "SFT + Pretrain DreamGen long-video PA-II sweep"
echo "Repo:              ${REPO_DIR}"
echo "SFT checkpoint:    ${SFT_CHECKPOINT_DIR}"
echo "Pretrain dir:      ${PRETRAIN_DIR}"
echo "EVAL_ROOT:         ${EVAL_ROOT}"
echo "SWEEP_ID:          ${SWEEP_ID}"
echo "GPU ids:           ${GPU_IDS}"
echo "PA-II GPU ids:     ${PA2_GPU_IDS}"
echo "PA-II batch size:  ${PA2_BATCH_SIZE}"
echo "Force rerun:       ${FORCE_RERUN}"
echo

for idx in "${!LENGTH_LABELS[@]}"; do
  run_one "sft" "SFT" "${SFT_CHECKPOINT_DIR}" "1" \
    "${LENGTH_LABELS[$idx]}" "${LENGTH_DISPLAY[$idx]}" "${LENGTH_FRAMES[$idx]}"
done

for idx in "${!LENGTH_LABELS[@]}"; do
  run_one "pretrain" "Pretrain" "${PRETRAIN_DIR}" "0" \
    "${LENGTH_LABELS[$idx]}" "${LENGTH_DISPLAY[$idx]}" "${LENGTH_FRAMES[$idx]}"
done

echo "All long-video SFT + Pretrain PA-II runs finished."
print_final_summary
