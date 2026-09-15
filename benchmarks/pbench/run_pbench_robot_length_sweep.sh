#!/usr/bin/env bash
set -euo pipefail

# Resumable PBench Robot length sweep.
#
# Default mode is smoke:
#   - DATA_LIMIT=1
#   - NUM_INFERENCE_STEPS=30 by default, so smoke tests the real denoising cost
#   - 8 GPUs by default, so GPU memory is comparable with DreamGen length runs
#   - runs Domain/VQA by default
#   - Quality/VBench is optional because it is heavier
#
# Full mode:
#   MODE=full DATA_LIMIT=0 ./benchmarks/pbench/run_pbench_robot_length_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${MODE:-smoke}"
SWEEP_ID="${SWEEP_ID:-pbench_len_sweep_${MODE}}"
DATA_LIMIT="${DATA_LIMIT:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE:-1}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"
SEED="${SEED:-6666}"
if [[ -z "${EXPECTED_COUNT:-}" ]]; then
  if [[ "${DATA_LIMIT}" == "0" ]]; then
    EXPECTED_COUNT="174"
  else
    EXPECTED_COUNT="${DATA_LIMIT}"
  fi
fi

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/giga_world_0_outputs}"
GEN_OUTPUT_ROOT="${GEN_OUTPUT_ROOT:-${EVAL_ROOT}/pbench_robot_length_sweep}"
DOMAIN_EVAL_ROOT="${DOMAIN_EVAL_ROOT:-${EVAL_ROOT}/pbench_robot_qwen_vqa_eval}"
QUALITY_OUTPUT_ROOT_BASE="${QUALITY_OUTPUT_ROOT_BASE:-${EVAL_ROOT}/pbench_robot_vbench_quality_length_sweep}"
DATA_PATH="${DATA_PATH:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"

RUN_DOMAIN="${RUN_DOMAIN:-1}"
RUN_QUALITY="${RUN_QUALITY:-0}"
DOMAIN_QWEN_BASE="${DOMAIN_QWEN_BASE:-http://127.0.0.1:8000/v1}"
DOMAIN_QWEN_TAG="${DOMAIN_QWEN_TAG:-58}"
DOMAIN_QWEN_MODEL="${DOMAIN_QWEN_MODEL:-auto}"
DOMAIN_CONCURRENCY="${DOMAIN_CONCURRENCY:-100}"
DOMAIN_MAX_INFLIGHT="${DOMAIN_MAX_INFLIGHT:-100}"
DOMAIN_FRAME_COUNT="${DOMAIN_FRAME_COUNT:-8}"
DOMAIN_MAX_IMAGE_SIDE="${DOMAIN_MAX_IMAGE_SIDE:-512}"
DOMAIN_MODEL_MAX_TOKENS="${DOMAIN_MODEL_MAX_TOKENS:-256}"

QUALITY_GPU_IDS="${QUALITY_GPU_IDS:-0}"
QUALITY_LIMIT="${QUALITY_LIMIT:-${DATA_LIMIT}}"
QUALITY_DIMENSIONS="${QUALITY_DIMENSIONS:-i2v_subject i2v_background aesthetic_quality imaging_quality background_consistency motion_smoothness subject_consistency overall_consistency}"

SUMMARY_CSV="${GEN_OUTPUT_ROOT}/${SWEEP_ID}_summary.csv"
SUMMARY_JSON="${GEN_OUTPUT_ROOT}/${SWEEP_ID}_summary.json"

LENGTH_LABELS=("3p8s" "5p8s" "9p8s" "15p8s" "19p8s" "25p8s" "29p8s")
LENGTH_FRAMES=("61" "93" "157" "253" "317" "413" "477")

mkdir -p "${GEN_OUTPUT_ROOT}"

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

json_value() {
  local path="$1"
  local key="$2"
  python3 - "$path" "$key" <<'PY'
import json
import sys
path, key = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get(key, ""))
PY
}

mp4_count() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name 'robot_*.mp4' 2>/dev/null | wc -l
}

generation_done() {
  local save_dir="$1"
  local summary_path="${save_dir}/generation_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  local generated request_count
  generated="$(json_value "${summary_path}" generated_count)"
  request_count="$(json_value "${summary_path}" request_count)"
  [[ "${generated}" == "${EXPECTED_COUNT}" && "${request_count}" == "${EXPECTED_COUNT}" ]]
}

wait_for_generation() {
  local run_name="$1"
  local save_dir="$2"
  local log_path="${save_dir}/run.log"
  echo "Waiting for generation: ${run_name}"
  while true; do
    if [[ -f "${log_path}" ]] && rg -q "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}"; then
      echo "Generation appears to have failed. Check log: ${log_path}" >&2
      rg -n "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}" -S | tail -n 80 >&2
      exit 1
    fi
    if generation_done "${save_dir}"; then
      echo "Generation complete: ${EXPECTED_COUNT}/${EXPECTED_COUNT}"
      break
    fi
    echo "  robot mp4 count so far: $(mp4_count "${save_dir}"); sleeping 60s"
    sleep 60
  done
}

domain_done() {
  local eval_dir="$1"
  local summary_path="${eval_dir}/qwen_vqa_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  local sample_count
  sample_count="$(json_value "${summary_path}" sample_count)"
  [[ "${sample_count}" == "${EXPECTED_COUNT}" ]]
}

run_domain_if_needed() {
  local run_name="$1"
  local save_dir="$2"
  local eval_dir="${DOMAIN_EVAL_ROOT}/${run_name}_domain_qwen36vl_${DOMAIN_QWEN_TAG}"
  if [[ "${RUN_DOMAIN}" != "1" ]]; then
    return
  fi
  if domain_done "${eval_dir}"; then
    echo "Domain already complete: ${eval_dir}"
    return
  fi
  echo "Running PBench Domain/VQA: ${run_name}"
  VIDEO_DIR="${save_dir}" \
  EVAL_DIR="${eval_dir}" \
  METADATA_JSONL="${METADATA_JSONL}" \
  QWEN_BASE="${DOMAIN_QWEN_BASE}" \
  QWEN_MODEL="${DOMAIN_QWEN_MODEL}" \
  LIMIT="${DATA_LIMIT}" \
  CONCURRENCY="${DOMAIN_CONCURRENCY}" \
  MAX_INFLIGHT="${DOMAIN_MAX_INFLIGHT}" \
  FRAME_COUNT="${DOMAIN_FRAME_COUNT}" \
  MAX_IMAGE_SIDE="${DOMAIN_MAX_IMAGE_SIDE}" \
  MODEL_MAX_TOKENS="${DOMAIN_MODEL_MAX_TOKENS}" \
  bash "${EVEWORLD_ROOT}/benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh"
}

quality_done() {
  local output_root="$1"
  [[ -f "${output_root}/quality_eval/pbench_robot_quality_overall_summary.json" ]]
}

wait_for_quality() {
  local run_name="$1"
  local output_root="$2"
  local run_log="${output_root}/kjob_logs/${run_name}_quality.log"
  echo "Waiting for Quality/VBench: ${run_name}"
  while true; do
    if [[ -f "${run_log}" ]] && rg -q "Traceback|RuntimeError|CUDA out|Killed|Error" "${run_log}"; then
      echo "Quality appears to have failed. Check log: ${run_log}" >&2
      rg -n "Traceback|RuntimeError|CUDA out|Killed|Error" "${run_log}" -S | tail -n 80 >&2
      exit 1
    fi
    if quality_done "${output_root}"; then
      echo "Quality complete: ${output_root}/quality_eval/pbench_robot_quality_overall_summary.json"
      break
    fi
    echo "  quality summary not ready; sleeping 60s"
    sleep 60
  done
}

run_quality_if_needed() {
  local run_name="$1"
  local save_dir="$2"
  local domain_summary="${DOMAIN_EVAL_ROOT}/${run_name}_domain_qwen36vl_${DOMAIN_QWEN_TAG}/qwen_vqa_summary.json"
  local output_root="${QUALITY_OUTPUT_ROOT_BASE}/${run_name}"
  local quality_run_name="${run_name}_quality"
  if [[ "${RUN_QUALITY}" != "1" ]]; then
    return
  fi
  if [[ ! -f "${domain_summary}" ]]; then
    echo "Skip Quality: missing domain summary ${domain_summary}"
    return
  fi
  if quality_done "${output_root}"; then
    echo "Quality already complete: ${output_root}"
    return
  fi
  echo "Submitting PBench Quality/VBench: ${run_name}"
  OUTPUT_ROOT="${output_root}" \
  RUN_NAME="${quality_run_name}" \
  SOURCE_VIDEO_DIR="${save_dir}" \
  DOMAIN_SUMMARY="${domain_summary}" \
  LIMIT="${QUALITY_LIMIT}" \
  DIMENSIONS="${QUALITY_DIMENSIONS}" \
  GPU_IDS="${QUALITY_GPU_IDS}" \
  RUN_LOG="${output_root}/kjob_logs/${quality_run_name}.log" \
  GPU_MEMORY_SAMPLES="${output_root}/kjob_logs/${quality_run_name}_gpu_memory_samples.csv" \
  GPU_MEMORY_PEAK="${output_root}/kjob_logs/${quality_run_name}_gpu_memory_peak.json" \
  "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_vbench_quality_kjob.sh"
  wait_for_quality "${run_name}" "${output_root}"
}

summarize_peak() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "-"
    return
  fi
  python3 - "$path" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
if data.get("peak_memory_used_mib") is None:
    print("-")
else:
    print(f"{data.get('peak_memory_used_mib')} MiB / {data.get('peak_memory_used_gib')} GiB")
PY
}

write_summary() {
  python3 - "${GEN_OUTPUT_ROOT}" "${DOMAIN_EVAL_ROOT}" "${QUALITY_OUTPUT_ROOT_BASE}" "${SWEEP_ID}" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${DOMAIN_QWEN_TAG}" <<'PY'
import csv
import json
import sys
from pathlib import Path

gen_root, domain_root, quality_root, sweep_id, summary_csv, summary_json, domain_qwen_tag = sys.argv[1:8]
gen_root = Path(gen_root)
domain_root = Path(domain_root)
quality_root = Path(quality_root)
summary_csv = Path(summary_csv)
summary_json = Path(summary_json)

labels = [("3.8s", "61"), ("5.8s", "93"), ("9.8s", "157"), ("15.8s", "253"), ("19.8s", "317"), ("25.8s", "413"), ("29.8s", "477")]

def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def peak(path):
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    return data

rows = []
for target, frames in labels:
    label = target.replace(".", "p").replace("s", "s")
    run_label = label
    # target labels are 3.8s, 5.8s, ...; run labels are 3p8s, 5p8s, ...
    run_label = target.replace(".", "p")
    run_name = f"pbench_robot_{run_label}_full_{sweep_id}"
    save_dir = gen_root / run_name
    domain_dir = domain_root / f"{run_name}_domain_qwen36vl_{domain_qwen_tag}"
    quality_dir = quality_root / run_name
    gen = load_json(save_dir / "generation_summary.json") or {}
    gen_peak = peak(save_dir / "gpu_memory_peak.json")
    domain = load_json(domain_dir / "qwen_vqa_summary.json") or {}
    quality = load_json(quality_dir / "quality_eval" / "pbench_robot_quality_overall_summary.json") or {}
    quality_peak = peak(quality_dir / "kjob_logs" / f"{run_name}_quality_gpu_memory_peak.json")
    rows.append({
        "target": target,
        "num_frames": frames,
        "fps": 16,
        "actual_sec": float(frames) / 16.0,
        "run_name": run_name,
        "generated_count": gen.get("generated_count"),
        "request_count": gen.get("request_count"),
        "generation_wall_time_sec": gen.get("wall_time_sec"),
        "generation_mean_sec": gen.get("model_elapsed_mean_sec"),
        "generation_median_sec": gen.get("model_elapsed_median_sec"),
        "generation_gpu_peak_mib": gen_peak.get("peak_memory_used_mib"),
        "generation_gpu_peak_gib": gen_peak.get("peak_memory_used_gib"),
        "generation_gpu_peak_name": gen_peak.get("peak_gpu_name"),
        "domain_score_like": domain.get("domain_score_like"),
        "domain_question_micro_accuracy": domain.get("question_micro_accuracy"),
        "domain_sample_macro_accuracy": domain.get("sample_macro_accuracy"),
        "domain_sample_count": domain.get("sample_count"),
        "quality_score": quality.get("quality_score"),
        "overall_score_like": quality.get("overall_score_like"),
        "quality_gpu_peak_mib": quality_peak.get("peak_memory_used_mib"),
        "quality_gpu_peak_gib": quality_peak.get("peak_memory_used_gib"),
        "quality_gpu_peak_name": quality_peak.get("peak_gpu_name"),
    })

fields = list(rows[0].keys())
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
summary_json.write_text(json.dumps({"rows": rows, "summary_csv": str(summary_csv)}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Summary CSV:  {summary_csv}")
print(f"Summary JSON: {summary_json}")
print("| Target | Run | Generated | Gen GPU peak | Domain | Quality | Overall | Quality GPU peak |")
print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for row in rows:
    gen_text = "-"
    if row["generated_count"] is not None:
        gen_text = f"{row['generated_count']} / {row['request_count']}"
    gen_peak_text = "-"
    if row["generation_gpu_peak_mib"] is not None:
        gen_peak_text = f"{row['generation_gpu_peak_mib']} MiB / {row['generation_gpu_peak_gib']} GiB"
    quality_peak_text = "-"
    if row["quality_gpu_peak_mib"] is not None:
        quality_peak_text = f"{row['quality_gpu_peak_mib']} MiB / {row['quality_gpu_peak_gib']} GiB"
    def fmt(value):
        return "-" if value is None else f"{float(value):.6f}"
    print(f"| {row['target']} | `{row['run_name']}` | {gen_text} | {gen_peak_text} | {fmt(row['domain_score_like'])} | {fmt(row['quality_score'])} | {fmt(row['overall_score_like'])} | {quality_peak_text} |")
PY
}

cd "${REPO_DIR}"

echo "PBench Robot length sweep"
echo "Mode:                  ${MODE}"
echo "Sweep id:              ${SWEEP_ID}"
echo "Data limit:            ${DATA_LIMIT}"
echo "Expected count:        ${EXPECTED_COUNT}"
echo "Steps:                 ${NUM_INFERENCE_STEPS}"
echo "GPU ids:               ${GPU_IDS}"
echo "Run Domain:            ${RUN_DOMAIN}"
echo "Run Quality:           ${RUN_QUALITY}"
echo "Generation output:     ${GEN_OUTPUT_ROOT}"
echo

for idx in "${!LENGTH_LABELS[@]}"; do
  label="${LENGTH_LABELS[$idx]}"
  frames="${LENGTH_FRAMES[$idx]}"
  run_name="pbench_robot_${label}_full_${SWEEP_ID}"
  save_dir="${GEN_OUTPUT_ROOT}/${run_name}"
  summary_path="${save_dir}/generation_summary.json"

  echo "============================================================"
  echo "PBench ${label}: NUM_FRAMES=${frames}, FPS=${FPS}, actual=$(python3 - <<PY
print(round(${frames} / ${FPS}, 4))
PY
)s"
  echo "Run: ${run_name}"
  echo "============================================================"

  if generation_done "${save_dir}"; then
    echo "Generation already complete: ${run_name}"
  elif [[ -d "${save_dir}" ]] && [[ -f "${save_dir}/run.log" || -f "${save_dir}/submit.log" ]]; then
    echo "Found existing generation state; waiting: ${save_dir}"
    wait_for_generation "${run_name}" "${save_dir}"
  else
    MODE="${MODE}" \
    RUN_NAME="${run_name}" \
    OUTPUT_ROOT="${GEN_OUTPUT_ROOT}" \
    SAVE_DIR="${save_dir}" \
    DATA_PATH="${DATA_PATH}" \
    DATA_LIMIT="${DATA_LIMIT}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    GPU_IDS="${GPU_IDS}" \
    ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE}" \
    NUM_FRAMES="${frames}" \
    FPS="${FPS}" \
    HEIGHT="${HEIGHT}" \
    WIDTH="${WIDTH}" \
    SEED="${SEED}" \
    GPU_MEMORY_SAMPLES="${save_dir}/gpu_memory_samples.csv" \
    GPU_MEMORY_PEAK="${save_dir}/gpu_memory_peak.json" \
    "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_kjob.sh"
    wait_for_generation "${run_name}" "${save_dir}"
  fi

  echo "Generation GPU peak: $(summarize_peak "${save_dir}/gpu_memory_peak.json")"
  run_domain_if_needed "${run_name}" "${save_dir}"
  run_quality_if_needed "${run_name}" "${save_dir}"
done

write_summary
