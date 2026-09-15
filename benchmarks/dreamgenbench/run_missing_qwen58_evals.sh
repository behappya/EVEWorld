#!/usr/bin/env bash
set -euo pipefail

# Backfill Qwen3.6-VL based evaluations through the 127.0.0.1 endpoint.
#
# This script is safe to rerun:
# - PBench: evaluates only generated-complete length runs, writing results to
#   *_domain_qwen36vl_58 directories.
# - DreamGen: evaluates only generated-complete runs and only metrics that do
#   not already have a complete no-error CSV, unless DREAMGEN_FORCE_QWEN58=1.
#
# It intentionally does not run GPT-IF or VideoPhy PA-II.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-auto}"
QWEN_TAG="${QWEN_TAG:-58}"

RUN_PBENCH="${RUN_PBENCH:-1}"
RUN_DREAMGEN="${RUN_DREAMGEN:-1}"
STATUS_ONLY="${STATUS_ONLY:-0}"

PBENCH_EXPECTED="${PBENCH_EXPECTED:-174}"
PBENCH_GEN_ROOT="${PBENCH_GEN_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_length_sweep}"
PBENCH_DOMAIN_ROOT="${PBENCH_DOMAIN_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval}"
PBENCH_SWEEP_ID="${PBENCH_SWEEP_ID:-pbench_len_sweep_full_8gpu}"
PBENCH_METADATA_JSONL="${PBENCH_METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"
PBENCH_CONCURRENCY="${PBENCH_CONCURRENCY:-100}"
PBENCH_MAX_INFLIGHT="${PBENCH_MAX_INFLIGHT:-100}"
PBENCH_FRAME_COUNT="${PBENCH_FRAME_COUNT:-8}"
PBENCH_MAX_IMAGE_SIDE="${PBENCH_MAX_IMAGE_SIDE:-512}"
PBENCH_MODEL_MAX_TOKENS="${PBENCH_MODEL_MAX_TOKENS:-256}"
PBENCH_SUMMARY_CSV="${PBENCH_SUMMARY_CSV:-${PBENCH_GEN_ROOT}/${PBENCH_SWEEP_ID}_qwen${QWEN_TAG}_domain_summary.csv}"
PBENCH_SUMMARY_JSON="${PBENCH_SUMMARY_JSON:-${PBENCH_GEN_ROOT}/${PBENCH_SWEEP_ID}_qwen${QWEN_TAG}_domain_summary.json}"

DREAMGEN_EXPECTED="${DREAMGEN_EXPECTED:-92}"
DREAMGEN_EVAL_ROOT="${DREAMGEN_EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
DREAMGEN_GENERATED_ROOT="${DREAMGEN_GENERATED_ROOT:-${DREAMGEN_EVAL_ROOT}/generated_side_by_side}"
DREAMGEN_VIDEO_ROOT="${DREAMGEN_VIDEO_ROOT:-${DREAMGEN_EVAL_ROOT}/dreamgenbench_video_dirs}"
DREAMGEN_OUTPUT_ROOT="${DREAMGEN_OUTPUT_ROOT:-${DREAMGEN_EVAL_ROOT}/eval_outputs}"
DREAMGEN_CONCURRENCY="${DREAMGEN_CONCURRENCY:-100}"
DREAMGEN_MAX_INFLIGHT="${DREAMGEN_MAX_INFLIGHT:-100}"
DREAMGEN_FRAME_COUNT="${DREAMGEN_FRAME_COUNT:-49}"
DREAMGEN_MAX_IMAGE_SIDE="${DREAMGEN_MAX_IMAGE_SIDE:-0}"
DREAMGEN_MODEL_MAX_TOKENS="${DREAMGEN_MODEL_MAX_TOKENS:-32000}"
DREAMGEN_FORCE_QWEN58="${DREAMGEN_FORCE_QWEN58:-0}"
DREAMGEN_SUMMARY_ID="${DREAMGEN_SUMMARY_ID:-dreamgen_task_completion_existing}"

PBENCH_TARGETS=(
  "3.8s|3p8s|61"
  "5.8s|5p8s|93"
  "9.8s|9p8s|157"
  "15.8s|15p8s|253"
  "19.8s|19p8s|317"
  "25.8s|25p8s|413"
  "29.8s|29p8s|477"
)

DREAMGEN_RUNS=(
  "SFT|5.8s|93 / 16|5.8125s|gr1_dreamgen_8gpu_full_20260625_212933"
  "SFT|9.8s|157 / 16|9.8125s|gr1_dreamgen_8gpu_9p8s_full_20260627_140442"
  "SFT|15.8s|253 / 16|15.8125s|gr1_dreamgen_8gpu_15p8s_full_20260627_145209"
  "SFT|19.8s|317 / 16|19.8125s|sft_dreamgen_8gpu_19p8s_full_long_20_25_30"
  "SFT|25.8s|413 / 16|25.8125s|sft_dreamgen_8gpu_25p8s_full_long_20_25_30"
  "SFT|29.8s|477 / 16|29.8125s|sft_dreamgen_8gpu_29p8s_full_long_20_25_30"
  "Pretrain|5.8s|93 / 16|5.8125s|pretrain_dreamgen_8gpu_5p8s_full_20260627_162849"
  "Pretrain|9.8s|157 / 16|9.8125s|pretrain_dreamgen_8gpu_9p8s_full_20260627_162849"
  "Pretrain|15.8s|253 / 16|15.8125s|pretrain_dreamgen_8gpu_15p8s_full_20260627_162849"
  "Pretrain|19.8s|317 / 16|19.8125s|pretrain_dreamgen_8gpu_19p8s_full_long_20_25_30"
  "Pretrain|25.8s|413 / 16|25.8125s|pretrain_dreamgen_8gpu_25p8s_full_long_20_25_30"
  "Pretrain|29.8s|477 / 16|29.8125s|pretrain_dreamgen_8gpu_29p8s_full_long_20_25_30"
)

json_get() {
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

generation_done() {
  local summary_path="$1"
  local expected="$2"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "$summary_path" "$expected" <<'PY'
import json
import sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("generated_count") == expected and data.get("request_count") == expected
raise SystemExit(0 if ok else 1)
PY
}

mp4_count() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

pbench_domain_valid() {
  local eval_dir="$1"
  local expected="$2"
  local summary_path="${eval_dir}/qwen_vqa_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "$summary_path" "$expected" <<'PY'
import json
import sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("sample_count") == expected and int(data.get("error_count") or 0) == 0
raise SystemExit(0 if ok else 1)
PY
}

jsonl_has_errors() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  python3 - "$path" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            raise SystemExit(0)
        if row.get("error"):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

csv_has_errors() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  python3 - "$path" <<'PY'
import csv
import sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("error"):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

archive_pbench_error_eval_if_needed() {
  local eval_dir="$1"
  local result_jsonl="${eval_dir}/qwen_vqa_results.jsonl"
  if jsonl_has_errors "${result_jsonl}"; then
    local archived="${eval_dir}_bad_$(date +%Y%m%d_%H%M%S)"
    echo "Archive previous errored PBench eval: ${eval_dir} -> ${archived}"
    mv "${eval_dir}" "${archived}"
  fi
}

archive_dreamgen_error_csv_if_needed() {
  local csv_path="$1"
  if csv_has_errors "${csv_path}"; then
    local archived="${csv_path%.csv}_bad_$(date +%Y%m%d_%H%M%S).csv"
    echo "Archive previous errored DreamGen CSV: ${csv_path} -> ${archived}"
    mv "${csv_path}" "${archived}"
  fi
}

dreamgen_metric_valid() {
  local run_name="$1"
  local suffix="$2"
  local expected="$3"
  local require_qwen_tag="$4"
  python3 - "${DREAMGEN_OUTPUT_ROOT}" "${run_name}" "${suffix}" "${expected}" "${require_qwen_tag}" "${QWEN_TAG}" <<'PY'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
run_name, suffix = sys.argv[2], sys.argv[3]
expected = int(sys.argv[4])
require_tag = sys.argv[5] == "1"
tag = sys.argv[6]
paths = sorted(root.glob(f"{run_name}*_{suffix}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
for path in paths:
    if require_tag and f"qwen{tag}" not in path.name:
        continue
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        continue
    if len(rows) == expected and not any(row.get("error") for row in rows):
        print(path)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

write_pbench_qwen_summary() {
  python3 - "${PBENCH_GEN_ROOT}" "${PBENCH_DOMAIN_ROOT}" "${PBENCH_SWEEP_ID}" "${QWEN_TAG}" "${PBENCH_SUMMARY_CSV}" "${PBENCH_SUMMARY_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

gen_root, domain_root, sweep_id, qwen_tag, summary_csv, summary_json = sys.argv[1:7]
gen_root = Path(gen_root)
domain_root = Path(domain_root)
summary_csv = Path(summary_csv)
summary_json = Path(summary_json)
labels = [("3.8s", "3p8s", 61), ("5.8s", "5p8s", 93), ("9.8s", "9p8s", 157), ("15.8s", "15p8s", 253), ("19.8s", "19p8s", 317), ("25.8s", "25p8s", 413), ("29.8s", "29p8s", 477)]

def load(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

rows = []
for target, label, frames in labels:
    run_name = f"pbench_robot_{label}_full_{sweep_id}"
    gen_dir = gen_root / run_name
    domain_dir = domain_root / f"{run_name}_domain_qwen36vl_{qwen_tag}"
    gen = load(gen_dir / "generation_summary.json")
    peak = load(gen_dir / "gpu_memory_peak.json")
    domain = load(domain_dir / "qwen_vqa_summary.json")
    rows.append({
        "target": target,
        "num_frames": frames,
        "fps": 16,
        "actual_sec": frames / 16.0,
        "run_name": run_name,
        "generated_count": gen.get("generated_count"),
        "request_count": gen.get("request_count"),
        "generation_wall_time_sec": gen.get("wall_time_sec"),
        "generation_mean_sec": gen.get("model_elapsed_mean_sec"),
        "generation_gpu_peak_mib": peak.get("peak_memory_used_mib"),
        "generation_gpu_peak_gib": peak.get("peak_memory_used_gib"),
        "domain_eval_dir": str(domain_dir),
        "domain_sample_count": domain.get("sample_count"),
        "domain_error_count": domain.get("error_count"),
        "domain_score_like": domain.get("domain_score_like"),
        "domain_question_micro_accuracy": domain.get("question_micro_accuracy"),
        "domain_sample_macro_accuracy": domain.get("sample_macro_accuracy"),
    })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
summary_json.write_text(json.dumps({"rows": rows, "summary_csv": str(summary_csv)}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"PBench Qwen58 summary CSV:  {summary_csv}")
print(f"PBench Qwen58 summary JSON: {summary_json}")
PY
}

run_pbench_backfill() {
  [[ "${RUN_PBENCH}" == "1" ]] || return 0
  echo
  echo "================ PBench Qwen${QWEN_TAG} backfill ================"
  for item in "${PBENCH_TARGETS[@]}"; do
    IFS='|' read -r target label frames <<<"${item}"
    local run_name="pbench_robot_${label}_full_${PBENCH_SWEEP_ID}"
    local video_dir="${PBENCH_GEN_ROOT}/${run_name}"
    local summary_path="${video_dir}/generation_summary.json"
    local eval_dir="${PBENCH_DOMAIN_ROOT}/${run_name}_domain_qwen36vl_${QWEN_TAG}"

    echo
    echo "PBench ${target}: ${run_name}"
    if ! generation_done "${summary_path}" "${PBENCH_EXPECTED}"; then
      echo "Skip: generation not complete. mp4=$(mp4_count "${video_dir}")/${PBENCH_EXPECTED}"
      continue
    fi
    if pbench_domain_valid "${eval_dir}" "${PBENCH_EXPECTED}"; then
      echo "Skip: valid Qwen${QWEN_TAG} domain already exists: ${eval_dir}"
      continue
    fi
    if [[ "${STATUS_ONLY}" == "1" ]]; then
      echo "Would run PBench Domain Qwen${QWEN_TAG}: ${eval_dir}"
      continue
    fi

    archive_pbench_error_eval_if_needed "${eval_dir}"

    VIDEO_DIR="${video_dir}" \
    EVAL_DIR="${eval_dir}" \
    METADATA_JSONL="${PBENCH_METADATA_JSONL}" \
    QWEN_BASE="${QWEN_BASE}" \
    QWEN_MODEL="${QWEN_MODEL}" \
    LIMIT=0 \
    CONCURRENCY="${PBENCH_CONCURRENCY}" \
    MAX_INFLIGHT="${PBENCH_MAX_INFLIGHT}" \
    FRAME_COUNT="${PBENCH_FRAME_COUNT}" \
    MAX_IMAGE_SIDE="${PBENCH_MAX_IMAGE_SIDE}" \
    MODEL_MAX_TOKENS="${PBENCH_MODEL_MAX_TOKENS}" \
    RERUN_ERRORS=1 \
    bash "${EVEWORLD_ROOT}/benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh"

    if ! pbench_domain_valid "${eval_dir}" "${PBENCH_EXPECTED}"; then
      echo "PBench Domain still has errors or incomplete rows: ${eval_dir}" >&2
      exit 1
    fi
  done
  write_pbench_qwen_summary
}

prepare_dreamgen_videos() {
  local run_name="$1"
  local video_dir="${DREAMGEN_VIDEO_ROOT}/${run_name}"
  if [[ "$(mp4_count "${video_dir}")" == "${DREAMGEN_EXPECTED}" ]]; then
    return
  fi
  SOURCE_VIDEO_DIR="${DREAMGEN_GENERATED_ROOT}/${run_name}" \
    python3 "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py" --overwrite
}

run_dreamgen_backfill() {
  [[ "${RUN_DREAMGEN}" == "1" ]] || return 0
  echo
  echo "================ DreamGen Qwen${QWEN_TAG} backfill ================"
  for item in "${DREAMGEN_RUNS[@]}"; do
    IFS='|' read -r model target frames actual run_name <<<"${item}"
    local gen_summary="${DREAMGEN_GENERATED_ROOT}/${run_name}/generation_summary.json"
    local video_dir="${DREAMGEN_VIDEO_ROOT}/${run_name}"
    local metrics=()

    echo
    echo "DreamGen ${model} ${target}: ${run_name}"
    if ! generation_done "${gen_summary}" "${DREAMGEN_EXPECTED}"; then
      echo "Skip: generation not complete."
      continue
    fi
    prepare_dreamgen_videos "${run_name}"
    if [[ "$(mp4_count "${video_dir}")" != "${DREAMGEN_EXPECTED}" ]]; then
      echo "Skip: generated-only video count is $(mp4_count "${video_dir}")/${DREAMGEN_EXPECTED}."
      continue
    fi

    if ! dreamgen_metric_valid "${run_name}" qwen_if "${DREAMGEN_EXPECTED}" "${DREAMGEN_FORCE_QWEN58}" >/dev/null; then
      metrics+=("qwen_if")
    fi
    if ! dreamgen_metric_valid "${run_name}" pa_i "${DREAMGEN_EXPECTED}" "${DREAMGEN_FORCE_QWEN58}" >/dev/null; then
      metrics+=("pa_i")
    fi
    if [[ "${#metrics[@]}" -eq 0 ]]; then
      echo "Skip: Qwen-IF and PA-I already have complete no-error CSVs."
      continue
    fi

    local metrics_csv
    metrics_csv="$(IFS=,; echo "${metrics[*]}")"
    local metrics_tag="${metrics_csv//,/_}"
    local eval_run_name="${run_name}_qwen${QWEN_TAG}_missing_${metrics_tag}_c${DREAMGEN_CONCURRENCY}"

    if [[ "${STATUS_ONLY}" == "1" ]]; then
      echo "Would run DreamGen Qwen${QWEN_TAG}: ${eval_run_name}, metrics=${metrics_csv}"
      continue
    fi

    for metric in "${metrics[@]}"; do
      archive_dreamgen_error_csv_if_needed "${DREAMGEN_OUTPUT_ROOT}/${eval_run_name}_${metric}.csv"
    done

    VIDEO_DIR="${video_dir}" \
    OUTPUT_ROOT="${DREAMGEN_OUTPUT_ROOT}" \
    RUN_NAME="${eval_run_name}" \
    QWEN_BASE="${QWEN_BASE}" \
    QWEN_MODEL="${QWEN_MODEL}" \
    METRICS="${metrics_csv}" \
    CONCURRENCY="${DREAMGEN_CONCURRENCY}" \
    MAX_INFLIGHT="${DREAMGEN_MAX_INFLIGHT}" \
    FRAME_COUNT="${DREAMGEN_FRAME_COUNT}" \
    MAX_IMAGE_SIDE="${DREAMGEN_MAX_IMAGE_SIDE}" \
    MODEL_MAX_TOKENS="${DREAMGEN_MODEL_MAX_TOKENS}" \
    RERUN_ERRORS=1 \
    bash "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.sh"

    for metric in "${metrics[@]}"; do
      if ! dreamgen_metric_valid "${run_name}" "${metric}" "${DREAMGEN_EXPECTED}" "${DREAMGEN_FORCE_QWEN58}" >/dev/null; then
        echo "DreamGen ${metric} still has errors or incomplete rows: ${run_name}" >&2
        exit 1
      fi
    done
  done

  STATUS_ONLY=1 \
  RUN_QWEN_IF=0 \
  RUN_PA_I=0 \
  RUN_GPT_IF=0 \
  EVAL_ID="${DREAMGEN_SUMMARY_ID}" \
  QWEN_BASE="${QWEN_BASE}" \
  bash "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/run_completed_dreamgen_task_completion_eval.sh"
}

cd "${REPO_DIR}"

echo "Qwen backfill"
echo "Repo:              ${REPO_DIR}"
echo "Qwen base:         ${QWEN_BASE}"
echo "Qwen model:        ${QWEN_MODEL}"
echo "Qwen tag:          ${QWEN_TAG}"
echo "Run PBench:        ${RUN_PBENCH}"
echo "Run DreamGen:      ${RUN_DREAMGEN}"
echo "Status only:       ${STATUS_ONLY}"
echo "DreamGen force 58: ${DREAMGEN_FORCE_QWEN58}"

run_pbench_backfill
run_dreamgen_backfill

echo
echo "Done."
