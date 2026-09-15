#!/usr/bin/env bash
set -euo pipefail

# Evaluate task completion for DreamGen runs that are already fully generated.
#
# Default benchmark:
#   - Qwen-IF: asks whether the generated robot video follows the instruction.
#
# Optional:
#   - GPT-IF: also measures instruction following, but is slower/costlier.
#   - PA-I: Qwen physical alignment; useful context, not the primary task
#     completion metric.
#
# The script is resumable: existing complete CSVs with 92 rows are reused.
# Incomplete generation runs are skipped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/eval_outputs}"
GENERATED_ROOT="${GENERATED_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
VIDEO_ROOT="${VIDEO_ROOT:-${EVAL_ROOT}/dreamgenbench_video_dirs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/miniconda/envs/giga_models/bin/python}"
EXPECTED_COUNT="${EXPECTED_COUNT:-92}"

EVAL_ID="${EVAL_ID:-dreamgen_task_completion_existing}"
MANIFEST_PATH="${OUTPUT_ROOT}/${EVAL_ID}_manifest.tsv"
SUMMARY_CSV="${OUTPUT_ROOT}/${EVAL_ID}_summary.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/${EVAL_ID}_summary.json"

RUN_QWEN_IF="${RUN_QWEN_IF:-1}"
RUN_GPT_IF="${RUN_GPT_IF:-0}"
RUN_PA_I="${RUN_PA_I:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-auto}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-100}"
QWEN_MAX_INFLIGHT="${QWEN_MAX_INFLIGHT:-100}"
QWEN_FRAME_COUNT="${QWEN_FRAME_COUNT:-49}"
QWEN_MAX_IMAGE_SIDE="${QWEN_MAX_IMAGE_SIDE:-0}"
QWEN_MODEL_MAX_TOKENS="${QWEN_MODEL_MAX_TOKENS:-32000}"

DIFROST_MODEL="${DIFROST_MODEL:-gpt-5.5}"
GPT_CONCURRENCY="${GPT_CONCURRENCY:-10}"
GPT_FRAME_COUNT="${GPT_FRAME_COUNT:-8}"
GPT_SCALE_FACTOR="${GPT_SCALE_FACTOR:-0.5}"
GPT_MODEL_MAX_TOKENS="${GPT_MODEL_MAX_TOKENS:-32000}"

mkdir -p "${OUTPUT_ROOT}"

write_manifest() {
  cat > "${MANIFEST_PATH}" <<'EOF'
model	target	frames	actual	run_name
SFT	5.8s	93 / 16	5.8125s	gr1_dreamgen_8gpu_full_20260625_212933
SFT	9.8s	157 / 16	9.8125s	gr1_dreamgen_8gpu_9p8s_full_20260627_140442
SFT	15.8s	253 / 16	15.8125s	gr1_dreamgen_8gpu_15p8s_full_20260627_145209
SFT	19.8s	317 / 16	19.8125s	sft_dreamgen_8gpu_19p8s_full_long_20_25_30
SFT	25.8s	413 / 16	25.8125s	sft_dreamgen_8gpu_25p8s_full_long_20_25_30
SFT	29.8s	477 / 16	29.8125s	sft_dreamgen_8gpu_29p8s_full_long_20_25_30
Pretrain	5.8s	93 / 16	5.8125s	pretrain_dreamgen_8gpu_5p8s_full_20260627_162849
Pretrain	9.8s	157 / 16	9.8125s	pretrain_dreamgen_8gpu_9p8s_full_20260627_162849
Pretrain	15.8s	253 / 16	15.8125s	pretrain_dreamgen_8gpu_15p8s_full_20260627_162849
Pretrain	19.8s	317 / 16	19.8125s	pretrain_dreamgen_8gpu_19p8s_full_long_20_25_30
Pretrain	25.8s	413 / 16	25.8125s	pretrain_dreamgen_8gpu_25p8s_full_long_20_25_30
Pretrain	29.8s	477 / 16	29.8125s	pretrain_dreamgen_8gpu_29p8s_full_long_20_25_30
EOF
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
  local run_name="$1"
  local summary_path="${GENERATED_ROOT}/${run_name}/generation_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "${summary_path}" "${EXPECTED_COUNT}" <<'PY'
import json
import sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("request_count") == expected and data.get("generated_count") == expected
raise SystemExit(0 if ok else 1)
PY
}

find_complete_csv() {
  local run_name="$1"
  local suffix="$2"
  python3 - "${OUTPUT_ROOT}" "${run_name}" "${suffix}" "${EXPECTED_COUNT}" <<'PY'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
run = sys.argv[2]
suffix = sys.argv[3]
expected = int(sys.argv[4])
paths = sorted(root.glob(f"{run}*_{suffix}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
for path in paths:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = sum(1 for _ in csv.DictReader(f))
    except Exception:
        continue
    if rows == expected:
        print(path)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

ensure_generated_only() {
  local run_name="$1"
  local source_dir="${GENERATED_ROOT}/${run_name}"
  local video_dir="${VIDEO_ROOT}/${run_name}"
  if [[ "$(mp4_count "${video_dir}")" == "${EXPECTED_COUNT}" ]]; then
    return
  fi
  echo "Preparing generated-only videos: ${run_name}"
  SOURCE_VIDEO_DIR="${source_dir}" \
    "${PYTHON_BIN}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py" --overwrite
}

run_qwen_eval_if_needed() {
  local run_name="$1"
  local video_dir="${VIDEO_ROOT}/${run_name}"
  local metrics=()

  if [[ "${RUN_QWEN_IF}" == "1" ]] && ! find_complete_csv "${run_name}" qwen_if >/dev/null; then
    metrics+=("qwen_if")
  fi
  if [[ "${RUN_PA_I}" == "1" ]] && ! find_complete_csv "${run_name}" pa_i >/dev/null; then
    metrics+=("pa_i")
  fi
  if [[ "${#metrics[@]}" -eq 0 ]]; then
    return
  fi

  local metrics_csv
  metrics_csv="$(IFS=,; echo "${metrics[*]}")"
  local run_suffix="${metrics_csv//,/_}"
  local qwen_run_name="${run_name}_qwen36vl_task_${run_suffix}_c${QWEN_CONCURRENCY}"

  echo "Running Qwen API eval: ${qwen_run_name} metrics=${metrics_csv}"
  VIDEO_DIR="${video_dir}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_NAME="${qwen_run_name}" \
  QWEN_BASE="${QWEN_BASE}" \
  QWEN_MODEL="${QWEN_MODEL}" \
  METRICS="${metrics_csv}" \
  CONCURRENCY="${QWEN_CONCURRENCY}" \
  MAX_INFLIGHT="${QWEN_MAX_INFLIGHT}" \
  FRAME_COUNT="${QWEN_FRAME_COUNT}" \
  MAX_IMAGE_SIDE="${QWEN_MAX_IMAGE_SIDE}" \
  MODEL_MAX_TOKENS="${QWEN_MODEL_MAX_TOKENS}" \
  bash "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.sh"
}

run_gpt_eval_if_needed() {
  local run_name="$1"
  local video_dir="${VIDEO_ROOT}/${run_name}"
  if [[ "${RUN_GPT_IF}" != "1" ]]; then
    return
  fi
  if find_complete_csv "${run_name}" gpt_if >/dev/null; then
    return
  fi

  local gpt_run_name="${run_name}_gpt55_task_c${GPT_CONCURRENCY}"
  echo "Running GPT-IF eval: ${gpt_run_name}"
  VIDEO_DIR="${video_dir}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_NAME="${gpt_run_name}" \
  DIFROST_MODEL="${DIFROST_MODEL}" \
  CONCURRENCY="${GPT_CONCURRENCY}" \
  FRAME_COUNT="${GPT_FRAME_COUNT}" \
  SCALE_FACTOR="${GPT_SCALE_FACTOR}" \
  MODEL_MAX_TOKENS="${GPT_MODEL_MAX_TOKENS}" \
  bash "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/eval_dreamgenbench_gpt_if_api.sh"
}

write_summary() {
  python3 - "${MANIFEST_PATH}" "${EVAL_ROOT}" "${OUTPUT_ROOT}" "${EXPECTED_COUNT}" "${SUMMARY_CSV}" "${SUMMARY_JSON}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

manifest, eval_root, output_root, expected, summary_csv, summary_json = sys.argv[1:7]
eval_root = Path(eval_root)
output_root = Path(output_root)
generated_root = eval_root / "generated_side_by_side"
video_root = eval_root / "dreamgenbench_video_dirs"
expected = int(expected)
summary_csv = Path(summary_csv)
summary_json = Path(summary_json)

def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

def find_complete(run: str, suffix: str) -> Path | None:
    paths = sorted(output_root.glob(f"{run}*_{suffix}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        if row_count(path) == expected:
            return path
    return None

def binary_score(path: Path | None):
    if path is None:
        return None
    vals = []
    errors = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(int(float(row.get("prediction", 0))))
            except Exception:
                vals.append(0)
            if row.get("error"):
                errors += 1
    return {
        "path": str(path),
        "count": len(vals),
        "positive": sum(vals),
        "score": sum(vals) / len(vals) if vals else None,
        "error_count": errors,
    }

def pa_ii_score(path: Path | None):
    if path is None:
        return None
    vals = []
    raw = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(int(float(row.get("prediction", 0))))
            except Exception:
                vals.append(0)
            if row.get("raw_score") not in (None, ""):
                raw.append(float(row["raw_score"]))
    return {
        "path": str(path),
        "count": len(vals),
        "positive": sum(vals),
        "score": sum(vals) / len(vals) if vals else None,
        "mean_raw": statistics.mean(raw) if raw else None,
        "median_raw": statistics.median(raw) if raw else None,
        "min_raw": min(raw) if raw else None,
        "max_raw": max(raw) if raw else None,
    }

def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def peak_info(path: Path):
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    return {
        "path": str(path),
        "peak_memory_used_mib": data.get("peak_memory_used_mib"),
        "peak_memory_used_gib": data.get("peak_memory_used_gib"),
        "peak_gpu_index": data.get("peak_gpu_index"),
        "peak_gpu_name": data.get("peak_gpu_name"),
        "sample_count": data.get("sample_count"),
        "cuda_visible_devices": data.get("cuda_visible_devices"),
    }

def find_pa2_peak(run: str) -> dict:
    pa_run_prefix = f"{run}_videophy_pa2"
    candidates = sorted(
        (output_root / "kjob_logs").glob(f"{pa_run_prefix}*_gpu_memory_peak.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        info = peak_info(path)
        if info:
            return info
    return {}

runs = []
with open(manifest, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        run = row["run_name"]
        save_dir = generated_root / run
        video_dir = video_root / run
        gen = load_json(save_dir / "generation_summary.json") or {}
        gen_done = gen.get("request_count") == expected and gen.get("generated_count") == expected
        video_count = len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0
        qwen_csv = find_complete(run, "qwen_if")
        gpt_csv = find_complete(run, "gpt_if")
        pa_i_csv = find_complete(run, "pa_i")
        pa_ii_csv = find_complete(run, "pa_ii")
        qwen = binary_score(qwen_csv)
        gpt = binary_score(gpt_csv)
        pa_i = binary_score(pa_i_csv)
        pa_ii = pa_ii_score(pa_ii_csv)
        gen_peak = peak_info(save_dir / "gpu_memory_peak.json")
        pa2_peak = find_pa2_peak(run)
        pa = None
        if pa_i and pa_ii and pa_i.get("score") is not None and pa_ii.get("score") is not None:
            pa = (pa_i["score"] + pa_ii["score"]) / 2.0
        runs.append({
            **row,
            "generation_complete": gen_done,
            "request_count": gen.get("request_count"),
            "generated_count": gen.get("generated_count"),
            "generated_only_count": video_count,
            "generation_wall_time_sec": gen.get("wall_time_sec"),
            "generation_mean_sec": gen.get("model_elapsed_mean_sec"),
            "generation_median_sec": gen.get("model_elapsed_median_sec"),
            "generation_gpu_peak": gen_peak,
            "qwen_if": qwen,
            "gpt_if": gpt,
            "pa_i": pa_i,
            "pa_ii": pa_ii,
            "pa_ii_gpu_peak": pa2_peak,
            "qwen_if_gpu_peak": {"note": "remote API judge; server-side GPU memory is not sampled by this local script"} if qwen else None,
            "gpt_if_gpu_peak": {"note": "remote API judge; server-side GPU memory is not sampled by this local script"} if gpt else None,
            "pa": pa,
        })

fields = [
    "model", "target", "frames", "actual", "run_name",
    "generation_complete", "generated_count", "request_count", "generated_only_count",
    "generation_wall_time_sec", "generation_mean_sec", "generation_median_sec",
    "generation_gpu_peak_mib", "generation_gpu_peak_gib", "generation_gpu_peak_index",
    "generation_gpu_peak_name", "generation_gpu_sample_count", "generation_gpu_peak_json",
    "qwen_if_positive", "qwen_if_score", "qwen_if_csv",
    "qwen_if_gpu_peak_note",
    "gpt_if_positive", "gpt_if_score", "gpt_if_csv",
    "gpt_if_gpu_peak_note",
    "pa_i_positive", "pa_i_score", "pa_i_csv",
    "pa_ii_positive", "pa_ii_score", "pa_ii_mean_raw", "pa_ii_csv",
    "pa_ii_gpu_peak_mib", "pa_ii_gpu_peak_gib", "pa_ii_gpu_peak_index",
    "pa_ii_gpu_peak_name", "pa_ii_gpu_sample_count", "pa_ii_gpu_peak_json",
    "pa", "complete_for_task_completion_eval",
]
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for item in runs:
        def metric(metric_name, key):
            metric_obj = item.get(metric_name)
            if not metric_obj:
                return None
            return metric_obj.get(key)
        gen_peak = item.get("generation_gpu_peak") or {}
        pa2_peak = item.get("pa_ii_gpu_peak") or {}
        qwen_peak = item.get("qwen_if_gpu_peak") or {}
        gpt_peak = item.get("gpt_if_gpu_peak") or {}
        writer.writerow({
            "model": item["model"],
            "target": item["target"],
            "frames": item["frames"],
            "actual": item["actual"],
            "run_name": item["run_name"],
            "generation_complete": item["generation_complete"],
            "generated_count": item["generated_count"],
            "request_count": item["request_count"],
            "generated_only_count": item["generated_only_count"],
            "generation_wall_time_sec": item["generation_wall_time_sec"],
            "generation_mean_sec": item["generation_mean_sec"],
            "generation_median_sec": item["generation_median_sec"],
            "generation_gpu_peak_mib": gen_peak.get("peak_memory_used_mib"),
            "generation_gpu_peak_gib": gen_peak.get("peak_memory_used_gib"),
            "generation_gpu_peak_index": gen_peak.get("peak_gpu_index"),
            "generation_gpu_peak_name": gen_peak.get("peak_gpu_name"),
            "generation_gpu_sample_count": gen_peak.get("sample_count"),
            "generation_gpu_peak_json": gen_peak.get("path"),
            "qwen_if_positive": metric("qwen_if", "positive"),
            "qwen_if_score": metric("qwen_if", "score"),
            "qwen_if_csv": metric("qwen_if", "path"),
            "qwen_if_gpu_peak_note": qwen_peak.get("note"),
            "gpt_if_positive": metric("gpt_if", "positive"),
            "gpt_if_score": metric("gpt_if", "score"),
            "gpt_if_csv": metric("gpt_if", "path"),
            "gpt_if_gpu_peak_note": gpt_peak.get("note"),
            "pa_i_positive": metric("pa_i", "positive"),
            "pa_i_score": metric("pa_i", "score"),
            "pa_i_csv": metric("pa_i", "path"),
            "pa_ii_positive": metric("pa_ii", "positive"),
            "pa_ii_score": metric("pa_ii", "score"),
            "pa_ii_mean_raw": metric("pa_ii", "mean_raw"),
            "pa_ii_csv": metric("pa_ii", "path"),
            "pa_ii_gpu_peak_mib": pa2_peak.get("peak_memory_used_mib"),
            "pa_ii_gpu_peak_gib": pa2_peak.get("peak_memory_used_gib"),
            "pa_ii_gpu_peak_index": pa2_peak.get("peak_gpu_index"),
            "pa_ii_gpu_peak_name": pa2_peak.get("peak_gpu_name"),
            "pa_ii_gpu_sample_count": pa2_peak.get("sample_count"),
            "pa_ii_gpu_peak_json": pa2_peak.get("path"),
            "pa": item["pa"],
            "complete_for_task_completion_eval": item["generation_complete"] and item["generated_only_count"] == expected,
        })

payload = {
    "summary_csv": str(summary_csv),
    "runs": runs,
    "note": "Qwen-IF and GPT-IF are task-completion / instruction-following metrics. PA-I and PA-II are physical-alignment metrics.",
}
summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Summary CSV:  {summary_csv}")
print(f"Summary JSON: {summary_json}")
print()
print("| Model | Target | Run | Ready | Gen GPU peak | Qwen-IF | GPT-IF | PA-I | PA-II | PA-II GPU peak |")
print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for item in runs:
    ready = item["generation_complete"] and item["generated_only_count"] == expected
    def show(metric_name):
        metric_obj = item.get(metric_name)
        if not metric_obj or metric_obj.get("score") is None:
            return "-"
        return f"{metric_obj.get('positive')} / {metric_obj.get('count')} = {metric_obj.get('score'):.6f}"
    def show_peak(peak_name):
        peak_obj = item.get(peak_name) or {}
        if peak_obj.get("peak_memory_used_mib") is None:
            return "-"
        return f"{peak_obj.get('peak_memory_used_mib')} MiB / {peak_obj.get('peak_memory_used_gib')} GiB"
    print(
        f"| {item['model']} | {item['target']} | `{item['run_name']}` | "
        f"{'yes' if ready else 'no'} | {show_peak('generation_gpu_peak')} | "
        f"{show('qwen_if')} | {show('gpt_if')} | {show('pa_i')} | {show('pa_ii')} | {show_peak('pa_ii_gpu_peak')} |"
    )
PY
}

cd "${REPO_DIR}"
write_manifest

echo "DreamGen task-completion benchmark sweep for completed runs"
echo "Manifest:        ${MANIFEST_PATH}"
echo "Output summary:  ${SUMMARY_CSV}"
echo "Qwen-IF:         ${RUN_QWEN_IF}"
echo "GPT-IF:          ${RUN_GPT_IF}"
echo "PA-I:            ${RUN_PA_I}"
echo "Status only:     ${STATUS_ONLY}"
echo

tail -n +2 "${MANIFEST_PATH}" | while IFS=$'\t' read -r model target frames actual run_name; do
  echo "============================================================"
  echo "${model} ${target}: ${run_name}"

  if ! generation_done "${run_name}"; then
    echo "Skip: generation is not complete."
    continue
  fi

  ensure_generated_only "${run_name}"
  video_count="$(mp4_count "${VIDEO_ROOT}/${run_name}")"
  if [[ "${video_count}" != "${EXPECTED_COUNT}" ]]; then
    echo "Skip: generated-only video count is ${video_count}, expected ${EXPECTED_COUNT}."
    continue
  fi

  if [[ "${STATUS_ONLY}" == "1" ]]; then
    echo "Ready for task-completion eval; status-only mode skips evaluation."
    continue
  fi

  run_qwen_eval_if_needed "${run_name}"
  run_gpt_eval_if_needed "${run_name}"
done

write_summary
