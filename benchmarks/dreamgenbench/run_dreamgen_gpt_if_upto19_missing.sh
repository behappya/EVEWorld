#!/usr/bin/env bash
set -euo pipefail

# Backfill DreamGenBench GPT-IF up to 19.8s only.
#
# Scope:
#   - SFT:      5.8s, 9.8s, 15.8s, 19.8s
#   - Pretrain: 5.8s, 9.8s, 15.8s, 19.8s
#
# Explicitly excludes 25.8s and 29.8s.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
GENERATED_ROOT="${GENERATED_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
VIDEO_ROOT="${VIDEO_ROOT:-${EVAL_ROOT}/dreamgenbench_video_dirs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_ROOT}/eval_outputs}"
EXPECTED_COUNT="${EXPECTED_COUNT:-92}"

STATUS_ONLY="${STATUS_ONLY:-0}"
STRICT="${STRICT:-1}"

DIFROST_GENAI_BASE_URL="${DIFROST_GENAI_BASE_URL:-https://api-gateway.example.com/v1}"
DIFROST_HOST="${DIFROST_HOST:-api-gateway.example.com}"
DIFROST_MODEL="${DIFROST_MODEL:-gpt-5.5}"
GPT_CONCURRENCY="${GPT_CONCURRENCY:-10}"
GPT_FRAME_COUNT="${GPT_FRAME_COUNT:-8}"
GPT_SCALE_FACTOR="${GPT_SCALE_FACTOR:-0.5}"
GPT_MODEL_MAX_TOKENS="${GPT_MODEL_MAX_TOKENS:-32000}"
GPT_MODEL_RETRIES="${GPT_MODEL_RETRIES:-6}"
GPT_THINKING_LEVEL="${GPT_THINKING_LEVEL:-low}"

RUNS=(
  "SFT|5.8s|gr1_dreamgen_8gpu_full_20260625_212933"
  "SFT|9.8s|gr1_dreamgen_8gpu_9p8s_full_20260627_140442"
  "SFT|15.8s|gr1_dreamgen_8gpu_15p8s_full_20260627_145209"
  "SFT|19.8s|sft_dreamgen_8gpu_19p8s_full_long_20_25_30"
  "Pretrain|5.8s|pretrain_dreamgen_8gpu_5p8s_full_20260627_162849"
  "Pretrain|9.8s|pretrain_dreamgen_8gpu_9p8s_full_20260627_162849"
  "Pretrain|15.8s|pretrain_dreamgen_8gpu_15p8s_full_20260627_162849"
  "Pretrain|19.8s|pretrain_dreamgen_8gpu_19p8s_full_long_20_25_30"
)

generation_done() {
  local summary_path="$1"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "${summary_path}" "${EXPECTED_COUNT}" <<'PY'
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

csv_valid() {
  local csv_path="$1"
  [[ -f "${csv_path}" ]] || return 1
  python3 - "${csv_path}" "${EXPECTED_COUNT}" <<'PY'
import csv
import sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ok = len(rows) == expected and not any(row.get("error") for row in rows)
raise SystemExit(0 if ok else 1)
PY
}

find_complete_gpt_csv() {
  local run_name="$1"
  python3 - "${OUTPUT_ROOT}" "${run_name}" "${EXPECTED_COUNT}" <<'PY'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
run = sys.argv[2]
expected = int(sys.argv[3])
for path in sorted(root.glob(f"{run}*_gpt_if.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
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

normalize_gpt_csv() {
  local csv_path="$1"
  [[ -f "${csv_path}" ]] || return 0
  python3 - "${csv_path}" <<'PY'
import csv
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    rows = list(reader)

latest = {}
for idx, row in enumerate(rows):
    key = row.get("video_path") or ""
    if not key or row.get("error"):
        continue
    latest[key] = (idx, row)

clean = [item[1] for _, item in sorted(latest.items(), key=lambda kv: kv[1][0])]
changed = len(clean) != len(rows)
if changed:
    backup = path.with_name(path.name + ".backup_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    path.replace(backup)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean)
    print(f"normalized {path}: kept={len(clean)} dropped={len(rows)-len(clean)} backup={backup}")
else:
    print(f"normalized {path}: no changes")
PY
}

ensure_video_dir() {
  local run_name="$1"
  local source_dir="${GENERATED_ROOT}/${run_name}"
  local video_dir="${VIDEO_ROOT}/${run_name}"
  if [[ "$(mp4_count "${video_dir}")" == "${EXPECTED_COUNT}" ]]; then
    return
  fi
  echo "Preparing generated-only videos: ${run_name}"
  if [[ "${STATUS_ONLY}" == "1" ]]; then
    echo "Would prepare ${video_dir}"
    return
  fi
  "${PYTHON_BIN:-python}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py" \
    --source-video-dir "${source_dir}" \
    --output-dir "${video_dir}" \
    --overwrite
}

run_one() {
  local model_label="$1"
  local target="$2"
  local run_name="$3"
  local source_dir="${GENERATED_ROOT}/${run_name}"
  local video_dir="${VIDEO_ROOT}/${run_name}"
  local gpt_run_name="${run_name}_gpt55_task_c${GPT_CONCURRENCY}"
  local csv_path="${OUTPUT_ROOT}/${gpt_run_name}_gpt_if.csv"

  echo
  echo "============================================================"
  echo "${model_label} DreamGen ${target}: ${run_name}"
  echo "Video dir: ${video_dir}"
  echo "GPT run:   ${gpt_run_name}"
  echo "============================================================"

  if ! generation_done "${source_dir}/generation_summary.json"; then
    echo "Skip: generation incomplete, source mp4=$(mp4_count "${source_dir}")/${EXPECTED_COUNT}"
    return
  fi

  ensure_video_dir "${run_name}"

  if complete_csv="$(find_complete_gpt_csv "${run_name}" 2>/dev/null)"; then
    echo "Skip: complete GPT-IF already exists: ${complete_csv}"
    return
  fi

  if [[ -f "${csv_path}" ]]; then
    normalize_gpt_csv "${csv_path}"
  fi

  if csv_valid "${csv_path}"; then
    echo "Skip: clean target CSV already complete: ${csv_path}"
    return
  fi

  if [[ "${STATUS_ONLY}" == "1" ]]; then
    echo "Would run GPT-IF: ${csv_path}"
    return
  fi

  VIDEO_DIR="${video_dir}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_NAME="${gpt_run_name}" \
  DIFROST_GENAI_BASE_URL="${DIFROST_GENAI_BASE_URL}" \
  DIFROST_HOST="${DIFROST_HOST}" \
  DIFROST_MODEL="${DIFROST_MODEL}" \
  CONCURRENCY="${GPT_CONCURRENCY}" \
  FRAME_COUNT="${GPT_FRAME_COUNT}" \
  SCALE_FACTOR="${GPT_SCALE_FACTOR}" \
  MODEL_RETRIES="${GPT_MODEL_RETRIES}" \
  MODEL_MAX_TOKENS="${GPT_MODEL_MAX_TOKENS}" \
  THINKING_LEVEL="${GPT_THINKING_LEVEL}" \
  python "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/eval_dreamgenbench_gpt_if_api.py" \
    --video-dir "${video_dir}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${gpt_run_name}" \
    --base-url "${DIFROST_GENAI_BASE_URL}" \
    --host "${DIFROST_HOST}" \
    --api-token "${DIFROST_API_TOKEN:?set DIFROST_API_TOKEN}" \
    --model "${DIFROST_MODEL}" \
    --concurrency "${GPT_CONCURRENCY}" \
    --frame-count "${GPT_FRAME_COUNT}" \
    --scale-factor "${GPT_SCALE_FACTOR}" \
    --model-retries "${GPT_MODEL_RETRIES}" \
    --model-max-tokens "${GPT_MODEL_MAX_TOKENS}" \
    --thinking-level "${GPT_THINKING_LEVEL}" \
    --rerun-errors

  if ! csv_valid "${csv_path}"; then
    echo "GPT-IF still incomplete or has errors: ${csv_path}" >&2
    [[ "${STRICT}" == "1" ]] && exit 1
  fi
}

write_summary() {
  python3 - "${OUTPUT_ROOT}" "${EXPECTED_COUNT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
runs = [
    ("SFT", "5.8s", "gr1_dreamgen_8gpu_full_20260625_212933"),
    ("SFT", "9.8s", "gr1_dreamgen_8gpu_9p8s_full_20260627_140442"),
    ("SFT", "15.8s", "gr1_dreamgen_8gpu_15p8s_full_20260627_145209"),
    ("SFT", "19.8s", "sft_dreamgen_8gpu_19p8s_full_long_20_25_30"),
    ("Pretrain", "5.8s", "pretrain_dreamgen_8gpu_5p8s_full_20260627_162849"),
    ("Pretrain", "9.8s", "pretrain_dreamgen_8gpu_9p8s_full_20260627_162849"),
    ("Pretrain", "15.8s", "pretrain_dreamgen_8gpu_15p8s_full_20260627_162849"),
    ("Pretrain", "19.8s", "pretrain_dreamgen_8gpu_19p8s_full_long_20_25_30"),
]

def summarize(path):
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    vals = []
    errors = 0
    for row in rows:
        try:
            vals.append(int(float(row.get("prediction") or 0)))
        except Exception:
            vals.append(0)
        if row.get("error"):
            errors += 1
    return {
        "path": str(path),
        "count": len(rows),
        "positive": sum(vals),
        "error_count": errors,
        "score": sum(vals) / len(vals) if vals else None,
    }

def find_best(run):
    candidates = sorted(root.glob(f"{run}*_gpt_if.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        s = summarize(path)
        if s and s["count"] == expected and s["error_count"] == 0:
            return path
    return candidates[0] if candidates else None

rows = []
for model, target, run in runs:
    path = find_best(run)
    s = summarize(path)
    rows.append({"model": model, "target": target, "run_name": run, "gpt_if": s})

json_path = root / "dreamgen_upto19_gpt_if_summary.json"
csv_path = root / "dreamgen_upto19_gpt_if_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    fields = ["model", "target", "run_name", "path", "count", "positive", "error_count", "score"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        s = row["gpt_if"] or {}
        writer.writerow({
            "model": row["model"],
            "target": row["target"],
            "run_name": row["run_name"],
            "path": s.get("path"),
            "count": s.get("count"),
            "positive": s.get("positive"),
            "error_count": s.get("error_count"),
            "score": s.get("score"),
        })
json_path.write_text(json.dumps({"rows": rows, "csv": str(csv_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"summary_csv: {csv_path}")
print(f"summary_json: {json_path}")
PY
}

cd "${REPO_DIR}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

echo "DreamGen <=19.8s GPT-IF backfill"
echo "Model:        ${DIFROST_MODEL}"
echo "Concurrency:  ${GPT_CONCURRENCY}"
echo "Max tokens:   ${GPT_MODEL_MAX_TOKENS}"
echo "Status only:  ${STATUS_ONLY}"

for item in "${RUNS[@]}"; do
  IFS='|' read -r model_label target run_name <<<"${item}"
  run_one "${model_label}" "${target}" "${run_name}"
done

write_summary
echo "Done."
