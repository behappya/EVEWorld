#!/usr/bin/env bash
set -euo pipefail

# Backfill and clean PBench Robot Qwen Domain/VQA evaluations up to 19.8s.
#
# Scope:
#   - Pretrain/base: 3.8s, 5.8s, 9.8s, 15.8s, 19.8s
#   - GR1/SFT:       3.8s, 5.8s, 9.8s, 15.8s, 19.8s
#
# Explicitly excludes 25.8s and 29.8s.
# Existing result JSONLs are de-duplicated and error rows are removed before
# rerunning, so --rerun-errors can refill only missing failed attempts while
# leaving successful attempts in place.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-auto}"
QWEN_TAG="${QWEN_TAG:-58_think32000}"
STATUS_ONLY="${STATUS_ONLY:-0}"
STRICT="${STRICT:-0}"
SKIP_EXISTING_ERROR_EVALS="${SKIP_EXISTING_ERROR_EVALS:-0}"

EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-174}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"

PRETRAIN_GEN_ROOT="${PRETRAIN_GEN_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_length_sweep}"
PRETRAIN_EVAL_ROOT="${PRETRAIN_EVAL_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval}"
PRETRAIN_SWEEP_ID="${PRETRAIN_SWEEP_ID:-pbench_len_sweep_full_8gpu}"

GR1_GEN_ROOT="${GR1_GEN_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gr1_pbench_robot_length_sweep}"
GR1_EVAL_ROOT="${GR1_EVAL_ROOT:-/data/datasets/gagi/giga_world_0_outputs/gr1_pbench_robot_qwen_vqa_eval}"
GR1_SWEEP_ID="${GR1_SWEEP_ID:-gr1_pbench_len_sweep_full_8gpu}"

CONCURRENCY="${CONCURRENCY:-100}"
MAX_INFLIGHT="${MAX_INFLIGHT:-100}"
FRAME_COUNT="${FRAME_COUNT:-8}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-512}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-32000}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-600}"
MODEL_RETRIES="${MODEL_RETRIES:-3}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"

TARGETS=(
  "3.8s|3p8s|61"
  "5.8s|5p8s|93"
  "9.8s|9p8s|157"
  "15.8s|15p8s|253"
  "19.8s|19p8s|317"
)

expected_attempts() {
  python3 - "${METADATA_JSONL}" "${EXPECTED_SAMPLES}" <<'PY'
import json
import sys
path, expected = sys.argv[1], int(sys.argv[2])
total = 0
count = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        if count >= expected:
            break
        row = json.loads(line)
        total += len(row.get("qa_pairs") or [])
        count += 1
print(total)
PY
}

EXPECTED_ATTEMPTS="${EXPECTED_ATTEMPTS:-$(expected_attempts)}"

generation_done() {
  local summary_path="$1"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "${summary_path}" "${EXPECTED_SAMPLES}" <<'PY'
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
  find "${dir}" -maxdepth 1 -type f -name 'robot_*.mp4' 2>/dev/null | wc -l
}

eval_valid() {
  local eval_dir="$1"
  local summary_path="${eval_dir}/qwen_vqa_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "${summary_path}" "${EXPECTED_SAMPLES}" "${EXPECTED_ATTEMPTS}" <<'PY'
import json
import sys
path, expected_samples, expected_attempts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = (
    data.get("sample_count") == expected_samples
    and data.get("attempt_count") == expected_attempts
    and int(data.get("error_count") or 0) == 0
    and int(data.get("build_error_count") or 0) == 0
)
raise SystemExit(0 if ok else 1)
PY
}

eval_complete_with_errors() {
  local eval_dir="$1"
  local summary_path="${eval_dir}/qwen_vqa_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "${summary_path}" "${EXPECTED_SAMPLES}" "${EXPECTED_ATTEMPTS}" <<'PY'
import json
import sys
path, expected_samples, expected_attempts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = (
    data.get("sample_count") == expected_samples
    and data.get("attempt_count") == expected_attempts
    and int(data.get("build_error_count") or 0) == 0
    and int(data.get("error_count") or 0) > 0
)
if ok:
    print(f"error_count={int(data.get('error_count') or 0)} domain_score_like={data.get('domain_score_like')}")
raise SystemExit(0 if ok else 1)
PY
}

normalize_result_jsonl() {
  local eval_dir="$1"
  local result_path="${eval_dir}/qwen_vqa_results.jsonl"
  [[ -f "${result_path}" ]] || return 0
  python3 - "${result_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = []
bad_lines = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            bad_lines += 1

latest = {}
error_count = 0
for idx, row in enumerate(records):
    key = (str(row.get("pbench_id") or ""), int(row.get("question_index") or 0), int(row.get("attempt") or 0))
    if not key[0] or not key[1] or not key[2]:
        error_count += 1
        continue
    if row.get("error"):
        error_count += 1
        continue
    latest[key] = (idx, row)

clean = [item[1] for _, item in sorted(latest.items(), key=lambda kv: kv[1][0])]
changed = len(clean) != len(records) or bad_lines
if changed:
    backup = path.with_name(path.name + ".backup_" + __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    path.replace(backup)
    with path.open("w", encoding="utf-8") as f:
        for row in clean:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"normalized {path}: kept={len(clean)} dropped={len(records)-len(clean)} bad_lines={bad_lines} backup={backup}")
else:
    print(f"normalized {path}: no changes")
PY
}

run_eval() {
  local model_label="$1"
  local target="$2"
  local label="$3"
  local run_name="$4"
  local video_dir="$5"
  local eval_dir="$6"

  echo
  echo "============================================================"
  echo "${model_label} PBench ${target}: ${run_name}"
  echo "Video dir: ${video_dir}"
  echo "Eval dir:  ${eval_dir}"
  echo "============================================================"

  if ! generation_done "${video_dir}/generation_summary.json"; then
    echo "Skip: generation incomplete, mp4=$(mp4_count "${video_dir}")/${EXPECTED_SAMPLES}"
    return
  fi

  if eval_valid "${eval_dir}"; then
    echo "Skip: clean eval already complete."
    return
  fi

  if [[ "${SKIP_EXISTING_ERROR_EVALS}" == "1" ]]; then
    local partial_status
    if partial_status="$(eval_complete_with_errors "${eval_dir}" 2>/dev/null)"; then
      echo "Skip: eval covers all samples but still has errors (${partial_status}); will backfill errors later."
      return
    fi
  fi

  if [[ "${STATUS_ONLY}" == "1" ]]; then
    if [[ -f "${eval_dir}/qwen_vqa_results.jsonl" ]]; then
      python3 - "${eval_dir}/qwen_vqa_results.jsonl" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
rows = []
bad = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
errors = sum(1 for row in rows if row.get("error"))
print(f"Existing result rows={len(rows)}, error_rows={errors}, bad_lines={bad}; would normalize before rerun.")
PY
    fi
    echo "Would run/rerun Qwen eval: ${eval_dir}"
    return
  fi

  if [[ -d "${eval_dir}" ]]; then
    normalize_result_jsonl "${eval_dir}"
  fi

  VIDEO_DIR="${video_dir}" \
  EVAL_DIR="${eval_dir}" \
  METADATA_JSONL="${METADATA_JSONL}" \
  QWEN_BASE="${QWEN_BASE}" \
  QWEN_MODEL="${QWEN_MODEL}" \
  LIMIT=0 \
  CONCURRENCY="${CONCURRENCY}" \
  MAX_INFLIGHT="${MAX_INFLIGHT}" \
  FRAME_COUNT="${FRAME_COUNT}" \
  MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE}" \
  MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS}" \
  MODEL_TIMEOUT="${MODEL_TIMEOUT}" \
  MODEL_RETRIES="${MODEL_RETRIES}" \
  DISABLE_THINKING="${DISABLE_THINKING}" \
  RERUN_ERRORS=1 \
  bash "${EVEWORLD_ROOT}/benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh"

  if ! eval_valid "${eval_dir}"; then
    echo "Eval still incomplete or has errors: ${eval_dir}" >&2
    if [[ "${SKIP_EXISTING_ERROR_EVALS}" == "1" ]]; then
      echo "Continue because SKIP_EXISTING_ERROR_EVALS=1; remaining errors will be backfilled later." >&2
      return
    fi
    [[ "${STRICT}" == "1" ]] && exit 1
  fi
}

write_summary() {
  python3 - \
    "${PRETRAIN_GEN_ROOT}" "${PRETRAIN_EVAL_ROOT}" "${PRETRAIN_SWEEP_ID}" \
    "${GR1_GEN_ROOT}" "${GR1_EVAL_ROOT}" "${GR1_SWEEP_ID}" \
    "${QWEN_TAG}" "${EXPECTED_ATTEMPTS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

pre_gen, pre_eval, pre_sweep, gr1_gen, gr1_eval, gr1_sweep, tag, expected_attempts = sys.argv[1:9]
expected_attempts = int(expected_attempts)
rows = []
targets = [("3.8s", "3p8s", 61), ("5.8s", "5p8s", 93), ("9.8s", "9p8s", 157), ("15.8s", "15p8s", 253), ("19.8s", "19p8s", 317)]

def load(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())

for model, gen_root, eval_root, sweep, prefix in [
    ("Pretrain", Path(pre_gen), Path(pre_eval), pre_sweep, "pbench_robot"),
    ("GR1/SFT", Path(gr1_gen), Path(gr1_eval), gr1_sweep, "gr1_pbench_robot"),
]:
    for target, label, frames in targets:
        run = f"{prefix}_{label}_full_{sweep}"
        gen_dir = gen_root / run
        eval_dir = eval_root / f"{run}_domain_qwen36vl_{tag}"
        gen = load(gen_dir / "generation_summary.json")
        peak = load(gen_dir / "gpu_memory_peak.json")
        q = load(eval_dir / "qwen_vqa_summary.json")
        rows.append({
            "model": model,
            "target": target,
            "frames": frames,
            "fps": 16,
            "run_name": run,
            "generated_count": gen.get("generated_count"),
            "request_count": gen.get("request_count"),
            "wall_time_sec": gen.get("wall_time_sec"),
            "mean_sec": gen.get("model_elapsed_mean_sec"),
            "peak_gib": peak.get("peak_memory_used_gib"),
            "eval_dir": str(eval_dir),
            "sample_count": q.get("sample_count"),
            "attempt_count": q.get("attempt_count"),
            "expected_attempts": expected_attempts,
            "error_count": q.get("error_count"),
            "domain_score_like": q.get("domain_score_like"),
        })

out_root = Path(gr1_eval).parent
csv_path = out_root / f"pbench_upto19_qwen{tag}_clean_summary.csv"
json_path = out_root / f"pbench_upto19_qwen{tag}_clean_summary.json"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(json.dumps({"rows": rows, "csv": str(csv_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"summary_csv: {csv_path}")
print(f"summary_json: {json_path}")
PY
}

cd "${REPO_DIR}"

echo "PBench <=19.8s Qwen clean backfill"
echo "Qwen base:        ${QWEN_BASE}"
echo "Qwen tag:         ${QWEN_TAG}"
echo "Expected samples: ${EXPECTED_SAMPLES}"
echo "Expected attempts:${EXPECTED_ATTEMPTS}"
echo "Max tokens:       ${MODEL_MAX_TOKENS}"
echo "Thinking enabled: $([[ "${DISABLE_THINKING}" == "0" ]] && echo yes || echo no)"
echo "Skip existing err:${SKIP_EXISTING_ERROR_EVALS}"
echo "Status only:      ${STATUS_ONLY}"

for item in "${TARGETS[@]}"; do
  IFS='|' read -r target label _frames <<<"${item}"

  pre_run="pbench_robot_${label}_full_${PRETRAIN_SWEEP_ID}"
  run_eval \
    "Pretrain" "${target}" "${label}" "${pre_run}" \
    "${PRETRAIN_GEN_ROOT}/${pre_run}" \
    "${PRETRAIN_EVAL_ROOT}/${pre_run}_domain_qwen36vl_${QWEN_TAG}"

  gr1_run="gr1_pbench_robot_${label}_full_${GR1_SWEEP_ID}"
  run_eval \
    "GR1/SFT" "${target}" "${label}" "${gr1_run}" \
    "${GR1_GEN_ROOT}/${gr1_run}" \
    "${GR1_EVAL_ROOT}/${gr1_run}_domain_qwen36vl_${QWEN_TAG}"
done

write_summary
echo "Done."
