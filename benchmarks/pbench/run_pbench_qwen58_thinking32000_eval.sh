#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate completed PBench length-sweep generations with Qwen thinking enabled
# and a large max_tokens budget. Outputs are written to separate directories so the
# existing 256-token / thinking-disabled baseline can keep running unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-auto}"
QWEN_TAG="${QWEN_TAG:-58_think32000}"
BASELINE_QWEN_TAG="${BASELINE_QWEN_TAG:-58}"

STATUS_ONLY="${STATUS_ONLY:-0}"
FORCE="${FORCE:-0}"

PBENCH_EXPECTED="${PBENCH_EXPECTED:-174}"
PBENCH_GEN_ROOT="${PBENCH_GEN_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_length_sweep}"
PBENCH_DOMAIN_ROOT="${PBENCH_DOMAIN_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval}"
PBENCH_SWEEP_ID="${PBENCH_SWEEP_ID:-pbench_len_sweep_full_8gpu}"
PBENCH_METADATA_JSONL="${PBENCH_METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"

PBENCH_CONCURRENCY="${PBENCH_CONCURRENCY:-100}"
PBENCH_MAX_INFLIGHT="${PBENCH_MAX_INFLIGHT:-100}"
PBENCH_FRAME_COUNT="${PBENCH_FRAME_COUNT:-8}"
PBENCH_MAX_IMAGE_SIDE="${PBENCH_MAX_IMAGE_SIDE:-512}"
PBENCH_MODEL_MAX_TOKENS="${PBENCH_MODEL_MAX_TOKENS:-32000}"
PBENCH_MODEL_TIMEOUT="${PBENCH_MODEL_TIMEOUT:-600}"
PBENCH_MODEL_RETRIES="${PBENCH_MODEL_RETRIES:-3}"
PBENCH_DISABLE_THINKING="${PBENCH_DISABLE_THINKING:-0}"

PBENCH_SUMMARY_CSV="${PBENCH_SUMMARY_CSV:-${PBENCH_GEN_ROOT}/${PBENCH_SWEEP_ID}_qwen${QWEN_TAG}_domain_summary.csv}"
PBENCH_SUMMARY_JSON="${PBENCH_SUMMARY_JSON:-${PBENCH_GEN_ROOT}/${PBENCH_SWEEP_ID}_qwen${QWEN_TAG}_domain_summary.json}"
PBENCH_COMPARE_CSV="${PBENCH_COMPARE_CSV:-${PBENCH_GEN_ROOT}/${PBENCH_SWEEP_ID}_qwen${BASELINE_QWEN_TAG}_vs_qwen${QWEN_TAG}_domain_compare.csv}"
PBENCH_COMPARE_JSON="${PBENCH_COMPARE_JSON:-${PBENCH_GEN_ROOT}/${PBENCH_SWEEP_ID}_qwen${BASELINE_QWEN_TAG}_vs_qwen${QWEN_TAG}_domain_compare.json}"

PBENCH_TARGETS=(
  "3.8s|3p8s|61"
  "5.8s|5p8s|93"
  "9.8s|9p8s|157"
  "15.8s|15p8s|253"
  "19.8s|19p8s|317"
  "25.8s|25p8s|413"
  "29.8s|29p8s|477"
)

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

archive_eval_dir() {
  local eval_dir="$1"
  [[ -d "${eval_dir}" ]] || return 0
  local archived="${eval_dir}_archived_$(date +%Y%m%d_%H%M%S)"
  echo "Archive previous eval dir: ${eval_dir} -> ${archived}"
  mv "${eval_dir}" "${archived}"
}

write_summary_and_compare() {
  python3 - \
    "${PBENCH_GEN_ROOT}" \
    "${PBENCH_DOMAIN_ROOT}" \
    "${PBENCH_SWEEP_ID}" \
    "${BASELINE_QWEN_TAG}" \
    "${QWEN_TAG}" \
    "${PBENCH_SUMMARY_CSV}" \
    "${PBENCH_SUMMARY_JSON}" \
    "${PBENCH_COMPARE_CSV}" \
    "${PBENCH_COMPARE_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

gen_root, domain_root, sweep_id, base_tag, new_tag = sys.argv[1:6]
summary_csv, summary_json, compare_csv, compare_json = map(Path, sys.argv[6:10])
gen_root = Path(gen_root)
domain_root = Path(domain_root)
labels = [
    ("3.8s", "3p8s", 61),
    ("5.8s", "5p8s", 93),
    ("9.8s", "9p8s", 157),
    ("15.8s", "15p8s", 253),
    ("19.8s", "19p8s", 317),
    ("25.8s", "25p8s", 413),
    ("29.8s", "29p8s", 477),
]

def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def delta(new, base):
    if new is None or base is None:
        return None
    try:
        return round(float(new) - float(base), 6)
    except Exception:
        return None

def fmt(value):
    return "-" if value is None else f"{float(value):.6f}"

summary_rows = []
compare_rows = []
for target, label, frames in labels:
    run_name = f"pbench_robot_{label}_full_{sweep_id}"
    gen_dir = gen_root / run_name
    new_dir = domain_root / f"{run_name}_domain_qwen36vl_{new_tag}"
    base_dir = domain_root / f"{run_name}_domain_qwen36vl_{base_tag}"
    gen = load_json(gen_dir / "generation_summary.json")
    peak = load_json(gen_dir / "gpu_memory_peak.json")
    new_summary = load_json(new_dir / "qwen_vqa_summary.json")
    base_summary = load_json(base_dir / "qwen_vqa_summary.json")
    new_config = load_json(new_dir / "run_config.json")
    base_config = load_json(base_dir / "run_config.json")
    summary_rows.append({
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
        "eval_dir": str(new_dir),
        "qwen_base": new_config.get("qwen_base"),
        "qwen_model": new_config.get("qwen_model"),
        "frame_count": new_config.get("frame_count"),
        "max_image_side": new_config.get("max_image_side"),
        "crop_mode": new_config.get("crop_mode"),
        "model_max_tokens": new_config.get("model_max_tokens"),
        "disable_thinking": new_config.get("disable_thinking"),
        "thinking_enabled": new_config.get("thinking_enabled"),
        "sample_count": new_summary.get("sample_count"),
        "error_count": new_summary.get("error_count"),
        "domain_score_like": new_summary.get("domain_score_like"),
        "question_micro_accuracy": new_summary.get("question_micro_accuracy"),
        "sample_macro_accuracy": new_summary.get("sample_macro_accuracy"),
    })
    compare_rows.append({
        "target": target,
        "num_frames": frames,
        "actual_sec": frames / 16.0,
        "run_name": run_name,
        "baseline_tag": base_tag,
        "baseline_eval_dir": str(base_dir),
        "baseline_model_max_tokens": base_config.get("model_max_tokens"),
        "baseline_disable_thinking": base_config.get("disable_thinking"),
        "baseline_sample_count": base_summary.get("sample_count"),
        "baseline_error_count": base_summary.get("error_count"),
        "baseline_domain_score_like": base_summary.get("domain_score_like"),
        "baseline_question_micro_accuracy": base_summary.get("question_micro_accuracy"),
        "new_tag": new_tag,
        "new_eval_dir": str(new_dir),
        "new_model_max_tokens": new_config.get("model_max_tokens"),
        "new_disable_thinking": new_config.get("disable_thinking"),
        "new_thinking_enabled": new_config.get("thinking_enabled"),
        "new_sample_count": new_summary.get("sample_count"),
        "new_error_count": new_summary.get("error_count"),
        "new_domain_score_like": new_summary.get("domain_score_like"),
        "new_question_micro_accuracy": new_summary.get("question_micro_accuracy"),
        "domain_score_like_delta": delta(new_summary.get("domain_score_like"), base_summary.get("domain_score_like")),
        "question_micro_accuracy_delta": delta(new_summary.get("question_micro_accuracy"), base_summary.get("question_micro_accuracy")),
    })

for path, rows in ((summary_csv, summary_rows), (compare_csv, compare_rows)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

summary_json.write_text(json.dumps({"rows": summary_rows, "summary_csv": str(summary_csv)}, ensure_ascii=False, indent=2), encoding="utf-8")
compare_json.write_text(json.dumps({"rows": compare_rows, "compare_csv": str(compare_csv)}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Thinking summary CSV: {summary_csv}")
print(f"Thinking summary JSON: {summary_json}")
print(f"Compare CSV:          {compare_csv}")
print(f"Compare JSON:         {compare_json}")
print("| Target | 256 score | 32000+thinking score | Delta | 32000 errors |")
print("| ---: | ---: | ---: | ---: | ---: |")
for row in compare_rows:
    print(
        f"| {row['target']} | "
        f"{fmt(row['baseline_domain_score_like'])} | "
        f"{fmt(row['new_domain_score_like'])} | "
        f"{fmt(row['domain_score_like_delta'])} | "
        f"{row['new_error_count'] if row['new_error_count'] is not None else '-'} |"
    )
PY
}

run_pbench() {
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

    if [[ "${FORCE}" == "1" && "${STATUS_ONLY}" != "1" ]]; then
      archive_eval_dir "${eval_dir}"
    elif pbench_domain_valid "${eval_dir}" "${PBENCH_EXPECTED}"; then
      echo "Skip: complete existing thinking eval: ${eval_dir}"
      continue
    fi

    if [[ "${STATUS_ONLY}" == "1" ]]; then
      echo "Would run Qwen thinking eval: ${eval_dir}"
      continue
    fi

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
    MODEL_TIMEOUT="${PBENCH_MODEL_TIMEOUT}" \
    MODEL_RETRIES="${PBENCH_MODEL_RETRIES}" \
    DISABLE_THINKING="${PBENCH_DISABLE_THINKING}" \
    RERUN_ERRORS=1 \
    bash "${EVEWORLD_ROOT}/benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh"

    if ! pbench_domain_valid "${eval_dir}" "${PBENCH_EXPECTED}"; then
      echo "PBench thinking eval incomplete or has errors: ${eval_dir}" >&2
      exit 1
    fi
  done
}

cd "${REPO_DIR}"

echo "PBench Qwen thinking-32000 re-eval"
echo "Repo:                  ${REPO_DIR}"
echo "Qwen base:             ${QWEN_BASE}"
echo "Qwen model:            ${QWEN_MODEL}"
echo "Output tag:            ${QWEN_TAG}"
echo "Baseline tag:          ${BASELINE_QWEN_TAG}"
echo "Max tokens:            ${PBENCH_MODEL_MAX_TOKENS}"
echo "Disable thinking:      ${PBENCH_DISABLE_THINKING}"
echo "Concurrency/inflight:  ${PBENCH_CONCURRENCY}/${PBENCH_MAX_INFLIGHT}"
echo "Frames/image side:     ${PBENCH_FRAME_COUNT}/${PBENCH_MAX_IMAGE_SIDE}"
echo "Status only:           ${STATUS_ONLY}"
echo "Force rerun:           ${FORCE}"

run_pbench
write_summary_and_compare

echo
echo "Done."
