#!/usr/bin/env bash
set -euo pipefail

# End-to-end PA-II evaluation for the PhysLatent DreamGen full92 run:
# 1) verify side-by-side generation is complete
# 2) crop generated-only videos
# 3) submit VideoPhy PA-II kjob
# 4) optionally wait for PA-II CSV and print a compact summary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
GENERATED_ROOT="${GENERATED_ROOT:-${EVAL_ROOT}/generated_side_by_side}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-physlatent_adapter_step200_full92}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-${GENERATED_ROOT}/${SOURCE_RUN_NAME}}"
VIDEO_DIR="${VIDEO_DIR:-${EVAL_ROOT}/dreamgenbench_video_dirs/${SOURCE_RUN_NAME}}"
EXPECTED_COUNT="${EXPECTED_COUNT:-92}"

EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${EVAL_ROOT}/eval_outputs}"
PA2_RUN_NAME="${PA2_RUN_NAME:-${SOURCE_RUN_NAME}_videophy_pa2}"
PA2_CSV="${PA2_CSV:-${EVAL_OUTPUT_ROOT}/${PA2_RUN_NAME}_pa_ii.csv}"
PA2_RAW_CSV="${PA2_RAW_CSV:-${EVAL_OUTPUT_ROOT}/${PA2_RUN_NAME}_pa_ii_raw.csv}"
PA2_LOG="${PA2_LOG:-${EVAL_OUTPUT_ROOT}/kjob_logs/${PA2_RUN_NAME}.log}"
PA2_GPU_PEAK="${PA2_GPU_PEAK:-${EVAL_OUTPUT_ROOT}/kjob_logs/${PA2_RUN_NAME}_gpu_memory_peak.json}"

PYTHON_BIN="${PYTHON_BIN:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
PA2_GPU_IDS="${PA2_GPU_IDS:-0}"
PA2_BATCH_SIZE="${PA2_BATCH_SIZE:-16}"
VIDEOPHY_CHECKPOINT="${VIDEOPHY_CHECKPOINT:-videophysics/videocon_physics}"
WAIT_PA2="${WAIT_PA2:-1}"
POLL_SEC="${POLL_SEC:-30}"
PA2_TIMEOUT_SEC="${PA2_TIMEOUT_SEC:-21600}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
FORCE_PA2="${FORCE_PA2:-0}"
DRY_RUN="${DRY_RUN:-0}"

mp4_count() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

csv_row_count() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo 0
    return
  fi
  "${PYTHON_BIN}" - "$path" <<'PY'
import csv
import sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

verify_generation_summary() {
  local summary_path="${SOURCE_VIDEO_DIR}/generation_summary.json"
  require_file "${summary_path}"
  "${PYTHON_BIN}" - "${summary_path}" "${EXPECTED_COUNT}" <<'PY'
import json
import sys
path, expected_s = sys.argv[1:3]
expected = int(expected_s)
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
request_count = int(data.get("request_count") or -1)
generated_count = int(data.get("generated_count") or -1)
physics_path = data.get("physics_latent_model_path")
if request_count != expected or generated_count != expected:
    raise SystemExit(f"generation incomplete: request_count={request_count}, generated_count={generated_count}, expected={expected}")
if not physics_path:
    raise SystemExit("generation summary does not record physics_latent_model_path; refusing to evaluate as PhysLatent run")
print(f"generation summary ok: {generated_count}/{request_count}, physics_latent={physics_path}")
PY
}

prepare_generated_only() {
  local current_count
  current_count="$(mp4_count "${VIDEO_DIR}")"
  if [[ "${FORCE_PREPARE}" != "1" && "${current_count}" == "${EXPECTED_COUNT}" ]]; then
    echo "Generated-only videos already exist: ${VIDEO_DIR} (${current_count}/${EXPECTED_COUNT})"
    return
  fi
  echo "Preparing generated-only videos:"
  echo "  source: ${SOURCE_VIDEO_DIR}"
  echo "  output: ${VIDEO_DIR}"
  SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR}" \
    "${PYTHON_BIN}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py" \
      --source-video-dir "${SOURCE_VIDEO_DIR}" \
      --output-dir "${VIDEO_DIR}" \
      --overwrite
  current_count="$(mp4_count "${VIDEO_DIR}")"
  if [[ "${current_count}" != "${EXPECTED_COUNT}" ]]; then
    echo "Generated-only count mismatch: ${current_count}/${EXPECTED_COUNT} in ${VIDEO_DIR}" >&2
    exit 1
  fi
}

pa2_done() {
  [[ "$(csv_row_count "${PA2_CSV}")" == "${EXPECTED_COUNT}" ]]
}

check_pa2_failure() {
  if [[ ! -s "${PA2_LOG}" ]]; then
    return
  fi
  local pattern="Traceback|CUDA out of memory|Killed|Missing |No such file|RuntimeError|ValueError"
  local tmp="/tmp/physlatent_pa2_failure_${PA2_RUN_NAME}.txt"
  if command -v rg >/dev/null 2>&1; then
    rg -n "${pattern}" "${PA2_LOG}" >"${tmp}" 2>/dev/null || return
  else
    grep -En "${pattern}" "${PA2_LOG}" >"${tmp}" 2>/dev/null || return
  fi
  if [[ -s "${tmp}" ]]; then
    echo "PA-II log contains a failure pattern:" >&2
    cat "${tmp}" >&2
    exit 1
  fi
}

submit_pa2() {
  if [[ "${FORCE_PA2}" != "1" && -s "${PA2_CSV}" && "$(csv_row_count "${PA2_CSV}")" == "${EXPECTED_COUNT}" ]]; then
    echo "PA-II CSV already complete: ${PA2_CSV}"
    return
  fi
  echo "Submitting PA-II kjob:"
  echo "  video_dir: ${VIDEO_DIR}"
  echo "  run_name:  ${PA2_RUN_NAME}"
  echo "  csv:       ${PA2_CSV}"
  VIDEO_DIR="${VIDEO_DIR}" \
  OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
  RUN_NAME="${PA2_RUN_NAME}" \
  CHECKPOINT="${VIDEOPHY_CHECKPOINT}" \
  GPU_IDS="${PA2_GPU_IDS}" \
  BATCH_SIZE="${PA2_BATCH_SIZE}" \
    "${EVEWORLD_ROOT}/benchmarks/pbench/launch_dreamgenbench_videophy_pa2_kjob.sh"
}

wait_for_pa2() {
  if [[ "${WAIT_PA2}" != "1" ]]; then
    echo "WAIT_PA2=0, not waiting. Monitor:"
    echo "  tail -f ${PA2_LOG}"
    return
  fi

  echo "Waiting for PA-II to finish: ${PA2_CSV}"
  local waited=0
  while true; do
    check_pa2_failure
    local rows
    rows="$(csv_row_count "${PA2_CSV}")"
    if [[ "${rows}" == "${EXPECTED_COUNT}" ]]; then
      echo "PA-II complete: ${rows}/${EXPECTED_COUNT}"
      break
    fi
    if [[ "${PA2_TIMEOUT_SEC}" != "0" && "${waited}" -ge "${PA2_TIMEOUT_SEC}" ]]; then
      echo "Timed out waiting for PA-II after ${waited}s. Current rows: ${rows}/${EXPECTED_COUNT}" >&2
      echo "Check log: ${PA2_LOG}" >&2
      exit 1
    fi
    echo "  PA-II rows: ${rows}/${EXPECTED_COUNT}; sleeping ${POLL_SEC}s"
    sleep "${POLL_SEC}"
    waited=$((waited + POLL_SEC))
  done
}

summarize_pa2() {
  require_file "${PA2_CSV}"
  "${PYTHON_BIN}" - "${PA2_CSV}" "${PA2_GPU_PEAK}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
peak_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
pred = [int(float(r["prediction"])) for r in rows]
raw = [float(r["raw_score"]) for r in rows]
print("PA-II summary")
print(f"  csv:        {csv_path}")
print(f"  count:      {len(rows)}")
print(f"  positives:  {sum(pred)}")
print(f"  score:      {sum(pred) / len(pred):.6f}" if pred else "  score:      nan")
print(f"  mean_raw:   {statistics.mean(raw):.6f}" if raw else "  mean_raw:   nan")
print(f"  median_raw: {statistics.median(raw):.6f}" if raw else "  median_raw: nan")
print(f"  min_raw:    {min(raw):.6f}" if raw else "  min_raw:    nan")
print(f"  max_raw:    {max(raw):.6f}" if raw else "  max_raw:    nan")
if peak_path.exists() and peak_path.stat().st_size > 0:
    peak = json.loads(peak_path.read_text(encoding="utf-8"))
    print(f"  gpu_peak:   {peak.get('peak_memory_used_gib')} GiB on GPU {peak.get('peak_gpu_index')} ({peak.get('peak_gpu_name')})")
else:
    print(f"  gpu_peak:   missing ({peak_path})")
PY
}

print_plan() {
  cat <<EOF
PhysLatent full92 PA-II plan
  repo:              ${REPO_DIR}
  source run:        ${SOURCE_RUN_NAME}
  source videos:     ${SOURCE_VIDEO_DIR}
  generated-only:    ${VIDEO_DIR}
  expected count:    ${EXPECTED_COUNT}
  eval output root:  ${EVAL_OUTPUT_ROOT}
  PA-II run name:    ${PA2_RUN_NAME}
  PA-II csv:         ${PA2_CSV}
  PA-II raw csv:     ${PA2_RAW_CSV}
  PA-II log:         ${PA2_LOG}
  PA-II GPU ids:     ${PA2_GPU_IDS}
  PA-II batch size:  ${PA2_BATCH_SIZE}
  wait PA-II:        ${WAIT_PA2}
  PA-II timeout sec: ${PA2_TIMEOUT_SEC}
  force prepare:     ${FORCE_PREPARE}
  force PA-II:       ${FORCE_PA2}
EOF
}

main() {
  print_plan
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, stopping before changes/submission."
    return
  fi

  require_dir "${SOURCE_VIDEO_DIR}"
  verify_generation_summary
  local source_count
  source_count="$(mp4_count "${SOURCE_VIDEO_DIR}")"
  if [[ "${source_count}" != "${EXPECTED_COUNT}" ]]; then
    echo "Source side-by-side mp4 count mismatch: ${source_count}/${EXPECTED_COUNT}" >&2
    exit 1
  fi
  echo "Source side-by-side mp4 count ok: ${source_count}/${EXPECTED_COUNT}"

  prepare_generated_only
  echo "Generated-only mp4 count ok: $(mp4_count "${VIDEO_DIR}")/${EXPECTED_COUNT}"

  submit_pa2
  wait_for_pa2
  if pa2_done; then
    summarize_pa2
  fi
}

main "$@"
