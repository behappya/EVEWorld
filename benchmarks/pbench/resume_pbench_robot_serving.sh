#!/usr/bin/env bash
set -euo pipefail

# Resume the PBench robot full run against an already-running GigaWorld-0
# HTTP service. This script does not load model weights locally.
#
# Common overrides:
#   URL=http://host:8000 SAVE_DIR=/path/to/run ./benchmarks/pbench/resume_pbench_robot_serving.sh
#   OFFSET=112 ./benchmarks/pbench/resume_pbench_robot_serving.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

URL="${URL:-http://127.0.0.1:8000}"
SAVE_DIR="${SAVE_DIR:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_serving_full_20260612_145541}"
DATA_PATH="${DATA_PATH:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json}"

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"

LIMIT="${LIMIT:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
NUM_FRAMES="${NUM_FRAMES:-61}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"
SEED="${SEED:-6666}"
TIMEOUT="${TIMEOUT:-7200}"

RESULTS_FILE="${SAVE_DIR}/call_results.jsonl"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing DATA_PATH: ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"
touch "${RESULTS_FILE}"

if [[ -z "${OFFSET:-}" ]]; then
  OFFSET="$(wc -l < "${RESULTS_FILE}")"
fi

CLIENT_LOG="${CLIENT_LOG:-${SAVE_DIR}/client_resume_$(date +%Y%m%d_%H%M%S).log}"

cd "${REPO_DIR}"

echo "============================================"
echo "GigaWorld-0 PBench robot serving resume"
echo "Repo:                    ${REPO_DIR}"
echo "URL:                     ${URL}"
echo "Data path:               ${DATA_PATH}"
echo "Save dir:                ${SAVE_DIR}"
echo "Results file:            ${RESULTS_FILE}"
echo "Client log:              ${CLIENT_LOG}"
echo "Offset:                  ${OFFSET}"
echo "Limit:                   ${LIMIT}"
echo "Steps/FPS/Frames:        ${NUM_INFERENCE_STEPS}/${FPS}/${NUM_FRAMES}"
echo "Height/Width:            ${HEIGHT}/${WIDTH}"
echo "Seed:                    ${SEED}"
echo "Timeout:                 ${TIMEOUT}"
echo "============================================"

echo "==> Health check"
HTTP_CODE="$(
  curl -sS -o /tmp/gw0_health_$$.json -w "%{http_code}" "${URL%/}/health" || true
)"
cat /tmp/gw0_health_$$.json || true
rm -f /tmp/gw0_health_$$.json
echo
echo "HTTP ${HTTP_CODE}"

if [[ "${HTTP_CODE}" != "200" ]]; then
  echo "Service is not healthy. Start or fix the serving job before resuming." >&2
  exit 1
fi

echo "==> Current counts before resume"
echo -n "jsonl lines: "
wc -l < "${RESULTS_FILE}"
echo -n "mp4 count:   "
find "${SAVE_DIR}" -maxdepth 1 -name 'robot_*.mp4' | wc -l

echo "==> Resuming"
python scripts/call_gigaworld0_serve.py \
  --url "${URL}" \
  --data-path "${DATA_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --fps "${FPS}" \
  --num-frames "${NUM_FRAMES}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --seed "${SEED}" \
  --timeout "${TIMEOUT}" 2>&1 | tee -a "${CLIENT_LOG}"

echo "==> Counts after resume"
echo -n "jsonl lines: "
wc -l < "${RESULTS_FILE}"
echo -n "mp4 count:   "
find "${SAVE_DIR}" -maxdepth 1 -name 'robot_*.mp4' | wc -l
echo "client log: ${CLIENT_LOG}"
