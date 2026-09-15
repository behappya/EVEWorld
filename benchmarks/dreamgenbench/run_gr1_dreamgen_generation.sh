#!/usr/bin/env bash
set -euo pipefail

# Call an already-running GigaWorld-0 HTTP service to generate GR1 videos
# for DreamGenBench-style evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"

if [[ -z "${URL:-}" ]]; then
  echo "Set URL to the running service, e.g. URL=http://127.0.0.1:8000" >&2
  exit 1
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval}"
DATA_PATH="${DATA_PATH:-${EVAL_ROOT}/giga_input/gr1_dreamgen_it2v.json}"
SAVE_DIR="${SAVE_DIR:-${EVAL_ROOT}/generated_side_by_side/${TIMESTAMP}_gr1_finetuned}"
SUMMARY_PATH="${SUMMARY_PATH:-${SAVE_DIR}/call_summary.json}"
LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
NUM_FRAMES="${NUM_FRAMES:-93}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
SEED="${SEED:-6666}"
TIMEOUT="${TIMEOUT:-7200}"

activate_giga_env() {
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  fi
  if ! command -v python >/dev/null 2>&1; then
    echo "python not found. Set CONDA_SH/CONDA_ENV or run from an environment with python." >&2
    exit 1
  fi
}

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing ${DATA_PATH}; preparing GR1 DreamGen inputs first." >&2
  activate_giga_env
  python "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/prepare_gr1_dreamgen_inputs.py" \
    --output-root "${EVAL_ROOT}"
fi

mkdir -p "${SAVE_DIR}"
echo "GigaWorld-0 GR1 DreamGen generation"
echo "URL:        ${URL}"
echo "Data path:  ${DATA_PATH}"
echo "Save dir:   ${SAVE_DIR}"
echo "Summary:    ${SUMMARY_PATH}"
echo "Offset:     ${OFFSET}"
echo "Limit:      ${LIMIT}"
echo "Steps:      ${NUM_INFERENCE_STEPS}"
echo "Frames:     ${NUM_FRAMES}"
echo "FPS:        ${FPS}"
echo "Size:       ${HEIGHT}x${WIDTH}"
echo "Seed:       ${SEED}"

activate_giga_env
cd "${REPO_DIR}"

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
  --timeout "${TIMEOUT}" \
  --summary-path "${SUMMARY_PATH}"

echo
echo "Side-by-side videos: ${SAVE_DIR}"
echo "Convert to generated-only videos with:"
echo "  SOURCE_VIDEO_DIR=${SAVE_DIR} ./benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py"
