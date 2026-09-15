#!/usr/bin/env bash
set -euo pipefail

# Standalone Qwen Domain/VQA eval for Pretrain PBench 19.8s.
# Uses Qwen3.6-VL endpoint, 32000 max tokens, thinking enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO_DIR="${VIDEO_DIR:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_length_sweep/pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu}"
EVAL_DIR="${EVAL_DIR:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval/pbench_robot_19p8s_full_pbench_len_sweep_full_8gpu_domain_qwen36vl_58_think32000}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"

QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-auto}"
CONCURRENCY="${CONCURRENCY:-100}"
MAX_INFLIGHT="${MAX_INFLIGHT:-100}"
FRAME_COUNT="${FRAME_COUNT:-8}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-512}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-32000}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-600}"
MODEL_RETRIES="${MODEL_RETRIES:-3}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"

cd "${REPO_DIR}"

echo "Pretrain PBench 19.8s Qwen 32000+thinking"
echo "Video dir:     ${VIDEO_DIR}"
echo "Eval dir:      ${EVAL_DIR}"
echo "Qwen base:     ${QWEN_BASE}"
echo "Concurrency:   ${CONCURRENCY}/${MAX_INFLIGHT}"
echo "Max tokens:    ${MODEL_MAX_TOKENS}"
echo "Thinking:      $([[ "${DISABLE_THINKING}" == "0" ]] && echo enabled || echo disabled)"

if [[ "${STATUS_ONLY}" == "1" ]]; then
  find "${VIDEO_DIR}" -maxdepth 1 -name 'robot_*.mp4' | wc -l
  [[ -f "${EVAL_DIR}/qwen_vqa_summary.json" ]] && cat "${EVAL_DIR}/qwen_vqa_summary.json" || true
  exit 0
fi

VIDEO_DIR="${VIDEO_DIR}" \
EVAL_DIR="${EVAL_DIR}" \
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

