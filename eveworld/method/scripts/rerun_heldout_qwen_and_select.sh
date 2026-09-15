#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

RUN_TAG="${RUN_TAG:-eve_heldout_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/eve/heldout_main}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
QWEN_ROOT="${QWEN_ROOT:-${RUN_ROOT}/validation/scores/qwen_28}"
EVAL_PYTHON="${EVAL_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
GENERATION_SEEDS="${GENERATION_SEEDS:-6666 1234}"
VAL_COUNT="${VAL_COUNT:-20}"

curl -fsS "${QWEN_BASE%/}/models" >/dev/null
mkdir -p "${QWEN_ROOT}"

tail -n +2 "${RUN_ROOT}/checkpoint_candidates.tsv" |
while IFS=$'\t' read -r method pipeline training_seed checkpoint_step lora_path; do
  for generation_seed in ${GENERATION_SEEDS}; do
    "${EVAL_PYTHON}" eveworld/evaluation/tea/qwen_laziness.py \
      --video-dir "${RUN_ROOT}/validation/${method}_step${checkpoint_step}/train_seed_${training_seed}/gen_seed_${generation_seed}/generated_only" \
      --out-root "${QWEN_ROOT}" \
      --run-name "${method}_train${training_seed}_step${checkpoint_step}_gen${generation_seed}_B" \
      --qwen-base "${QWEN_BASE}" \
      --judge b --frame-offset 0.5 --concurrency "${QWEN_CONCURRENCY:-64}" \
      --rerun-errors
  done
done

"${EVAL_PYTHON}" eveworld/method/scripts/select_validation_checkpoints.py \
  --candidates "${RUN_ROOT}/checkpoint_candidates.tsv" \
  --score-root "${QWEN_ROOT}" \
  --generation-seeds ${GENERATION_SEEDS} \
  --expected-per-seed "${VAL_COUNT}" \
  --out "${RUN_ROOT}/selected_checkpoints.tsv"

echo "[heldout-qwen] selected checkpoints: ${RUN_ROOT}/selected_checkpoints.tsv"
