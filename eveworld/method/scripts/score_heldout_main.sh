#!/usr/bin/env bash
# Score merged held-out videos with Judge B/A and paired TEA statistics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

SELECTED_CHECKPOINTS="${SELECTED_CHECKPOINTS:?SELECTED_CHECKPOINTS is required}"
GEN_ROOT="${GEN_ROOT:?GEN_ROOT is required}"
GENERATION_SEEDS="${GENERATION_SEEDS:-6666 1234}"
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
PYTHON="${PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
SCORE_ROOT="${SCORE_ROOT:-${GEN_ROOT}/scores}"
RUN_JUDGE_A="${RUN_JUDGE_A:-1}"
METHODS="${METHODS:-joint_lora frontier_only eve}"
TRAINING_SEEDS="${TRAINING_SEEDS:-20260716 20260717 20260718}"
mkdir -p "${SCORE_ROOT}/qwen" "${SCORE_ROOT}/tea" "${SCORE_ROOT}/comparisons"

read -r -a REQUIRED_METHOD_ARRAY <<< "${METHODS}"
read -r -a REQUIRED_SEED_ARRAY <<< "${TRAINING_SEEDS}"
"${PYTHON}" eveworld/method/scripts/validate_selected_checkpoints.py \
  --selected "${SELECTED_CHECKPOINTS}" \
  --methods "${REQUIRED_METHOD_ARRAY[@]}" \
  --training-seeds "${REQUIRED_SEED_ARRAY[@]}"

score_one() {
  local method="$1" training_seed="$2" generation_seed="$3"
  local run_name="${method}_train${training_seed}_gen${generation_seed}"
  local video_dir="${GEN_ROOT}/${method}/train_seed_${training_seed}/gen_seed_${generation_seed}/generated_only"
  [[ -d "${video_dir}" ]] || { echo "Missing generated videos: ${video_dir}" >&2; exit 1; }
  "${PYTHON}" eveworld/evaluation/tea/qwen_laziness.py \
    --video-dir "${video_dir}" --out-root "${SCORE_ROOT}/qwen" \
    --run-name "${run_name}_B" --qwen-base "${QWEN_BASE}" \
    --judge b --frame-offset 0.5 --concurrency "${QWEN_CONCURRENCY:-64}"
  if [[ "${RUN_JUDGE_A}" == "1" ]]; then
    "${PYTHON}" eveworld/evaluation/tea/qwen_laziness.py \
      --video-dir "${video_dir}" --out-root "${SCORE_ROOT}/qwen" \
      --run-name "${run_name}_A" --qwen-base "${QWEN_BASE}" \
      --judge a --frame-offset 0.0 --concurrency "${QWEN_CONCURRENCY:-64}"
  fi
  "${PYTHON}" eveworld/evaluation/tea/ncm.py score \
    --video-dir "${video_dir}" \
    --out "${SCORE_ROOT}/tea/${run_name}.json" --tag "${run_name}"
}

compare_pair() {
  local baseline_method="$1" baseline_training_seed="$2"
  local method="$3" method_training_seed="$4" label="$5"
  local baseline_b=() method_b=()
  for generation_seed in ${GENERATION_SEEDS}; do
    baseline_b+=("${SCORE_ROOT}/qwen/${baseline_method}_train${baseline_training_seed}_gen${generation_seed}_B_laziness.csv")
    method_b+=("${SCORE_ROOT}/qwen/${method}_train${method_training_seed}_gen${generation_seed}_B_laziness.csv")
    "${PYTHON}" eveworld/evaluation/tea/compare_ncm.py \
      --baseline "${SCORE_ROOT}/tea/${baseline_method}_train${baseline_training_seed}_gen${generation_seed}.json" \
      --method "${SCORE_ROOT}/tea/${method}_train${method_training_seed}_gen${generation_seed}.json" \
      --out "${SCORE_ROOT}/comparisons/${label}_gen${generation_seed}_tea.json"
  done
  "${PYTHON}" eveworld/evaluation/tea/compare_qwen.py \
    --baseline "${baseline_b[@]}" --method "${method_b[@]}" \
    --out "${SCORE_ROOT}/comparisons/${label}_pooled_B.json"
}

aggregate_pair() {
  local baseline_method="$1" baseline_training_seeds="$2"
  local method="$3" method_training_seeds="$4" label="$5"
  local command=("${PYTHON}" eveworld/evaluation/tea/compare_qwen_training_seeds.py)
  local baseline_training_seed method_training_seed generation_seed
  for baseline_training_seed in ${baseline_training_seeds}; do
    local baseline_group=()
    for generation_seed in ${GENERATION_SEEDS}; do
      baseline_group+=("${SCORE_ROOT}/qwen/${baseline_method}_train${baseline_training_seed}_gen${generation_seed}_B_laziness.csv")
    done
    command+=(--baseline-group "${baseline_group[@]}")
  done
  for method_training_seed in ${method_training_seeds}; do
    local method_group=()
    for generation_seed in ${GENERATION_SEEDS}; do
      method_group+=("${SCORE_ROOT}/qwen/${method}_train${method_training_seed}_gen${generation_seed}_B_laziness.csv")
    done
    command+=(--method-group "${method_group[@]}")
  done
  command+=(--out "${SCORE_ROOT}/comparisons/${label}_all_training_seeds_B.json")
  "${command[@]}"
}

for generation_seed in ${GENERATION_SEEDS}; do
  score_one joint_base base "${generation_seed}"
done

while IFS=$'\t' read -r method pipeline training_seed checkpoint_step lora_path; do
  [[ "${method}" == "method" || -z "${method}" ]] && continue
  for generation_seed in ${GENERATION_SEEDS}; do
    score_one "${method}" "${training_seed}" "${generation_seed}"
  done

  base_b=()
  method_b=()
  for generation_seed in ${GENERATION_SEEDS}; do
    base_b+=("${SCORE_ROOT}/qwen/joint_base_trainbase_gen${generation_seed}_B_laziness.csv")
    method_b+=("${SCORE_ROOT}/qwen/${method}_train${training_seed}_gen${generation_seed}_B_laziness.csv")
    "${PYTHON}" eveworld/evaluation/tea/compare_ncm.py \
      --baseline "${SCORE_ROOT}/tea/joint_base_trainbase_gen${generation_seed}.json" \
      --method "${SCORE_ROOT}/tea/${method}_train${training_seed}_gen${generation_seed}.json" \
      --out "${SCORE_ROOT}/comparisons/${method}_train${training_seed}_gen${generation_seed}_tea.json"
  done
  "${PYTHON}" eveworld/evaluation/tea/compare_qwen.py \
    --baseline "${base_b[@]}" --method "${method_b[@]}" \
    --out "${SCORE_ROOT}/comparisons/${method}_train${training_seed}_pooled_B.json"
done < "${SELECTED_CHECKPOINTS}"

for training_seed in ${TRAINING_SEEDS}; do
  compare_pair joint_lora "${training_seed}" frontier_only "${training_seed}" \
    "frontier_only_vs_joint_lora_train${training_seed}"
  compare_pair frontier_only "${training_seed}" eve "${training_seed}" \
    "eve_vs_frontier_only_train${training_seed}"
  compare_pair joint_lora "${training_seed}" eve "${training_seed}" \
    "eve_vs_joint_lora_train${training_seed}"
done

aggregate_pair joint_base "base" joint_lora "${TRAINING_SEEDS}" "joint_lora_vs_joint_base"
aggregate_pair joint_base "base" frontier_only "${TRAINING_SEEDS}" "frontier_only_vs_joint_base"
aggregate_pair joint_base "base" eve "${TRAINING_SEEDS}" "eve_vs_joint_base"
aggregate_pair joint_lora "${TRAINING_SEEDS}" frontier_only "${TRAINING_SEEDS}" \
  "frontier_only_vs_joint_lora"
aggregate_pair frontier_only "${TRAINING_SEEDS}" eve "${TRAINING_SEEDS}" \
  "eve_vs_frontier_only"

echo "[heldout-score] results: ${SCORE_ROOT}"
