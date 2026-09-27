#!/usr/bin/env bash
#
# Table 1 (tab:dreamgen_overall): the DreamGenBench overall comparison.
#
# Stages, in order:
#   1. build the DreamGen splits and the per-clip metadata under data/
#   2. fine-tune the two GigaWorld-0 arms this repository owns, SFT for 200 steps and
#      EVEWorld for 250 steps, at the Table 1 recipe
#   3. generate the 126 evaluation prompts once per arm, 30 denoising steps at CFG 7.0
#   4. score every arm with the MLR protocol, then with the Qwen and the Gemini judge
#
# The released GigaWorld-0 backbone and the four external baselines of the paper
# table are third-party outputs; the wrapper drives the two arms it trains and
# reads their checkpoints from ${EVEWORLD_RUN_ROOT}. Point CKPT_SFT or
# CKPT_EVEWORLD at another checkpoint directory to score an existing run.
#
# Usage:
#   bash scripts/reproduce/table1_dreamgen.sh
#
# Environment:
#   PYTHON                 interpreter of the stages (default: python)
#   DREAMGEN_DATA_ROOT     DreamGen source tree, read by the preparation stage
#   EVEWORLD_RUN_ROOT      training runs and generated clips (default: outputs)
#   EVEWORLD_RESULTS_ROOT  item-level results (default: results)
#   SHARDS                 generation shards, run one after another (default: 1)
#   LIMIT                  cap the clips of every stage, for a smoke run
#   DRY_RUN                1 prints the commands of every stage instead of running them
#   CKPT_SFT, CKPT_EVEWORLD  checkpoint of one arm, over its run directory
#
# The two judge stages need credentials and are counted as failed when they do not
# finish; every other stage stops the wrapper. The MLR summary of each arm is
# written to ${EVEWORLD_RESULTS_ROOT}/item_level/dreamgen/mlr_<arm>.json.
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-python}"
RUN_ROOT="${EVEWORLD_RUN_ROOT:-outputs}"
RESULTS_ROOT="${EVEWORLD_RESULTS_ROOT:-results}"
SHARDS="${SHARDS:-1}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
FAILED=0

SPLIT="data/splits/dreamgen/test.txt"
METADATA="data/metadata/dreamgenbench"
ARMS=(sft eveworld)
declare -A RUN_STEPS=([sft]=200 [eveworld]=250)
declare -A CONFIG=(
    [sft]="configs/paper/gigaworld/dreamgen/sft.yaml"
    [eveworld]="configs/paper/gigaworld/dreamgen/eveworld.yaml"
)
declare -A CKPT=(
    [sft]="${CKPT_SFT:-${RUN_ROOT}/dreamgen_sft}"
    [eveworld]="${CKPT_EVEWORLD:-${RUN_ROOT}/dreamgen_eveworld}"
)

LIMIT_ARGS=()
if [ -n "${LIMIT}" ]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

run() {
    printf '+ %s\n' "$*"
    if [ "${DRY_RUN}" != "1" ]; then
        "$@"
    fi
}

soft() {
    printf '+ %s\n' "$*"
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    if ! "$@"; then
        printf 'failed: %s\n' "$*" >&2
        FAILED=$((FAILED + 1))
    fi
}

# 1. Data preparation.
if [ -f "${SPLIT}" ]; then
    echo "skip: ${SPLIT} exists"
else
    run "${PYTHON}" scripts/prepare/prepare_dreamgen.py --output-dir data "${LIMIT_ARGS[@]}"
fi

# 2. Post-training.
for arm in "${ARMS[@]}"; do
    run "${PYTHON}" scripts/train/train_gigaworld.py \
        --config "${CONFIG[${arm}]}" \
        --steps "${RUN_STEPS[${arm}]}" \
        --output-dir "${RUN_ROOT}/dreamgen_${arm}"
done

# 3. Generation, one shard after another.
for arm in "${ARMS[@]}"; do
    for index in $(seq 0 $((SHARDS - 1))); do
        run "${PYTHON}" scripts/inference/infer_gigaworld.py \
            --config "${CONFIG[${arm}]}" \
            --prompt-file "${SPLIT}" \
            --checkpoint "${CKPT[${arm}]}" \
            --output-dir "${RUN_ROOT}/dreamgen_${arm}" \
            --num-steps 30 --cfg-scale 7.0 \
            --num-shards "${SHARDS}" --shard-index "${index}"
    done
done

# 4. Metrics.
for arm in "${ARMS[@]}"; do
    run "${PYTHON}" scripts/evaluate/eval_mlr.py \
        --config configs/eval/mlr/dreamgen.yaml \
        --pred-dir "${RUN_ROOT}/dreamgen_${arm}/generated_only" \
        --metadata "${METADATA}" \
        --model "${arm}" \
        --output "${RESULTS_ROOT}/item_level/dreamgen/mlr_${arm}.json" \
        "${LIMIT_ARGS[@]}"
    soft "${PYTHON}" scripts/evaluate/eval_instruction_following.py \
        --config configs/eval/instruction_following/qwen_if.yaml \
        --judge qwen_if \
        --pred-dir "${RUN_ROOT}/dreamgen_${arm}/generated_only" \
        --model "${arm}" \
        --output "${RESULTS_ROOT}/item_level/dreamgen/qwen_if_${arm}.json" \
        "${LIMIT_ARGS[@]}"
    soft "${PYTHON}" scripts/evaluate/eval_instruction_following.py \
        --config configs/eval/instruction_following/gemini_if.yaml \
        --judge gemini_if \
        --pred-dir "${RUN_ROOT}/dreamgen_${arm}/generated_only" \
        --model "${arm}" \
        --output "${RESULTS_ROOT}/item_level/dreamgen/gemini_if_${arm}.json" \
        "${LIMIT_ARGS[@]}"
done

if [ "${FAILED}" -gt 0 ]; then
    printf '%d stage(s) failed; rerun them for a complete table\n' "${FAILED}" >&2
    exit 1
fi
echo "done: Table 1 summaries under ${RESULTS_ROOT}/item_level/dreamgen"
