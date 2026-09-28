#!/usr/bin/env bash
#
# Table 6 (tab:component_ablation): the DreamGenBench component ablation.
#
# Stages, in order:
#   1. build the DreamGen splits and the per-clip metadata under data/
#   2. fine-tune the four arms at the shared 250-step budget: standard SFT,
#      IGR only, TIA only and the joint EVEWorld arm
#   3. generate the 126 evaluation prompts once per arm, 30 denoising steps at CFG 7.0
#   4. judge every arm with Gemini and score it with the MLR protocol
#
# The joint arm and the standard baseline are the two runs of Table 1, so a run of
# scripts/reproduce/table1_dreamgen.sh already trained them; the wrapper reuses the
# same output directories and the same CKPT_* overrides.
#
# Usage:
#   bash scripts/reproduce/table6_ablation.sh
#
# Environment:
#   PYTHON                 interpreter of the stages (default: python)
#   DREAMGEN_DATA_ROOT     DreamGen source tree, read by the preparation stage
#   EVEWORLD_RUN_ROOT      training runs and generated clips (default: outputs)
#   EVEWORLD_RESULTS_ROOT  item-level results (default: outputs/evaluation)
#   SHARDS                 generation shards, run one after another (default: 1)
#   LIMIT                  cap the clips of every stage, for a smoke run
#   DRY_RUN                1 prints the commands of every stage instead of running them
#   CKPT_SFT, CKPT_IGR, CKPT_TIA, CKPT_EVEWORLD  checkpoint of one arm
#
# The Gemini judge needs credentials and is counted as failed when it does not
# finish; the MLR stage and the training stages stop the wrapper.
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-python}"
RUN_ROOT="${EVEWORLD_RUN_ROOT:-outputs}"
RESULTS_ROOT="${EVEWORLD_RESULTS_ROOT:-outputs/evaluation}"
SHARDS="${SHARDS:-1}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
FAILED=0

SPLIT="data/splits/dreamgen/test.txt"
METADATA="data/metadata/dreamgenbench"
ARMS=(sft igr tia eveworld)
declare -A RUN_STEPS=([sft]=250 [igr]=250 [tia]=250 [eveworld]=250)
declare -A CONFIG=(
    [sft]="configs/paper/gigaworld/dreamgen/sft.yaml"
    [igr]="configs/paper/gigaworld/dreamgen/igr.yaml"
    [tia]="configs/paper/gigaworld/dreamgen/tia.yaml"
    [eveworld]="configs/paper/gigaworld/dreamgen/eveworld.yaml"
)
declare -A CKPT=(
    [sft]="${CKPT_SFT:-${RUN_ROOT}/dreamgen_sft}"
    [igr]="${CKPT_IGR:-${RUN_ROOT}/dreamgen_igr}"
    [tia]="${CKPT_TIA:-${RUN_ROOT}/dreamgen_tia}"
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

# 2. Post-training of the four arms.
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

# 4. Gemini judge and MLR per arm.
for arm in "${ARMS[@]}"; do
    soft "${PYTHON}" scripts/evaluate/eval_instruction_following.py \
        --config configs/eval/instruction_following/gemini_if.yaml \
        --judge gemini_if \
        --pred-dir "${RUN_ROOT}/dreamgen_${arm}/generated_only" \
        --model "${arm}" \
        --output "${RESULTS_ROOT}/item_level/dreamgen/gemini_if_${arm}.json" \
        "${LIMIT_ARGS[@]}"
    run "${PYTHON}" scripts/evaluate/eval_mlr.py \
        --config configs/eval/mlr/dreamgen.yaml \
        --pred-dir "${RUN_ROOT}/dreamgen_${arm}/generated_only" \
        --metadata "${METADATA}" \
        --model "${arm}" \
        --output "${RESULTS_ROOT}/item_level/dreamgen/mlr_${arm}.json" \
        "${LIMIT_ARGS[@]}"
done

if [ "${FAILED}" -gt 0 ]; then
    printf '%d stage(s) failed; rerun them for a complete table\n' "${FAILED}" >&2
    exit 1
fi
echo "done: Table 6 summaries under ${RESULTS_ROOT}/item_level/dreamgen"
