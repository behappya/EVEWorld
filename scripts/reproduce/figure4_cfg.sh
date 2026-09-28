#!/usr/bin/env bash
#
# Figure 4 (fig:cfg): the classifier-free guidance grid.
#
# Stages, in order:
#   1. build the DreamGen splits and the per-clip metadata under data/
#   2. train the grid run for 300 steps, one checkpoint every 50 steps
#   3. stage one checkpoint directory per grid step below ${GRID_ROOT}
#   4. generate the 126 evaluation prompts for every cell of the six training steps
#      times the four guidance weights, at generation seed 4 and 30 denoising steps
#   5. score every cell with Gemini and with the MLR protocol
#
# The staged directory of a step holds a link to the checkpoint-<step>.pt of the run
# and is passed as --checkpoint, so the generated tree keeps the
# step_%03d/cfg_<tag>/generated_only layout that the Figure 4 aggregation reads. The
# tag replaces the decimal point of the guidance weight with p. Override
# CKPT_STEP_<step> when the weights of a step are loaded from somewhere else.
#
# The wrapper creates the staging directories of the grid itself; every directory
# below them is created by the generator that writes into it.
#
# Usage:
#   bash scripts/reproduce/figure4_cfg.sh
#
# Environment:
#   PYTHON                 interpreter of the stages (default: python)
#   DREAMGEN_DATA_ROOT     DreamGen source tree, read by the preparation stage
#   EVEWORLD_RUN_ROOT      training runs and generated clips (default: outputs)
#   EVEWORLD_RESULTS_ROOT  item-level results (default: outputs/evaluation)
#   GRID_ROOT              output tree of the grid (default: <grid run>/grid)
#   SHARDS                 generation shards per cell, run one after another (default: 1)
#   LIMIT                  cap the clips of every stage, for a smoke run
#   DRY_RUN                1 prints the commands of every stage instead of running them
#   CKPT_STEP_50 ... CKPT_STEP_300  checkpoint directory of one grid step
#
# The Gemini judge needs credentials and is counted as failed when it does not
# finish; every other stage stops the wrapper.
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
GRID_CONFIG="configs/ablations/cfg/grid.yaml"
RUN_DIR="${RUN_ROOT}/ablate_cfg_grid"
GRID_ROOT="${GRID_ROOT:-${RUN_DIR}/grid}"
STEPS=(50 100 150 200 250 300)
CFG_VALUES=(1.0 2.5 5.0 7.0)

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

cfg_tag() {
    printf 'cfg_%sp%s' "${1%%.*}" "${1#*.}"
}

# 1. Data preparation.
if [ -f "${SPLIT}" ]; then
    echo "skip: ${SPLIT} exists"
else
    run "${PYTHON}" scripts/prepare/prepare_dreamgen.py --output-dir data "${LIMIT_ARGS[@]}"
fi

# 2. The grid run.
run "${PYTHON}" scripts/train/train_gigaworld.py \
    --config "${GRID_CONFIG}" \
    --steps 300 \
    --output-dir "${RUN_DIR}"

# 3. One staged checkpoint directory per grid step.
for step in "${STEPS[@]}"; do
    cell="$(printf 'step_%03d' "${step}")"
    source_file="$(printf '%s/checkpoint-%06d.pt' "${RUN_DIR}" "${step}")"
    staged_dir="${GRID_ROOT}/${cell}/checkpoint"
    staged_file="$(printf '%s/checkpoint-%06d.pt' "${staged_dir}" "${step}")"
    if [ "${DRY_RUN}" = "1" ]; then
        printf '+ mkdir -p %s\n' "${staged_dir}"
        printf '+ ln -sfn %s %s\n' "${source_file}" "${staged_file}"
        continue
    fi
    if [ ! -f "${source_file}" ]; then
        printf 'error: %s is missing; the grid run did not reach step %d\n' \
            "${source_file}" "${step}" >&2
        exit 1
    fi
    mkdir -p "${staged_dir}"
    ln -sfn "${source_file}" "${staged_file}"
done

# 4. The 24 cells of the grid.
for step in "${STEPS[@]}"; do
    cell="$(printf 'step_%03d' "${step}")"
    step_var="CKPT_STEP_${step}"
    cell_ckpt="${!step_var:-${GRID_ROOT}/${cell}/checkpoint}"
    for cfg in "${CFG_VALUES[@]}"; do
        tag="$(cfg_tag "${cfg}")"
        for index in $(seq 0 $((SHARDS - 1))); do
            run "${PYTHON}" scripts/inference/infer_gigaworld.py \
                --config "${GRID_CONFIG}" \
                --prompt-file "${SPLIT}" \
                --checkpoint "${cell_ckpt}" \
                --output-dir "${GRID_ROOT}/${cell}/${tag}" \
                --num-steps 30 --cfg-scale "${cfg}" --seed 4 \
                --num-shards "${SHARDS}" --shard-index "${index}"
        done
    done
done

# 5. MLR and the Gemini judge of every cell.
for step in "${STEPS[@]}"; do
    cell="$(printf 'step_%03d' "${step}")"
    for cfg in "${CFG_VALUES[@]}"; do
        tag="$(cfg_tag "${cfg}")"
        run "${PYTHON}" scripts/evaluate/eval_mlr.py \
            --config configs/eval/mlr/dreamgen.yaml \
            --pred-dir "${GRID_ROOT}/${cell}/${tag}/generated_only" \
            --metadata "${METADATA}" \
            --model "${cell}_${tag}" \
            --output "${RESULTS_ROOT}/item_level/dreamgen/mlr_fig4_${cell}_${tag}.json" \
            "${LIMIT_ARGS[@]}"
        soft "${PYTHON}" scripts/evaluate/eval_instruction_following.py \
            --config configs/eval/instruction_following/gemini_if.yaml \
            --judge gemini_if \
            --pred-dir "${GRID_ROOT}/${cell}/${tag}/generated_only" \
            --model "${cell}_${tag}" \
            --output "${RESULTS_ROOT}/item_level/dreamgen/gemini_if_fig4_${cell}_${tag}.json" \
            "${LIMIT_ARGS[@]}"
    done
done

if [ "${FAILED}" -gt 0 ]; then
    printf '%d stage(s) failed; rerun them for a complete figure\n' "${FAILED}" >&2
    exit 1
fi
echo "done: Figure 4 cells under ${GRID_ROOT} and summaries under ${RESULTS_ROOT}/item_level/dreamgen"
