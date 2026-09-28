#!/usr/bin/env bash
#
# Table 3 (tab:worldarena): WorldArena 1.0 zero-shot generation of the two
# DreamGenBench arms, with the eight core metrics and the MLR pass.
#
# Stages, in order:
#   1. build the WorldArena MLR metadata and splits under data/
#   2. generate the 1,000 manifest requests once per arm from the step-250
#      DreamGenBench checkpoint, 30 denoising steps at CFG 7.0
#   3. rebuild the eight core metrics and Overall from the WorldArena metric files
#   4. score both arms with the WorldArena MLR protocol
#
# This wrapper trains nothing: run scripts/reproduce/table1_dreamgen.sh first, or
# point CKPT_SFT and CKPT_EVEWORLD at two checkpoints of the same recipe.
#
# The eight core metrics are produced by the released WorldArena evaluator over the
# generated clips; the wrapper reads its files back with --scores when the layout is
# not the one eval_worldarena.py probes around --pred-dir.
#
# Usage:
#   bash scripts/reproduce/table3_worldarena.sh
#
# Environment:
#   PYTHON                 interpreter of the stages (default: python)
#   WORLDARENA_DATA_ROOT   WorldArena source tree, read by the preparation stage
#   EVEWORLD_RUN_ROOT      training runs and generated clips (default: outputs)
#   EVEWORLD_RESULTS_ROOT  item-level results (default: outputs/evaluation)
#   SHARDS                 generation shards, run one after another (default: 1)
#   LIMIT                  cap the requests and clips of every stage, for a smoke run
#   DRY_RUN                1 prints the commands of every stage instead of running them
#   CKPT_SFT, CKPT_EVEWORLD  checkpoint of one arm, over its DreamGenBench run directory
#   WORLDARENA_SCORES      evaluation root or model directory holding <model>/core/*.json
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

SPLIT="data/splits/worldarena/eval.txt"
METADATA="data/metadata/worldarena"
ARMS=(sft eveworld)
declare -A CONFIG=(
    [sft]="configs/paper/gigaworld/dreamgen/sft.yaml"
    [eveworld]="configs/paper/gigaworld/dreamgen/eveworld.yaml"
)
declare -A CKPT=(
    [sft]="${CKPT_SFT:-${RUN_ROOT}/dreamgen_sft}"
    [eveworld]="${CKPT_EVEWORLD:-${RUN_ROOT}/dreamgen_eveworld}"
)
declare -A TABLE=([sft]="table3_sft.json" [eveworld]="table3.json")

LIMIT_ARGS=()
if [ -n "${LIMIT}" ]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

SCORE_ARGS=()
if [ -n "${WORLDARENA_SCORES:-}" ]; then
    SCORE_ARGS=(--scores "${WORLDARENA_SCORES}")
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
    run "${PYTHON}" scripts/prepare/build_mlr_metadata.py --output-dir data "${LIMIT_ARGS[@]}"
fi

# 2. Zero-shot generation from the DreamGenBench checkpoints.
for arm in "${ARMS[@]}"; do
    if [ "${DRY_RUN}" != "1" ] && [ ! -e "${CKPT[${arm}]}" ]; then
        printf 'error: %s is missing; train it with scripts/reproduce/table1_dreamgen.sh\n' \
            "${CKPT[${arm}]}" >&2
        exit 1
    fi
    for index in $(seq 0 $((SHARDS - 1))); do
        run "${PYTHON}" scripts/inference/infer_gigaworld.py \
            --config "${CONFIG[${arm}]}" \
            --prompt-file "${SPLIT}" \
            --checkpoint "${CKPT[${arm}]}" \
            --output-dir "${RUN_ROOT}/worldarena_${arm}" \
            --num-steps 30 --cfg-scale 7.0 \
            --num-shards "${SHARDS}" --shard-index "${index}"
    done
done

# 3. WorldArena core metrics; the metric files are written by the released evaluator.
for arm in "${ARMS[@]}"; do
    soft "${PYTHON}" scripts/evaluate/eval_worldarena.py \
        --config configs/eval/worldarena.yaml \
        --pred-dir "${RUN_ROOT}/worldarena_${arm}/generated_only" \
        --model "${arm}" \
        --output "${RESULTS_ROOT}/item_level/worldarena/${TABLE[${arm}]}" \
        "${SCORE_ARGS[@]}" "${LIMIT_ARGS[@]}"
done

# 4. MLR over the frozen eligible set.
for arm in "${ARMS[@]}"; do
    run "${PYTHON}" scripts/evaluate/eval_mlr.py \
        --config configs/eval/mlr/worldarena.yaml \
        --pred-dir "${RUN_ROOT}/worldarena_${arm}/generated_only" \
        --metadata "${METADATA}" \
        --model "${arm}" \
        --output "${RESULTS_ROOT}/item_level/worldarena/mlr_${arm}.json" \
        "${LIMIT_ARGS[@]}"
done

if [ "${FAILED}" -gt 0 ]; then
    printf '%d stage(s) failed; rerun them for a complete table\n' "${FAILED}" >&2
    exit 1
fi
echo "done: Table 3 summaries under ${RESULTS_ROOT}/item_level/worldarena"
