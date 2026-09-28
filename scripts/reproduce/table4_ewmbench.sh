#!/usr/bin/env bash
#
# Table 4 (tab:agibot_transfer): the AgiBot transfer experiment scored with EWMBench.
#
# Stages, in order:
#   1. build the AgiBot split and the per-clip metadata under data/
#   2. fine-tune the EVEWorld arm on AgiBot for 50 steps at 640 x 480
#   3. generate the 777 clips three times, one run per generation seed 42, 43 and 44
#   4. score the first repeat with the MLR protocol
#   5. rebuild the Table 4 columns from the official EWMBench result CSV
#
# The official EWMBench scorer reads its own layout, eval_layout/<model>_dataset,
# which generation does not write: the wrapper generates <request_id>.mp4 clips
# under ${EVEWORLD_RUN_ROOT}/agibot_eveworld/seed_<seed>/generated_only and expects
# the scorer to have filled that tree. Set EVAL_LAYOUT to another root and
# EWMBENCH_SCORES to the result CSV when the scoring runs elsewhere.
#
# Usage:
#   bash scripts/reproduce/table4_ewmbench.sh
#
# Environment:
#   PYTHON                 interpreter of the stages (default: python)
#   AGIBOT_DATA_ROOT       AgiBot source clips, read by the preparation stage
#   EWMBENCH_DATA_ROOT     EWMBench checkout holding the GT/ episodes
#   EVEWORLD_RUN_ROOT      training runs and generated clips (default: outputs)
#   EVEWORLD_RESULTS_ROOT  item-level results (default: outputs/evaluation)
#   SHARDS                 generation shards, run one after another (default: 1)
#   LIMIT                  cap the clips of every stage, for a smoke run
#   DRY_RUN                1 prints the commands of every stage instead of running them
#   CKPT_AGIBOT            AgiBot checkpoint, over the ${EVEWORLD_RUN_ROOT} run
#   EVAL_LAYOUT            EWMBench layout root (default: eval_layout)
#   EWMBENCH_SCORES        official result CSV, or the directory holding it
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

SPLIT="data/splits/agibot/train.txt"
METADATA="data/metadata/agibot"
CONFIG="configs/paper/gigaworld/agibot/eveworld.yaml"
EVAL_CONFIG="configs/eval/ewmbench.yaml"
SEEDS=(42 43 44)
CKPT="${CKPT_AGIBOT:-${RUN_ROOT}/agibot_eveworld}"

LIMIT_ARGS=()
if [ -n "${LIMIT}" ]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

SCORE_ARGS=()
if [ -n "${EWMBENCH_SCORES:-}" ]; then
    SCORE_ARGS=(--scores "${EWMBENCH_SCORES}")
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
    run "${PYTHON}" scripts/prepare/prepare_agibot.py --output-dir data "${LIMIT_ARGS[@]}"
fi

# 2. Post-training, 50 steps, the Table 4 recipe.
run "${PYTHON}" scripts/train/train_gigaworld.py \
    --config "${CONFIG}" \
    --steps 50 \
    --output-dir "${RUN_ROOT}/agibot_eveworld"

# 3. Generation, three repeats at the frozen seeds.
for seed in "${SEEDS[@]}"; do
    for index in $(seq 0 $((SHARDS - 1))); do
        run "${PYTHON}" scripts/inference/infer_gigaworld.py \
            --config "${CONFIG}" \
            --prompt-file "${SPLIT}" \
            --checkpoint "${CKPT}" \
            --output-dir "${RUN_ROOT}/agibot_eveworld/seed_${seed}" \
            --num-steps 30 --cfg-scale 7.0 --seed "${seed}" \
            --num-shards "${SHARDS}" --shard-index "${index}"
    done
done

# 4. MLR over the first repeat; the AgiBot protocol uses the WorldArena thresholds.
run "${PYTHON}" scripts/evaluate/eval_mlr.py \
    --config "${EVAL_CONFIG}" \
    --pred-dir "${RUN_ROOT}/agibot_eveworld/seed_${SEEDS[0]}/generated_only" \
    --metadata "${METADATA}" \
    --model eveworld \
    --output "${RESULTS_ROOT}/item_level/ewmbench/mlr_eveworld.json" \
    "${LIMIT_ARGS[@]}"

# 5. EWMBench columns, read back from the official CSV.
soft "${PYTHON}" scripts/evaluate/eval_ewmbench.py \
    --config "${EVAL_CONFIG}" \
    --pred-dir "${EVAL_LAYOUT:-eval_layout}/eveworld_dataset" \
    --model eveworld \
    --output "${RESULTS_ROOT}/item_level/ewmbench/ewmbench_eveworld.json" \
    "${SCORE_ARGS[@]}" "${LIMIT_ARGS[@]}"

if [ "${FAILED}" -gt 0 ]; then
    printf '%d stage(s) failed; rerun them for a complete table\n' "${FAILED}" >&2
    exit 1
fi
echo "done: Table 4 summaries under ${RESULTS_ROOT}/item_level/ewmbench"
