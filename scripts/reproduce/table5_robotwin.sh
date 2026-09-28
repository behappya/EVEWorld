#!/usr/bin/env bash
#
# Table 5 (tab:flowwam_transfer): the RoboTwin transfer experiment on FlowWAM.
#
# Stages, in order:
#   1. build the RoboTwin splits and the per-clip metadata under data/
#   2. fine-tune the four FlowWAM arms, 1,128 steps each at the Table 5 recipe
#   3. generate the 250 held-out episodes once per arm, plus once from the released
#      FlowWAM checkpoint, 40 denoising steps at CFG 5.0 with the robot-only flow
#      condition, inference-time TIA off and the direct full-trajectory setting
#   4. score every arm with the RoboTwin MLR protocol
#   5. pool the five variants into the Table 5 rows with the reconstruction metrics
#
# The reference frames of step 5 are the recorded episodes of the held-out range,
# named after the request id of the clip; the wrapper reads them from
# ${ROBOTWIN_HELDOUT_ROOT:-data/robotwin/heldout} and stops when that directory is
# missing, because neither the release nor the preparation stage writes it.
#
# Usage:
#   bash scripts/reproduce/table5_robotwin.sh
#
# Environment:
#   PYTHON                 interpreter of the stages (default: python)
#   ROBOTWIN_DATA_ROOT     RoboTwin source tree, read by the preparation stage
#   EVEWORLD_RUN_ROOT      training runs and generated clips (default: outputs)
#   EVEWORLD_RESULTS_ROOT  item-level results (default: outputs/evaluation)
#   SHARDS                 generation shards, run one after another (default: 1)
#   LIMIT                  cap the rows of every stage, for a smoke run
#   DRY_RUN                1 prints the commands of every stage instead of running them
#   EVEWORLD_CHECKPOINT_ROOT  root of the released checkpoints (default: checkpoints)
#   CKPT_FLOWWAM_STAGE1    released FlowWAM weights file of the FlowWAM row
#   CKPT_SFT, CKPT_IGR, CKPT_TIA, CKPT_EVEWORLD  checkpoint of one arm
#   ROBOTWIN_HELDOUT_ROOT  directory of the reference recordings of the held-out rows
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-python}"
RUN_ROOT="${EVEWORLD_RUN_ROOT:-outputs}"
RESULTS_ROOT="${EVEWORLD_RESULTS_ROOT:-outputs/evaluation}"
CHECKPOINT_ROOT="${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}"
TARGET_ROOT="${ROBOTWIN_HELDOUT_ROOT:-data/robotwin/heldout}"
SHARDS="${SHARDS:-1}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
FAILED=0

SPLIT="data/splits/robotwin/heldout.txt"
METADATA="data/metadata/robotwin"
EVAL_CONFIG="configs/eval/mlr/robotwin.yaml"
TRAIN_ARMS=(sft igr tia eveworld)
VARIANTS=(flowwam sft igr tia eveworld)
declare -A CONFIG=(
    [flowwam]="configs/paper/flowwam/robotwin/sft.yaml"
    [sft]="configs/paper/flowwam/robotwin/sft.yaml"
    [igr]="configs/paper/flowwam/robotwin/igr.yaml"
    [tia]="configs/paper/flowwam/robotwin/tia.yaml"
    [eveworld]="configs/paper/flowwam/robotwin/eveworld.yaml"
)
declare -A RUN_NAME=(
    [flowwam]="robotwin_flowwam_stage1"
    [sft]="robotwin_flowwam_sft"
    [igr]="robotwin_flowwam_igr"
    [tia]="robotwin_flowwam_tia"
    [eveworld]="robotwin_flowwam_eveworld"
)
declare -A CKPT=(
    [flowwam]="${CKPT_FLOWWAM_STAGE1:-${CHECKPOINT_ROOT}/flowwam/flowwam_robotwin.safetensors}"
    [sft]="${CKPT_SFT:-${RUN_ROOT}/robotwin_flowwam_sft}"
    [igr]="${CKPT_IGR:-${RUN_ROOT}/robotwin_flowwam_igr}"
    [tia]="${CKPT_TIA:-${RUN_ROOT}/robotwin_flowwam_tia}"
    [eveworld]="${CKPT_EVEWORLD:-${RUN_ROOT}/robotwin_flowwam_eveworld}"
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
    run "${PYTHON}" scripts/prepare/prepare_robotwin.py --output-dir data "${LIMIT_ARGS[@]}"
fi

# 2. Post-training.
for arm in "${TRAIN_ARMS[@]}"; do
    run "${PYTHON}" scripts/train/train_flowwam.py \
        --config "${CONFIG[${arm}]}" \
        --steps 1128 \
        --output-dir "${RUN_ROOT}/${RUN_NAME[${arm}]}"
done

# 3. Generation at the Table 5 protocol.
for variant in "${VARIANTS[@]}"; do
    for index in $(seq 0 $((SHARDS - 1))); do
        run "${PYTHON}" scripts/inference/infer_flowwam.py \
            --config "${CONFIG[${variant}]}" \
            --prompt-file "${SPLIT}" \
            --checkpoint "${CKPT[${variant}]}" \
            --output-dir "${RUN_ROOT}/${RUN_NAME[${variant}]}" \
            --num-steps 40 --cfg-scale 5.0 --seed 42 \
            --flow-cond robot_only --tia-inject off --full-traj direct \
            --num-shards "${SHARDS}" --shard-index "${index}"
    done
done

# 4. MLR per variant.
for variant in "${VARIANTS[@]}"; do
    run "${PYTHON}" scripts/evaluate/eval_mlr.py \
        --config "${EVAL_CONFIG}" \
        --pred-dir "${RUN_ROOT}/${RUN_NAME[${variant}]}/generated_only" \
        --metadata "${METADATA}" \
        --model "${variant}" \
        --output "${RESULTS_ROOT}/item_level/robotwin/mlr_${variant}.json" \
        "${LIMIT_ARGS[@]}"
done

# 5. Reconstruction metrics of the five variants against the recorded episodes.
if [ "${DRY_RUN}" != "1" ] && [ ! -d "${TARGET_ROOT}" ]; then
    printf 'error: %s is missing; it holds the reference recordings of the held-out rows\n' \
        "${TARGET_ROOT}" >&2
    exit 1
fi

PAIRS=()
for variant in "${VARIANTS[@]}"; do
    PAIRS+=(--pair "${variant}=${RUN_ROOT}/${RUN_NAME[${variant}]}/generated_only:${TARGET_ROOT}")
done

run "${PYTHON}" scripts/evaluate/eval_robotwin.py \
    --config "${EVAL_CONFIG}" \
    --output "${RESULTS_ROOT}/item_level/robotwin/table5.json" \
    "${PAIRS[@]}" "${LIMIT_ARGS[@]}"

if [ "${FAILED}" -gt 0 ]; then
    printf '%d stage(s) failed; rerun them for a complete table\n' "${FAILED}" >&2
    exit 1
fi
echo "done: Table 5 summary under ${RESULTS_ROOT}/item_level/robotwin/table5.json"
