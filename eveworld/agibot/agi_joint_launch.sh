#!/usr/bin/env bash
set -euo pipefail

# AgiBot 双臂配方训练 launcher (方案 Phase 6.2)。
#   bash agi_joint_launch.sh check|submit [ARM=wmaponly|full] [T4G_ID_BLOCK=blockN] ...
# ARM=wmaponly (默认, 最小验证档 s150) / full (完整配方 s300)。
# 关键: T4G_W_LAT/T4G_WPIX 必须作尾随 KEY=VALUE 透传进 kjob (transform import 时读 env,
# 不传则建 (24,30,48) 网格触发 shape assert)。

for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}" || true; done

REPO_DIR="giga-world-0"
GAGI="/data/datasets/gagi"
PACKED="${GAGI}/agibot_ewm_packed"
PRETRAIN="${GAGI}/giga_world_0_video_pretrain/transformer"
OUT_ROOT="${GAGI}/eve_v2_outputs/agibot_ewm_apre"
ARM="${ARM:-wmaponly}"
PYTHON="/home/jovyan/miniconda/envs/EVEWorld/bin/python"

case "${ARM}" in
  wmaponly)
    BASE_CONFIG_MODULE="eveworld.agibot.agi_wmaponly_config"
    MAX_STEPS="${MAX_STEPS:-150}"
    CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-4}"
    ;;
  full)
    BASE_CONFIG_MODULE="eveworld.agibot.agi_joint_config"
    MAX_STEPS="${MAX_STEPS:-300}"
    CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
    ;;
  *) echo "unknown ARM=${ARM} (wmaponly|full)"; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-agi_apre_${ARM}_s${MAX_STEPS}}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
WMAP_VARIANT="${WMAP_VARIANT:-motionauto}"

run_check() {
  [[ -f "${PACKED}/config.json" ]] || { echo "missing packed data: ${PACKED}"; exit 1; }
  [[ -f "${PRETRAIN}/diffusion_pytorch_model.safetensors" ]] || { echo "missing pretrain"; exit 1; }
  cd "${REPO_DIR}"
  "${PYTHON}" eveworld/agibot/agi_preflight.py --wmap "${WMAP_VARIANT}"
  "${PYTHON}" -m py_compile \
    eveworld/agibot/agi_aug_trainer.py \
    eveworld/agibot/agi_joint_config.py \
    eveworld/agibot/agi_wmaponly_config.py
}

run_submit() {
  run_check
  cd "${REPO_DIR}"
  SKIP_TRAIN_ENV_SETUP=1 \
  PACKED_DATA_DIR="${PACKED}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments" \
  RUN_NAME="${RUN_NAME}" \
  MAX_STEPS="${MAX_STEPS}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT}" \
  BATCH_SIZE_PER_GPU=1 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
    ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
      "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}" \
      "TRANSFORMER_MODEL_PATH=${PRETRAIN}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${OUT_ROOT}/experiments" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8" \
      "WIDTH=640" \
      "HEIGHT=480" \
      "SEED=42" \
      "T4G_W_LAT=40" \
      "T4G_WPIX=640" \
      ${T4G_ID_BLOCK:+"T4G_ID_BLOCK=${T4G_ID_BLOCK}"} \
      ${T4G_LAMBDAS:+"T4G_LAMBDAS=${T4G_LAMBDAS}"} \
      ${T4G_WMAP_DIR:+"T4G_WMAP_DIR=${T4G_WMAP_DIR}"}
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  *)
    cat <<EOF
AgiBot 双臂配方 launcher
  packed: ${PACKED} (777)
  base:   ${PRETRAIN}
  output: ${OUT_ROOT}
  arm:    ${ARM} -> ${BASE_CONFIG_MODULE} (${MAX_STEPS} steps)

Commands:
  bash $(basename "$0") check  [ARM=wmaponly|full]
  bash $(basename "$0") submit [ARM=wmaponly|full] [T4G_ID_BLOCK=blockN] [T4G_WMAP_DIR=...]

Recipe length follows MAX_STEPS (defaults: wmaponly=150, full=300); both are
overridable, e.g. MAX_STEPS=50 for the AgiBot clip budget and MAX_STEPS=250
for the GR1/DreamGenBench budget. RUN_NAME/CHECKPOINT_INTERVAL track it too.
EOF
    ;;
esac
