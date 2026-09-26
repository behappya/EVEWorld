#!/usr/bin/env bash
# IGR 权重图变体 interaction_2x —— 提交入口 (六变体并列, 无推荐; 见 t4g_weightmap_variants.md)。
# 除"权重图"这一个变量外, 其余超参与六变体完全平行 (cleanv2 配方, 300 步, seed 6666)。
# 默认打印计划, `submit` 才提交; `check` 只做静态检查 + 自检 + 缓存体检 (不训练)。
set -euo pipefail

REPO_DIR="${REPO_DIR:-giga-world-0}"
GAGI="/data/datasets/gagi"
PACKED="${GAGI}/gr1_finetune_data/packed_data"
PRETRAIN="${GAGI}/giga_world_0_video_pretrain/transformer"
WMAP_ROOT="${T4G_WMAP_ROOT:-${GAGI}/eve_v2_outputs/track4gen_probe}"
WMAP_DIR="${WMAP_ROOT}/weightmap_cache_interaction_2x"
OUT_ROOT="${GAGI}/eve_v2_outputs/t4g_weightmap_variants/interaction_2x"
RUN_NAME="${RUN_NAME:-t4g_wmap_interaction_2x_s300}"
BASE_CONFIG_MODULE="eveworld.pipeline.t4g_weightmap_interaction_2x_config"
MAX_STEPS="${MAX_STEPS:-300}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
SEED="6666"
EXPECTED_SAMPLES=92
PYTHON="${PYTHON:-/home/jovyan/miniconda/envs/EVEWorld/bin/python}"

run_check() {
  [[ -f "${PACKED}/config.json" ]] || { echo "missing packed data: ${PACKED}"; exit 1; }
  [[ -f "${PRETRAIN}/diffusion_pytorch_model.safetensors" ]] || { echo "missing pretrain: ${PRETRAIN}"; exit 1; }
  cd "${REPO_DIR}"
  "${PYTHON}" -m py_compile \
    eveworld/pipeline/t4g_weightmap.py \
    eveworld/pipeline/t4g_weightmap_variants.py \
    eveworld/pipeline/t4g_weightmap_variants_precompute.py \
    eveworld/pipeline/t4g_weightmap_interaction_2x_config.py
  "${PYTHON}" eveworld/pipeline/t4g_weightmap_variants.py --self-test

  N_WMAP=0
  if [[ -d "${WMAP_DIR}" ]]; then
    N_WMAP=$(find "${WMAP_DIR}" -maxdepth 1 -name '*.npy' | wc -l)
  fi
  if [[ "${N_WMAP}" -lt "${EXPECTED_SAMPLES}" ]]; then
    echo "WARNING: 变体缓存 ${WMAP_DIR} 只有 ${N_WMAP}/${EXPECTED_SAMPLES} 个 .npy" >&2
    echo "  先生成: T4G_WMAP_ROOT=${WMAP_ROOT} ${PYTHON} eveworld/pipeline/t4g_weightmap_variants_precompute.py --variant interaction_2x" >&2
  else
    echo "权重图缓存 ${WMAP_DIR}: ${N_WMAP} 个 .npy"
  fi
}

run_submit() {
  run_check
  if find "${OUT_ROOT}/experiments/models" -maxdepth 1 -type d -name 'checkpoint*' \
      -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to resume existing checkpoints under ${OUT_ROOT}" >&2
    exit 1
  fi

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
      "SEED=${SEED}"
}

case "${1:-}" in
  check) run_check ;;
  submit) run_submit ;;
  *)
    cat <<EOF
IGR 权重图变体 interaction_2x (六变体并列, 无推荐)
  config: ${BASE_CONFIG_MODULE}
  级别:   1x(背景) / 2x 二值
  区域:   交互区 = Path/Loc.{obj,trans,b_dest,empty} + Arm/Grip.{grip} 共 2x
  data:   original packed_data, ${EXPECTED_SAMPLES} samples
  base:   ${PRETRAIN}
  wmap:   ${WMAP_DIR}
  output: ${OUT_ROOT}
  train:  ${MAX_STEPS} steps, checkpoints every ${CHECKPOINT_INTERVAL}, seed ${SEED}

Commands:
  bash $(basename "$0") check
  bash $(basename "$0") submit
EOF
    ;;
esac
