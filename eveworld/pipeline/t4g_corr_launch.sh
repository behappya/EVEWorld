#!/usr/bin/env bash
# Track4Gen 式对应监督训练 —— 50 步探针 kjob 提交入口 (42 号 §3/§4)。
#
# 铁律 (照 isoa/dpo 已核对的三层透传事实):
#   1) kjobctl 不把工作机环境变量带进 pod; 唯一端到端可靠通道是尾随位置参数 KEY=VALUE,
#      经 launch "$@" -> submit "$@" -> kjob_train_gr1_finetune.sh argv 逐个 export
#      (晚出现者覆盖先出现者)。
#   2) BASE_CONFIG_MODULE / TRANSFORMER_MODEL_PATH / CHECKPOINT_TOTAL_LIMIT 三者不在
#      launch 的显式转发列表, 也不在 submit 白名单 -> 必须走尾随 KEY=VALUE (下面已放尾随)。
#   3) PACKED_DATA_DIR / OUTPUT_ROOT / TRAIN_PROJECT_DIR / RUN_NAME / MAX_STEPS /
#      CHECKPOINT_INTERVAL / BATCH / GA 在 launch 列表或 submit 白名单内, 用 env 喂给
#      launch 让其本地检查/banner 正确; 同名尾随参数在 pod 侧最终覆盖 (双保险)。
#
# 单节点 8 卡串行, 禁碎片化 (kjob 铁律)。本脚本默认只打印计划, 不提交;
#   需真正提交时显式加子命令: `bash t4g_corr_launch.sh submit`
#
# ---- 四哨兵 (训练启动后监控; 第 4 哨兵 = 本探针纪律本身) ----
#   grep 't4g-sentinel' <train_log>   每步一行:
#     [S1] 四道 loss 裸值 L_id/L_cnt/L_neg/L_stat + 各自生效帧数 nf
#     [S2] param_disp_l2   (固定采样 1000 参数相对 init 的 L2 位移)
#     [S3] sim_tgt/sim_off/margin   (sim 分布, margin=目标格 sim − explained 外最大 sim)
#     [S4] 50 步探针先行 (流程纪律): 四道 loss 裸值下降 + 生成不崩 -> 才放全量 (42 号 §4 gate)
#   起步一行:  grep '\[t4g\] block=' —— 确认 anno/idx2vid 数量、λ、时间档、探针标量数
#   hook 一行: grep 'forward hook -> transformer.blocks\[block17\]'

set -euo pipefail

REPO_DIR="giga-world-0"
DATA_ROOT="/data/datasets/gagi/gr1_finetune_data"
PACKED="${DATA_ROOT}/packed_data"                                   # 92 条 GT
OUT_ROOT="/data/datasets/gagi/eve_v2_outputs/t4g_corr"
ROUND0_EMA_TRANSFORMER="/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno"

BASE_CONFIG_MODULE="eveworld.pipeline.t4g_corr_config"
RUN_NAME="${RUN_NAME:-t4g_corr_probe50}"
PROBE_STEPS="${PROBE_STEPS:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
CHECKPOINT_START_STEP="${CHECKPOINT_START_STEP:-0}"

# ---- 预检 (缺件早失败, 不进 kjob) ----
[[ -f "${PACKED}/config.json" ]]                                   || { echo "缺 ${PACKED}/config.json"; exit 1; }
[[ -f "${ROUND0_EMA_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || { echo "缺底座 ${ROUND0_EMA_TRANSFORMER}"; exit 1; }
[[ -f "${EVEWORLD_ROOT}/eveworld/pipeline/t4g_corr_config.py" ]]      || { echo "缺 t4g_corr_config.py"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                          || { echo "缺 data_index->vid 映射 ${ANNO_DIR}/_packidx2vid.json"; exit 1; }
[[ -d "${ANNO_DIR}" ]]                                            || { echo "缺 anno 目录 ${ANNO_DIR}"; exit 1; }

TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments"

# submit_cmd: 打印即事实; 用户执行 `submit` 才真正提交。
run_submit() {
  cd "${REPO_DIR}"
  mkdir -p "${OUT_ROOT}"
  # 前段走 env 通道 (launch 本地检查/banner); 尾随 KEY=VALUE 走 argv 通道 (pod 内最终生效)。
  PACKED_DATA_DIR="${PACKED}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  MAX_STEPS="${PROBE_STEPS}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  BATCH_SIZE_PER_GPU=1 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
    ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
      "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}" \
      "TRANSFORMER_MODEL_PATH=${ROUND0_EMA_TRANSFORMER}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${PROBE_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "CHECKPOINT_START_STEP=${CHECKPOINT_START_STEP}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8"
}

print_plan() {
  cat <<EOF
================ Track4Gen 对应监督 · 50 步探针提交计划 (未提交) ================
  BASE_CONFIG_MODULE   = ${BASE_CONFIG_MODULE}
  TRANSFORMER (底座)   = ${ROUND0_EMA_TRANSFORMER}   (round0_ema, safetensors)
  PACKED_DATA (92 GT)  = ${PACKED}
  ANNO_DIR             = ${ANNO_DIR}
  OUTPUT_ROOT          = ${OUT_ROOT}
  RUN_NAME             = ${RUN_NAME}
  MAX_STEPS            = ${PROBE_STEPS}   (探针; gate 通过后再放全量)
  batch                = 8 卡 × 1/卡 × GA8 = 64
  CHECKPOINT           = interval ${CHECKPOINT_INTERVAL} / total_limit ${CHECKPOINT_TOTAL_LIMIT} / start_step ${CHECKPOINT_START_STEP}
--------------------------------------------------------------------------------
  真正提交:  bash $(basename "$0") submit
  监控:      tail -f ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
  哨兵:      grep 't4g-sentinel' ${TRAIN_PROJECT_DIR}/logs/train_*.log
================================================================================
EOF
}

case "${1:-}" in
  submit) run_submit ;;
  *)      print_plan ;;
esac
