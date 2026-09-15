#!/usr/bin/env bash
# EVE · LAD-LoRA 训练级反偷懒微调(方案27 §二十 正路)。
# 冻结预训 LAD, 对 backbone 去噪 x0 算 soft-top-k transition_error(sigma门控)作正则, 蒸馏进 LoRA。
# 复用 GigaWorld kjob 训练 payload(同 physlatent 范式); backbone 冻结+LoRA, 8×H20 显存充裕。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = giga-world-0
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# 同时接受位置参数 KEY=VALUE(与环境变量等价), 避免 `bash launch W_LAD=0.2` 被忽略的坑。
for _arg in "$@"; do
  [[ "${_arg}" == *=* ]] && export "${_arg}"
done

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh}"
export TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
export DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_finetune_data}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
export MODEL_DIR="${MODEL_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/eve}"

# ★ W_LAD/SEED 提前定义: TRAIN_PROJECT_DIR 需用它们隔离消融目录(否则 w=0.1/0.0/0.2 共用一个
# project_dir -> resume 互相误 resume + ckpt 互相覆盖, 已两次踩坑)。runtime config 由
# write_train_runtime_config.py 用【环境变量位置参数】覆盖 config 字段, 所以隔离/存点必须
# 改这里的环境变量, 改 config 文件会被顶掉(max_grad_norm 例外, 无对应位置参数, 从 config 生效)。
W_LAD="${W_LAD:-0.1}"
SEED="${SEED:-6666}"
BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.method.configs.eve_lad_lora}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-gradbal_v2}"
LAD_LORA_LR="${LAD_LORA_LR:-4.315837287515549e-05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
RESUME="${RESUME:-0}"

# 每个消融独立 project_dir(w+seed), 避免 resume 旧坏 ckpt / 互相覆盖。
export TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR:-${OUTPUT_ROOT}/experiments_lad_lora_${EXPERIMENT_TAG}_w${W_LAD}_seed${SEED}}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
# max_steps=50 (~35 epoch): 只训到老 run 开始发散处(step35), 密集存点挑最优。
# (老 run 无 grad clip 在 step35 发散; 本次加 max_grad_norm=1.0 于 config, 先验证能否止住。)
export MAX_STEPS="${MAX_STEPS:-50}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"           # step 5/10/../50 存 10 个点
export CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:--1}"    # 全保留(默认5会删早期健康点)
export SKIP_TRAIN_ENV_SETUP="${SKIP_TRAIN_ENV_SETUP:-1}"   # venv 已就绪, 跳过重装
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_eve_lad_lora_w${W_LAD}_seed${SEED}}"
export RUN_NAME   # 这个 launch_gr1 有默认+白名单, export 生效

if [[ -d "${TRAIN_PROJECT_DIR}" ]] && find "${TRAIN_PROJECT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  if [[ "${ALLOW_EXISTING_PROJECT:-0}" != "1" ]]; then
    echo "Refusing to reuse non-empty TRAIN_PROJECT_DIR: ${TRAIN_PROJECT_DIR}" >&2
    echo "Use a new EXPERIMENT_TAG, or explicitly set ALLOW_EXISTING_PROJECT=1." >&2
    exit 2
  fi
fi

PASS_ARGS=(
  "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}"
  "RUN_NAME=${RUN_NAME}"
  "SEED=${SEED}"
  "WITH_EMA=${WITH_EMA:-0}"
  "NUM_FRAMES=${NUM_FRAMES:-93}"
  "HEIGHT=${HEIGHT:-480}"
  "WIDTH=${WIDTH:-768}"
  "FPS=${FPS:-16}"
  "W_LAD=${W_LAD}"
  "EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  "LAD_LORA_LR=${LAD_LORA_LR}"
  "MAX_GRAD_NORM=${MAX_GRAD_NORM}"
  "RESUME=${RESUME}"
  "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}"
  "LAM_CKPT=${LAM_CKPT:-/data/datasets/gagi/eve_outputs/lam/lam_gr1.pt}"
  "LAD_TOPK=${LAD_TOPK:-3}"
  "LAD_TAU=${LAD_TAU:-0.5}"
  "LAD_SIGMA_MAX=${LAD_SIGMA_MAX:-0.0}"
  "LAD_BALANCE_MAX_SCALE=${LAD_BALANCE_MAX_SCALE:-1.0}"
)

echo "[EVE] 提交 LAD-LoRA 训练 config=${BASE_CONFIG_MODULE} steps=${MAX_STEPS} seed=${SEED} W_LAD=${W_LAD}"
echo "[EVE] RUN_NAME=${RUN_NAME}"
echo "[EVE] TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}"
echo "[EVE] lr=${LAD_LORA_LR} DeepSpeed-gradient-clipping=${MAX_GRAD_NORM} resume=${RESUME}"
echo "[EVE] 前置: lam_gr1.pt 须是 go/no-go PASS 的 LAD; venv 已就绪(SKIP_TRAIN_ENV_SETUP=1)"
echo "[EVE] 诊断必须单节点串行: 先 W_LAD=0.0 control, 验收稳定后再跑 W_LAD=0.1。"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] pass-args: ${PASS_ARGS[*]}"
  exit 0
fi
# launch_gr1 以 "$@" 结尾 -> submit 以 "$@" 结尾 -> kjob for-arg export。尾随参数一路透传。
exec "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_train_kjob.sh" "${PASS_ARGS[@]}"
