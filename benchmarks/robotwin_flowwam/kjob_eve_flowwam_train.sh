#!/usr/bin/env bash
#SBATCH --job-name=eve_flowwam_train
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-flowwam}"
FLOWWAM_ROOT="${FLOWWAM_ROOT:-/home/jovyan/FlowWAM}"
ARMS="${ARMS:-eve}"                       # 空格分隔, 串行
OUT_ROOT="${OUT_ROOT:-/data/datasets/gagi/flowwam/train_runs}"
RUN_TAG="${RUN_TAG:-run}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"
OVERFIT_N="${OVERFIT_N:-0}"
LR="${LR:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LR_MAX_STEPS="${LR_MAX_STEPS:-0}"
FLOW_MODE="${FLOW_MODE:-full_scene}"
FULL_OFFSET="${FULL_OFFSET:-off}"
INIT_ARM_CKPT="${INIT_ARM_CKPT:-}"
T_LAT_WIN="${T_LAT_WIN:-8}"
N_GPU="${N_GPU:-8}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
# tokenizer 相对路径依赖: cwd 内需有 models -> flowwam/models
cd "${FLOWWAM_ROOT}/training"
export PYTHONUNBUFFERED=1 FLOWWAM_ROOT
# 本集群分布式铁律: 不钉网卡 NCCL/gloo 集合通信必段错误(所有可用脚本均设)
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

overall_rc=0
for arm in ${ARMS}; do
  out="${OUT_ROOT}/${RUN_TAG}_${arm}"
  mkdir -p "${out}"
  echo "START arm=${arm} out=${out}"
  extra=()
  if [[ -n "${SAMPLES_PER_EPOCH}" ]]; then
    extra+=(--samples-per-epoch "${SAMPLES_PER_EPOCH}")
  fi
  if [[ -n "${INIT_ARM_CKPT}" ]]; then
    extra+=(--init-arm-ckpt "${INIT_ARM_CKPT}")
  fi
  extra+=(--full-offset "${FULL_OFFSET}" --t-lat-win "${T_LAT_WIN}")
  accelerate launch --num_processes "${N_GPU}" --mixed_precision bf16 \
    "${EVEWORLD_ROOT}/eveworld/flowwam_port/eve_flowwam_train.py" \
    --arm "${arm}" \
    --output-path "${out}" \
    --num-epochs "${NUM_EPOCHS}" \
    --overfit-n "${OVERFIT_N}" \
    --learning-rate "${LR}" \
    --save-steps "${SAVE_STEPS}" \
    --lr-max-steps "${LR_MAX_STEPS}" \
    --flow-mode "${FLOW_MODE}" \
    "${extra[@]}" \
    >"${out}/train.log" 2>&1
  rc=$?
  echo "DONE arm=${arm} rc=${rc}"
  [[ "${rc}" != "0" ]] && overall_rc=1
done
exit "${overall_rc}"
