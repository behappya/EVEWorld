#!/usr/bin/env bash
#SBATCH --job-name=agibot_detect
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# AgiBot 777 条 GDINO 实体+夹爪检测, 单节点 8 卡, Python 分发器编排两轮串行。

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_world1}"
AGI_DIR="${AGI_DIR:-eveworld/agibot}"
LOG_DIR=/data/datasets/gagi/eve_v2_outputs/agibot_t4g_probe
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset CUDA_VISIBLE_DEVICES

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${AGI_DIR}"
nvidia-smi || true

python agi_detect_dispatch.py >"${LOG_DIR}/agi_detect.log" 2>&1
rc=$?
n1=$(ls "${LOG_DIR}/t4g_anno/"*.json 2>/dev/null | wc -l)
n2=$(ls "${LOG_DIR}/gripper_anno/"*.json 2>/dev/null | wc -l)
echo "agibot detect done rc=${rc} anno=${n1}/778 gripper=${n2}/777" | tee -a "${LOG_DIR}/agi_detect.log"
exit "${rc}"
