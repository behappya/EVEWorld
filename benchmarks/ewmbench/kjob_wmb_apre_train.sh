#!/usr/bin/env bash
#SBATCH --job-name=wmb_apre
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# WMB 适配 A-pre 训练(56号 T1): 直接用现成 runtime json(T4GJointTrainer),
# 不重新生成 config。网格 640 宽须 export T4G_W_LAT/T4G_WPIX(t4g_aug_paste 兼容补丁)。

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-/data/datasets/gagi/wmb_adapt/runtime_configs/t4g_wmb_apre_s600.json}"
LOG_DIR=/data/datasets/gagi/wmb_adapt/t4g_wmb_apre
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
touch "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export HF_HOME=/data/datasets/gagi/.hf_home
export T4G_W_LAT=40
export T4G_WPIX=640
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES

# shellcheck disable=SC1090
source "${TRAIN_VENV}/bin/activate"
TRAIN_PYTHON="${TRAIN_VENV}/bin/python"

# DeepSpeed: 节点无 nvcc, 关闭 op 编译与 CUDA 探测(与 kjob_train_gr1_finetune.sh 同)
export DS_BUILD_OPS=0
export DS_BUILD_FP_QUANTIZER=0
export DS_IGNORE_CUDA_DETECTION=1
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12 "${CONDA_PREFIX:-}/targets/x86_64-linux"; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi

cd "${REPO_DIR}"
echo "host=$(hostname) config=${RUNTIME_CONFIG} T4G_W_LAT=${T4G_W_LAT}"
nvidia-smi -L | head -2

# 先补齐 aug 资产(缺则跑; 断点靠 npz 覆盖写, 幂等), 用 giga_world1 的 python(transformers 支持 GDINO 链)
N_AUG=$(ls /data/datasets/gagi/wmb_adapt/aug_assets_v1/*.npz 2>/dev/null | wc -l)
echo "aug assets present: ${N_AUG}/500"
if [[ "${N_AUG}" -lt "500" ]]; then
  N_SHARDS=16 /home/jovyan/miniconda/envs/giga_world1/bin/python \
    eveworld/agibot/w6_dispatch.py
  echo "aug fill rc=$? now=$(ls /data/datasets/gagi/wmb_adapt/aug_assets_v1/*.npz 2>/dev/null | wc -l)/500"
fi

"${TRAIN_PYTHON}" scripts/train.py --config "${RUNTIME_CONFIG}"
rc=$?
echo "train exit rc=${rc}"
exit "${rc}"
