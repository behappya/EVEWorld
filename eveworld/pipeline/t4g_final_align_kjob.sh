#!/usr/bin/env bash
#SBATCH --job-name=t4g_final_align
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TP="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate giga_models
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
# 40 例在线 latent 回填目标 vs 离线像素金标准对账 (只载 VAE) (单卡, 整节点占位同 g1p 模式)
CUDA_VISIBLE_DEVICES=0 "$TP" t4g_final_align.py
echo KJOB_DONE
