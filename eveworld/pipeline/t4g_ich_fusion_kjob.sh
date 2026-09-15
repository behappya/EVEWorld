#!/usr/bin/env bash
#SBATCH --job-name=t4g_ich_fusion
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-giga-world-0}"
TP="${TRAIN_PYTHON:-/data/datasets/wkq_vlm/gagi/envs/giga_world_train_venv/bin/python}"
source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate giga_models
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${REPO_DIR}/eveworld/pipeline"
# 34 例 x 2 sigma x 10 层重抽 + 逐格特征落盘 + CPU 融合 LOO (整节点占位, 与 g1p_kjob 同款)
CUDA_VISIBLE_DEVICES=0 "$TP" t4g_ich_fusion_ablation.py
echo KJOB_DONE
