#!/usr/bin/env bash
#SBATCH --job-name=selfcase_g1
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TP="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-/data/datasets/gagi/eve_v2_outputs/selfcase/g1}"
source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate giga_models
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
# 12 条病例量小: 单进程单卡即可, 不分片
CUDA_VISIBLE_DEVICES=0 "$TP" selfcase_g1.py --out-dir "$OUT_DIR"
echo KJOB_DONE
