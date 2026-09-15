#!/usr/bin/env bash
#SBATCH --job-name=selfcase_g1p
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-giga-world-0}"
TP="${TRAIN_PYTHON:-/data/datasets/wkq_vlm/gagi/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-/data/datasets/wkq_vlm/gagi/eve_v2_outputs/selfcase/g1p}"
source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate giga_models
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${REPO_DIR}/eveworld/pipeline"
# 34 例 x 2 sigma x 10 层: 单卡单进程串行即可 (整节点占位, 与 selfcase_g1_kjob.sh 同款)
CUDA_VISIBLE_DEVICES=0 "$TP" selfcase_g1p.py --out-dir "$OUT_DIR"
echo KJOB_DONE
