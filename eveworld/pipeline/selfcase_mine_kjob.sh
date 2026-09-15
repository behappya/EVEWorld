#!/usr/bin/env bash
#SBATCH --job-name=selfcase_mine
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-giga-world-0}"
TP="${TRAIN_PYTHON:-/data/datasets/wkq_vlm/gagi/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-/data/datasets/wkq_vlm/gagi/eve_v2_outputs/selfcase/mine_round0}"
source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate giga_models
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${REPO_DIR}/eveworld/pipeline"
python selfcase_mine_dispatch.py 8 "$OUT_DIR" "$TP"
echo KJOB_DONE
