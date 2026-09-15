#!/usr/bin/env bash
#SBATCH --job-name=t4g_measure
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TP="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-/data/datasets/gagi/eve_v2_outputs/track4gen_probe/measure}"
VIDS="${VIDS:?}"
source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate giga_models
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
python t4g_measure_dispatch.py 8 "$OUT_DIR" "$TP" "$VIDS"
echo MEASURE_DONE
