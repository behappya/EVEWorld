#!/usr/bin/env bash
#SBATCH --job-name=flowwam_gdino
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
CONDA_ENV="${CONDA_ENV:-giga_models}"
MANIFEST="${MANIFEST:-/data/datasets/gagi/flowwam/igr/manifest_640.json}"
OUT_DIR="${OUT_DIR:-/data/datasets/gagi/flowwam/igr/anno_640}"
N_GPU="${N_GPU:-8}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1
export N_GPU

python eveworld/flowwam_port/gdino_dispatch.py \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}"
