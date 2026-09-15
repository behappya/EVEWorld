#!/usr/bin/env bash
#SBATCH --job-name=flowwam_tia_probe
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
PER_TASK="${PER_TASK:-3}"
T_FRACS="${T_FRACS:-0.1,0.3,0.5}"
N_GPU="${N_GPU:-8}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1
export N_GPU FLOWWAM_ROOT=/home/jovyan/FlowWAM

python eveworld/flowwam_port/tia_probe_dispatch.py \
  --per-task "${PER_TASK}" \
  --t-fracs "${T_FRACS}"
