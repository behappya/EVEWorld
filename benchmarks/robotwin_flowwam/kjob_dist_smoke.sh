#!/usr/bin/env bash
#SBATCH --job-name=dist_smoke
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

set -uo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-flowwam}"
OUT="${OUT:-/data/datasets/gagi/flowwam/train_runs/dist_smoke_${CONDA_ENV}.log}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO

torchrun --standalone --nproc-per-node=8 \
  eveworld/flowwam_port/dist_smoke.py >"${OUT}" 2>&1
rc=$?
echo "dist smoke rc=${rc} log=${OUT}"
exit "${rc}"
