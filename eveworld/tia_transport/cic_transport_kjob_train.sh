#!/usr/bin/env bash
#SBATCH --job-name=eve_cic_transport_pair
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

VARIANT="${CIC_TRANSPORT_VARIANT:?CIC_TRANSPORT_VARIANT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

case "${VARIANT}" in
  control|transport) ;;
  *) echo "Unknown CIC_TRANSPORT_VARIANT=${VARIANT}" >&2; exit 2 ;;
esac

export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
"${TRAIN_PYTHON}" -m eveworld.tia_transport.cic_transport_campaign \
  node-preflight --variant "${VARIANT}" --output "${OUTPUT_ROOT}"

exec "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh" "$@"
