#!/usr/bin/env bash
#SBATCH --job-name=eve_cic_t6666_s400
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"

for arg in "$@"; do
  [[ "${arg}" == *=* ]] || { echo "Unknown argument: ${arg}" >&2; exit 2; }
  export "${arg}"
done

OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
ALLOW_RESUME="${CIC_TRANSPORT_ALLOW_RESUME:-0}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

preflight_args=(
  -m eveworld.tia_transport.cic_transport_seed6666_s400_campaign
  node-preflight --output "${OUTPUT_ROOT}"
)
if [[ "${ALLOW_RESUME}" == "1" ]]; then
  preflight_args+=(--allow-resume)
fi
"${TRAIN_PYTHON}" "${preflight_args[@]}"

exec "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh" "$@"
