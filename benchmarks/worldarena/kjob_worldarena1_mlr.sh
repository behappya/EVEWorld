#!/usr/bin/env bash
#SBATCH --job-name=wa1_mlr
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

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
TRAIN_PYTHON="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
WA1_ROOT="${WA1_ROOT:-/data/datasets/gagi/worldarena1}"
MANIFEST="${MANIFEST:-${WA1_ROOT}/manifests/track1_it2v.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/data/datasets/gagi/eve_v2_outputs/worldarena_eval_videos}"
OUT_DIR="${OUT_DIR:-${WA1_ROOT}/mlr_v2}"
MODELS="${MODELS:-pretrain round0 t4g_wmapA_pre_seed42_s250}"
LIMIT="${LIMIT:-}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

args=(
  --manifest "${MANIFEST}"
  --video-root "${VIDEO_ROOT}"
  --models ${MODELS}
  --output-dir "${OUT_DIR}"
  --python "${TRAIN_PYTHON}"
  --num-shards 8
)
if [[ -n "${LIMIT}" ]]; then
  args+=(--limit "${LIMIT}")
fi

"${TRAIN_PYTHON}" benchmarks/worldarena/mlr_dispatch.py "${args[@]}"
