#!/usr/bin/env bash
#SBATCH --job-name=eve_eval175_serial
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host (no GPU)." >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
OUT_BASE="${OUT_BASE:-/data/datasets/gagi/eve_v2_outputs/eval175_gen}"
SERIAL_BATCH_SIZE="${SERIAL_BATCH_SIZE:-2}"

if [[ -z "${EVAL175_MODELS:-}" ]]; then
  echo "EVAL175_MODELS is required." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

MODEL_ARGS=(${EVAL175_MODELS})
SPLIT_ARGS=()
if [[ -n "${EVAL175_SPLITS:-}" ]]; then
  SPLIT_ARGS=(--splits ${EVAL175_SPLITS})
fi

echo "== EVAL-175 serial generation =="
echo "models: ${MODEL_ARGS[*]}"
echo "batch size: ${SERIAL_BATCH_SIZE}"
nvidia-smi || true

"${TRAIN_PYTHON}" eveworld/evaluation/x13_eval175_serial_dispatch.py \
  --out-base "${OUT_BASE}" \
  --python "${TRAIN_PYTHON}" \
  --batch-size "${SERIAL_BATCH_SIZE}" \
  --models "${MODEL_ARGS[@]}" \
  "${SPLIT_ARGS[@]}"
