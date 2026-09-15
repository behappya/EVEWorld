#!/usr/bin/env bash
#SBATCH --job-name=eve_eval175_gen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# EVE · 官方 DreamGenBench(EVAL-175 gr1 三 split) 生成 —— round0 + anmix_s200, 单 seed。
# (模型×split) 6 单元各钉一张卡, Python 编排。
set -uo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host (no GPU)." >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
OUT_BASE="${OUT_BASE:-/data/datasets/gagi/eve_v2_outputs/eval175_gen}"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

echo "== EVAL-175 官方基准生成: out=${OUT_BASE} =="
nvidia-smi || true

EXTRA_ARGS=()
[[ -n "${EVAL175_MODELS:-}" ]] && EXTRA_ARGS+=(--models ${EVAL175_MODELS})
[[ -n "${EVAL175_SPLITS:-}" ]] && EXTRA_ARGS+=(--splits ${EVAL175_SPLITS})
"${TRAIN_PYTHON}" eveworld/evaluation/x9_eval175_dispatch.py --out-base "${OUT_BASE}" --python "${TRAIN_PYTHON}" "${EXTRA_ARGS[@]}"
exit "$?"
