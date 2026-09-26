#!/usr/bin/env bash
#SBATCH --job-name=eve_eval_t4g_exam_v2
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

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
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
OUT_DIR="${OUT_DIR:-/data/datasets/gagi/eve_v2_outputs/track4gen_probe/exam175_v2}"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

echo "== T4G-EXAM v2 恒存性体检分发 (带夹爪剔除与物理去噪): out=${OUT_DIR} =="
nvidia-smi || true

export EXAM_ARMS="${EXAM_ARMS:-pretrain round0 gr1_2b t4g_wmapA_pre_seed42_s250 t4g_wmapA_pre_s250 t4g_wmapA_pre_cleanv4_u3_s300 t4g_wmapA_s100}"
"${TRAIN_PYTHON}" eveworld/pipeline/t4g_exam_dispatch_v2.py 8 "${OUT_DIR}" "${TRAIN_PYTHON}"
exit "$?"
