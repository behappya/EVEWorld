#!/usr/bin/env bash
#SBATCH --job-name=t4g_detect
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# 92 条 GT 逐帧 GDINO 检测 -> 训练标注缓存 (单节点8卡分片, Python 编排)。
set -uo pipefail
if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing on workspace." >&2; exit 2
fi
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}"; exit 1; }; done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno}"
NGPU="${NGPU:-8}"
source "${CONDA_SH}"; conda activate "${CONDA_ENV:-EVEWorld}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
nvidia-smi || true
python t4g_detect_dispatch.py "$NGPU" "$OUT_DIR" "$TRAIN_PYTHON"
echo "DETECT_DONE -> ${OUT_DIR}"
