#!/usr/bin/env bash
#SBATCH --job-name=t4g_viz
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# Track4Gen 可视化诊断 payload: 导相似度热力图/argmax落点 PNG (人工看 NO-GO 真假)。
set -uo pipefail
if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host." >&2; exit 2
fi
for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
OUT_DIR="${OUT_DIR:-/data/datasets/gagi/eve_v2_outputs/track4gen_probe/viz}"
VIDEO_IDS="${VIDEO_IDS:-13,76}"
LAYERS="${LAYERS:-block13,block17,block21}"
SIGMAS="${SIGMAS:-0.2,0.4}"

source "${CONDA_SH}"; conda activate giga_models
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
nvidia-smi || true
IFS=',' read -r -a VIDS <<< "${VIDEO_IDS}"
IFS=',' read -r -a SIGS <<< "${SIGMAS}"
for vid in "${VIDS[@]}"; do
  for sig in "${SIGS[@]}"; do
    echo "== viz vid=${vid} sig=${sig} =="
    "${TRAIN_PYTHON}" t4g_viz.py --video-id "${vid}" --layers "${LAYERS}" \
      --sigma "${sig}" --out-dir "${OUT_DIR}" || echo "FAIL vid=${vid} sig=${sig}"
  done
done
echo "VIZ_DONE -> ${OUT_DIR}"
