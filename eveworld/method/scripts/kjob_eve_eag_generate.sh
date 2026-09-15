#!/usr/bin/env bash
#SBATCH --job-name=eve_eag_gen
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# EVE · EAG 采样生成 kjob payload(单卡)。
# 对 DreamGen 输入生成视频: EAG_WEIGHT=0 -> baseline; >0 -> EAG 引导。同 seed 便于配对对比。
set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host (no GPU). Submit via launch_eve_eag_generate_kjob.sh" >&2
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

GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
MODEL_DIR="${MODEL_DIR:-${GAGI}/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
SAVE_DIR="${SAVE_DIR:-${GAGI}/eve_outputs/eag_eval/run}"

EAG_WEIGHT="${EAG_WEIGHT:-0.03}"
EAG_TOPK="${EAG_TOPK:-3}"
EAG_TAU="${EAG_TAU:-0.5}"
NUM_FRAMES="${NUM_FRAMES:-93}"
SEED="${SEED:-6666}"
STEPS="${STEPS:-30}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"; FPS="${FPS:-16}"
LIMIT="${LIMIT:-0}"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
cd "${REPO_DIR}"

echo "=========================================="
echo " EVE EAG generate  (eag_weight=${EAG_WEIGHT})"
echo "  data     = ${DATA_PATH}"
echo "  save     = ${SAVE_DIR}"
echo "  lam      = ${LAM}"
echo "  frames=${NUM_FRAMES} seed=${SEED} steps=${STEPS} limit=${LIMIT}"
echo "=========================================="

"${TRAIN_PYTHON}" eveworld/method/scripts/generate_eag.py \
  --data-path "${DATA_PATH}" --save-dir "${SAVE_DIR}" \
  --transformer "${MODEL_DIR}/transformer" \
  --text-encoder "${MODEL_DIR}/text_encoder" \
  --vae "${MODEL_DIR}/vae" \
  --lam "${LAM}" \
  --eag-weight "${EAG_WEIGHT}" --eag-topk "${EAG_TOPK}" --eag-tau "${EAG_TAU}" \
  --num-inference-steps "${STEPS}" --num-frames "${NUM_FRAMES}" \
  --height "${HEIGHT}" --width "${WIDTH}" --fps "${FPS}" \
  --seed "${SEED}" --limit "${LIMIT}"

echo "[eag-gen] DONE -> ${SAVE_DIR}"
