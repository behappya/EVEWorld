#!/usr/bin/env bash
#SBATCH --job-name=eve_lora_eval_4gpu
#SBATCH --gpus-per-task=nvidia.com/gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host." >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
MODEL_DIR="${MODEL_DIR:-${GAGI}/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_outputs/lad_lora_eval}"
TASK_SPECS="${TASK_SPECS:?TASK_SPECS is required}"
NUM_FRAMES="${NUM_FRAMES:-93}"
STEPS="${STEPS:-30}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
FPS="${FPS:-16}"
LIMIT="${LIMIT:-16}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

source "${TRAIN_VENV}/bin/activate"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

read -r -a TASK_ARR <<< "${TASK_SPECS}"
if (( ${#TASK_ARR[@]} > 4 )); then
  echo "4-GPU payload received ${#TASK_ARR[@]} tasks" >&2
  exit 2
fi
SKIP_ARG=()
[[ "${SKIP_EXISTING}" == "1" ]] && SKIP_ARG=(--skip-existing)

echo "[lora-eval] host=$(hostname) tasks=${#TASK_ARR[@]} out=${OUT_ROOT} limit=${LIMIT}"
nvidia-smi
"${TRAIN_PYTHON}" eveworld/method/scripts/lora_eval_dispatch.py \
  --tasks "${TASK_ARR[@]}" \
  --data-path "${DATA_PATH}" --out-root "${OUT_ROOT}" \
  --transformer "${MODEL_DIR}/transformer" \
  --text-encoder "${MODEL_DIR}/text_encoder" \
  --vae "${MODEL_DIR}/vae" --lam "${LAM}" \
  --python "${TRAIN_PYTHON}" \
  --gen-script "${EVEWORLD_ROOT}/eveworld/method/scripts/generate_eag.py" \
  --num-frames "${NUM_FRAMES}" --steps "${STEPS}" \
  --height "${HEIGHT}" --width "${WIDTH}" --fps "${FPS}" \
  --limit "${LIMIT}" "${SKIP_ARG[@]}"
