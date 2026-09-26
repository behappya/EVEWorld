#!/usr/bin/env bash
#SBATCH --job-name=eve_cfg_grid_serial
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -euo pipefail

for arg in "$@"; do [[ "$arg" == *=* ]] || { echo "bad arg: $arg" >&2; exit 2; }; export "$arg"; done
[[ "$(hostname)" == coder-workspace-* && "${ALLOW_LOCAL_RUN:-0}" != 1 ]] && exit 2

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
PYTHON="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/data/datasets/gagi/eve_v2_outputs/anchor_models/cfg_repro_seed42}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/giga_input}"
OUT_ROOT="${OUT_ROOT:-/data/datasets/gagi/eve_v2_outputs/cfg_grid_seed004}"
CFG_VALUES="${CFG_VALUES:-1.0 2.5 5.0 7.0}"
source /home/jovyan/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-EVEWorld}"
cd "$REPO_DIR"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
exec "$PYTHON" eveworld/evaluation/cfg_grid_serial_dispatch.py \
  --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" --out-root "$OUT_ROOT" \
  --steps-list 50 100 150 200 250 300 --cfg-values ${CFG_VALUES} \
  --gpu-count 8 --seed 4 --inference-steps 30 \
  --data-pythonpath "${EVEWORLD_ROOT}:${REPO_DIR}"
