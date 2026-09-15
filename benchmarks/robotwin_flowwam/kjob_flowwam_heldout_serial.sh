#!/usr/bin/env bash
#SBATCH --job-name=flowwam_heldout_r250
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -euo pipefail

for arg in "$@"; do [[ "$arg" == *=* ]] || { echo "bad arg: $arg" >&2; exit 2; }; export "$arg"; done
[[ "$(hostname)" == coder-workspace-* && "${ALLOW_LOCAL_RUN:-0}" != 1 ]] && exit 2

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
FLOWWAM_ROOT="${FLOWWAM_ROOT:-/home/jovyan/FlowWAM}"
FLOWWAM_DATA="${FLOWWAM_DATA:-/data/datasets/gagi/flowwam/data_worldarena/640_extracted}"
OUT_ROOT="${OUT_ROOT:-/data/datasets/gagi/flowwam/heldout_r250_v1}"
MANIFEST="${MANIFEST:-${OUT_ROOT}/manifest.json}"
PYTHON="${PYTHON:-/home/jovyan/miniconda/envs/flowwam/bin/python}"

source /home/jovyan/miniconda/etc/profile.d/conda.sh
conda activate flowwam
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}:${FLOWWAM_ROOT}:${FLOWWAM_ROOT}/inference:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

for arm in control eve; do
  ckpt="/data/datasets/gagi/flowwam/train_runs/arm_${arm}/final.safetensors"
  [[ -s "$ckpt" ]] || { echo "missing FlowWAM checkpoint: $ckpt" >&2; exit 1; }
done

"$PYTHON" eveworld/flowwam_port/build_flowwam_heldout_manifest.py \
  --data-root "$FLOWWAM_DATA" --out "$MANIFEST"

N_GPU=8 ARMS="control eve" RUN_TAG=arm \
TRAIN_ROOT=/data/datasets/gagi/flowwam/train_runs \
CKPT_NAME=final.safetensors \
OUT_ROOT="$OUT_ROOT" MANIFEST="$MANIFEST" \
FLOW_COND=robot_only CFG_SCALE=5.0 GEN_STEPS=40 TIA_INJECT=off \
FULL_TRAJ=direct GEN_SEED=42 LIMIT=0 \
bash "${EVEWORLD_ROOT}/benchmarks/robotwin_flowwam/kjob_flowwam_arm_generate.sh"

echo "FLOWWAM_HELDOUT_DONE output=$OUT_ROOT"
