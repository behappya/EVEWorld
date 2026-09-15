#!/usr/bin/env bash
#SBATCH --job-name=eve_cfg_train_sweep
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -euo pipefail

for arg in "$@"; do [[ "$arg" == *=* ]] || { echo "bad arg: $arg" >&2; exit 2; }; export "$arg"; done
[[ "$(hostname)" == coder-workspace-* && "${ALLOW_LOCAL_RUN:-0}" != 1 ]] && exit 2

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GAGI="${GAGI:-/data/datasets/gagi}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI}/envs/giga_world_train_venv}"
PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
PRETRAIN="${PRETRAIN:-${GAGI}/giga_world_0_video_pretrain}"
PACKED="${PACKED_DATA_DIR:-${GAGI}/gr1_finetune_data/packed_data}"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_v2_outputs/t4g_cfg_repro_seed42_s300}"
PROJECT="${TRAIN_PROJECT_DIR:-${OUT_ROOT}/experiments}"
ANCHOR_ROOT="${MODEL_ROOT:-${GAGI}/eve_v2_outputs/anchor_models/cfg_repro_seed42}"
CFG_OUT="${CFG_OUT:-${GAGI}/eve_v2_outputs/cfg_grid_seed004}"
BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.pipeline.t4g_cfg_repro_seed42_config}"

source /home/jovyan/miniconda/etc/profile.d/conda.sh
conda activate giga_models
cd "$REPO_DIR"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"

for p in "$PACKED/config.json" "$PRETRAIN/transformer/config.json" "$PRETRAIN/vae/config.json"; do
  [[ -f "$p" ]] || { echo "missing required path: $p" >&2; exit 1; }
done

echo "== CFG sweep training preflight =="
echo "base_config=$BASE_CONFIG_MODULE output=$OUT_ROOT"
echo "Training from scratch/restart to step 300; checkpoints retained in a new root."

# Reuse the repository's audited training payload.  It writes runtime config,
# records GPU memory, and verifies the final step/checkpoint before returning.
bash "$REPO_DIR/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh" \
  "BASE_CONFIG_MODULE=$BASE_CONFIG_MODULE" \
  "TRAIN_VENV=$TRAIN_VENV" \
  "TRAIN_PROJECT_DIR=$PROJECT" \
  "OUTPUT_ROOT=$OUT_ROOT" \
  "RUN_NAME=cfg_repro_seed42_s300" \
  "PACKED_DATA_DIR=$PACKED" \
  "MODEL_DIR=$PRETRAIN" \
  "TRANSFORMER_MODEL_PATH=$PRETRAIN/transformer" \
  "VAE_MODEL_PATH=$PRETRAIN/vae" \
  "MAX_STEPS=300" "CHECKPOINT_INTERVAL=50" "CHECKPOINT_TOTAL_LIMIT=8" \
  "BATCH_SIZE_PER_GPU=1" "GRADIENT_ACCUMULATION_STEPS=8" \
  "SEED=42" "GPU_IDS=0 1 2 3 4 5 6 7" \
  "SKIP_TRAIN_ENV_SETUP=1"

echo "== Assemble immutable CFG probe roots =="
mkdir -p "$ANCHOR_ROOT"
for step in 50 100 150 200 250 300; do
  ckpt="$(find "$PROJECT/models" -maxdepth 1 -type d -name "*_step_${step}" -print -quit)"
  [[ -n "$ckpt" && -f "$ckpt/transformer/config.json" ]] || {
    echo "missing checkpoint for step $step under $PROJECT/models" >&2; exit 1;
  }
  d="$ANCHOR_ROOT/step_$(printf '%03d' "$step")"
  mkdir -p "$d"
  ln -sfn "$ckpt/transformer" "$d/transformer"
  ln -sfn "$PRETRAIN/text_encoder" "$d/text_encoder"
  ln -sfn "$PRETRAIN/vae" "$d/vae"
  for p in "$d/transformer/config.json" "$d/text_encoder/config.json" "$d/vae/config.json"; do
    [[ -f "$p" ]] || { echo "broken probe path: $p" >&2; exit 1; }
  done
done

echo "== Start serial checkpoint CFG sweep =="
"$PYTHON" eveworld/evaluation/cfg_grid_serial_dispatch.py \
  --model-root "$ANCHOR_ROOT" \
  --data-root "$GAGI/gr1_dreamgen_eval/giga_input" \
  --out-root "$CFG_OUT" \
  --steps-list 50 100 150 200 250 300 \
  --cfg-values 1.0 2.5 5.0 7.5 \
  --gpu-count 8 --seed 4 --inference-steps 30 \
  --data-pythonpath "${EVEWORLD_ROOT}:$REPO_DIR"

echo "CFG_TRAIN_SWEEP_DONE output=$CFG_OUT"
