#!/usr/bin/env bash
#SBATCH --job-name=eve_cic_ema_ckpt_sweep
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

for arg in "$@"; do
  [[ "${arg}" == *=* ]] || { echo "Bad argument: ${arg}" >&2; exit 2; }
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
GAGI="${GAGI:-/data/datasets/gagi}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI}/envs/giga_world_train_venv}"
PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1/cic_only_seed42_s250}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175_ema_ckpt_sweep}"
PROBE_ROOT="${PROBE_ROOT:-${GAGI}/eve_v2_outputs/anchor_models/eve_ablation_strict_v1_ema_ckpt_sweep}"
STEPS="${STEPS:?STEPS is required, for example: STEPS=50 100}"
CHAIN_NAME="${CHAIN_NAME:-cic_ema_ckpt_sweep}"
SEED="${SEED:-4}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
cd "${REPO_DIR}"

read -r -a step_args <<< "${STEPS}"
[[ "${#step_args[@]}" -gt 0 ]] || { echo "STEPS is empty" >&2; exit 2; }

for step in "${step_args[@]}"; do
  [[ "${step}" =~ ^(50|100|150|200)$ ]] || {
    echo "Unsupported CIC checkpoint step: ${step}" >&2
    exit 2
  }
  printf -v step_tag '%03d' "${step}"
  model="cic_only_seed42_s${step_tag}_ema"

  mapfile -t checkpoints < <(
    find "${TRAIN_MODEL_ROOT}/experiments/models" -maxdepth 1 -type d \
      -name "checkpoint*_step_${step}" -print | sort
  )
  [[ "${#checkpoints[@]}" -eq 1 ]] || {
    echo "Expected one step${step} checkpoint, found ${#checkpoints[@]}" >&2
    printf '%s\n' "${checkpoints[@]:-}" >&2
    exit 1
  }
  checkpoint="${checkpoints[0]}"
  weight_dir="${checkpoint}/transformer_ema"
  [[ -s "${weight_dir}/config.json" ]] || {
    echo "Missing EMA config: ${weight_dir}/config.json" >&2
    exit 1
  }
  compgen -G "${weight_dir}/diffusion_pytorch_model.*" >/dev/null || {
    echo "Missing EMA weights: ${weight_dir}" >&2
    exit 1
  }

  probe="${PROBE_ROOT}/${model}"
  mkdir -p "${probe}"
  ln -sfn "${weight_dir}" "${probe}/transformer"
  ln -sfn "${GAGI}/giga_world_0_video_pretrain/text_encoder" "${probe}/text_encoder"
  ln -sfn "${GAGI}/giga_world_0_video_pretrain/vae" "${probe}/vae"

  echo "[generate] model=${model} seed=${SEED} checkpoint=${checkpoint} weight_kind=transformer_ema"
  "${PYTHON}" eveworld/evaluation/eval175_seed_sharded_dispatch.py \
    --model "${model}" \
    --model-dir "${probe}" \
    --data-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${OUTPUT_ROOT}" \
    --chain-name "${CHAIN_NAME}_${model}" \
    --seeds "${SEED}" \
    --gpu-count 8 \
    --python "${PYTHON}" \
    --num-inference-steps 30 \
    --num-frames 93 \
    --fps 16 \
    --height 480 \
    --width 768 \
    --skip-existing
done
