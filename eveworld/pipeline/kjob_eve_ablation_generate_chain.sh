#!/usr/bin/env bash
#SBATCH --job-name=eve_ablation_seed004
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
TRAIN_ROOT="${TRAIN_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1_eval175}"
PROBE_ROOT="${PROBE_ROOT:-${GAGI}/eve_v2_outputs/anchor_models/eve_ablation_strict_v1}"
EVEWORLD_PROJECT_DIR="${EVEWORLD_PROJECT_DIR:-${GAGI}/eve_v2_outputs/t4g_joint_wmapA_pre_seed42_s300/experiments}"
VARIANTS="${VARIANTS:?VARIANTS is required}"
CHAIN_NAME="${CHAIN_NAME:-strict_seed004}"
SEED="${SEED:-4}"
WEIGHT_KIND="${WEIGHT_KIND:-transformer}"
OUTPUT_MODEL_SUFFIX="${OUTPUT_MODEL_SUFFIX:-}"

case "${WEIGHT_KIND}" in
  transformer|transformer_ema) ;;
  *) echo "WEIGHT_KIND must be transformer or transformer_ema" >&2; exit 2 ;;
esac

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
cd "${REPO_DIR}"

read -r -a VARIANT_ARGS <<< "${VARIANTS}"
for variant in "${VARIANT_ARGS[@]}"; do
  if [[ "${variant}" == "eveworld" ]]; then
    base_model="eveworld_seed42_s250"
    project_dir="${EVEWORLD_PROJECT_DIR}"
  else
    base_model="${variant}_seed42_s250"
    project_dir="${TRAIN_ROOT}/${base_model}/experiments"
  fi
  model="${base_model}${OUTPUT_MODEL_SUFFIX}"
  checkpoint="$(find "${project_dir}/models" -maxdepth 1 -type d -name 'checkpoint*_step_250' -print -quit)"
  [[ -n "${checkpoint}" ]] || { echo "Missing step250 checkpoint for ${variant}" >&2; exit 1; }
  weight_dir="${checkpoint}/${WEIGHT_KIND}"
  [[ -s "${weight_dir}/config.json" ]] || {
    echo "Missing ${WEIGHT_KIND} config for ${variant}: ${weight_dir}" >&2
    exit 1
  }
  compgen -G "${weight_dir}/diffusion_pytorch_model.*" >/dev/null || {
    echo "Missing ${WEIGHT_KIND} weights for ${variant}: ${weight_dir}" >&2
    exit 1
  }

  probe="${PROBE_ROOT}/${model}"
  mkdir -p "${probe}"
  ln -sfn "${weight_dir}" "${probe}/transformer"
  ln -sfn "${GAGI}/giga_world_0_video_pretrain/text_encoder" "${probe}/text_encoder"
  ln -sfn "${GAGI}/giga_world_0_video_pretrain/vae" "${probe}/vae"

  echo "[generate] ${model} seed=${SEED} checkpoint=${checkpoint} weight_kind=${WEIGHT_KIND}"
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
