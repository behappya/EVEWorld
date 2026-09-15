#!/usr/bin/env bash
#SBATCH --job-name=eve_ablation_train_chain
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
GAGI="${GAGI:-/data/datasets/gagi}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/eve_ablation_strict_v1}"
PACKED_DATA_DIR="${PACKED_DATA_DIR:-${GAGI}/gr1_finetune_data/packed_data}"
PRETRAIN_TRANSFORMER="${PRETRAIN_TRANSFORMER:-${GAGI}/giga_world_0_video_pretrain/transformer}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-${GAGI}/giga_world_0_video_pretrain/vae}"
VARIANTS="${VARIANTS:?VARIANTS is required}"
CHAIN_NAME="${CHAIN_NAME:-ablation_chain}"
CHAIN_LOG="${OUTPUT_ROOT}/chain_logs/${CHAIN_NAME}.log"

mkdir -p "$(dirname "${CHAIN_LOG}")"
exec > >(tee -a "${CHAIN_LOG}") 2>&1

module_for() {
  case "$1" in
    matched_sft) echo "eveworld.pipeline.t4g_ablation_sft_config" ;;
    cp_only) echo "eveworld.pipeline.t4g_ablation_cp_only_config" ;;
    cwm_only) echo "eveworld.pipeline.t4g_ablation_cwm_only_config" ;;
    cic_only) echo "eveworld.pipeline.t4g_ablation_cic_only_config" ;;
    igt) echo "eveworld.pipeline.t4g_ablation_igt_config" ;;
    *) echo "Unknown variant: $1" >&2; return 2 ;;
  esac
}

verify_checkpoints() {
  local project_dir="$1"
  local step
  for step in 50 100 150 200 250; do
    find "${project_dir}/models" -maxdepth 1 -type d -name "checkpoint*_step_${step}" | grep -q . || {
      echo "Missing checkpoint step ${step}: ${project_dir}" >&2
      return 1
    }
  done
}

echo "== EVEWorld strict ablation training chain =="
echo "host=$(hostname) chain=${CHAIN_NAME} variants=${VARIANTS}"
echo "output=${OUTPUT_ROOT} seed=42 steps=250 checkpoint_interval=50 retain=5"
nvidia-smi

read -r -a VARIANT_ARGS <<< "${VARIANTS}"
for variant in "${VARIANT_ARGS[@]}"; do
  module="$(module_for "${variant}")"
  variant_root="${OUTPUT_ROOT}/${variant}_seed42_s250"
  project_dir="${variant_root}/experiments"
  run_name="${variant}_seed42_s250"

  if verify_checkpoints "${project_dir}" 2>/dev/null; then
    echo "[chain] ${variant}: all checkpoints already exist; skipping"
    continue
  fi

  echo "[chain] ${variant}: START module=${module} at $(date -u +%FT%TZ)"
  bash "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh" \
    "REPO_DIR=${REPO_DIR}" \
    "BASE_CONFIG_MODULE=${module}" \
    "TRANSFORMER_MODEL_PATH=${PRETRAIN_TRANSFORMER}" \
    "VAE_MODEL_PATH=${VAE_MODEL_PATH}" \
    "PACKED_DATA_DIR=${PACKED_DATA_DIR}" \
    "OUTPUT_ROOT=${variant_root}" \
    "TRAIN_PROJECT_DIR=${project_dir}" \
    "RUN_NAME=${run_name}" \
    "MAX_STEPS=250" \
    "SEED=42" \
    "CHECKPOINT_INTERVAL=50" \
    "CHECKPOINT_TOTAL_LIMIT=5" \
    "BATCH_SIZE_PER_GPU=1" \
    "GRADIENT_ACCUMULATION_STEPS=8" \
    "NUM_WORKERS=6" \
    "WITH_EMA=1" \
    "MIXED_PRECISION=bf16" \
    "ACTIVATION_CHECKPOINTING=1"

  verify_checkpoints "${project_dir}"
  final_transformer="$(find "${project_dir}/models" -maxdepth 3 -path '*step_250/transformer/diffusion_pytorch_model.bin' -print -quit)"
  [[ -n "${final_transformer}" ]] || { echo "Missing final raw transformer" >&2; exit 1; }
  sha256sum "${final_transformer}" | tee "${variant_root}/step250_transformer.sha256"
  echo "[chain] ${variant}: COMPLETE at $(date -u +%FT%TZ)"
done

echo "[chain] all variants complete at $(date -u +%FT%TZ)"
