#!/usr/bin/env bash
#SBATCH --job-name=eve_ref_s049_s062
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing EVAL-175 generation on the workspace host." >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] || { echo "Bad argument: ${arg}" >&2; exit 2; }
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
GAGI="${GAGI:-/data/datasets/gagi}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
PYTHON="${TRAIN_PYTHON:-${GAGI}/envs/giga_world_train_venv/bin/python}"
CAMPAIGN_ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CAMPAIGN_ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062}"
PROBE_ROOT="${PROBE_ROOT:-${CAMPAIGN_ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062_probes}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${CAMPAIGN_ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062_eval}"
PRETRAIN_WEIGHT_DIR="${GAGI}/giga_world_0_video_pretrain/transformer"
SFT_WEIGHT_DIR="${GAGI}/eve_v2_outputs/round0_fullft/experiments_round0/models/checkpoint_epoch_150_step_150/transformer"
WORKER="${EVEWORLD_ROOT}/eveworld/evaluation/eval175_multiseed_worker.py"
RESUME="${RESUME:-0}"
models=(pretrain_raw_s000 standard_sft_raw_s150)
seeds=(49 62)

[[ "$(realpath -m "${OUTPUT_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062" ]] || {
  echo "Refusing unexpected OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
}
[[ "$(realpath -m "${PROBE_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062_probes" ]] || {
  echo "Refusing unexpected PROBE_ROOT=${PROBE_ROOT}" >&2
  exit 2
}
[[ "$(realpath -m "${MANIFEST_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_references_pretrain_s000_sft_s150_seed049_seed062_eval" ]] || {
  echo "Refusing unexpected MANIFEST_ROOT=${MANIFEST_ROOT}" >&2
  exit 2
}
[[ -s "${OUTPUT_ROOT}/audit/PREPARED" ]] || { echo "Missing PREPARED marker" >&2; exit 2; }
for weight_dir in "${PRETRAIN_WEIGHT_DIR}" "${SFT_WEIGHT_DIR}"; do
  [[ -s "${weight_dir}/config.json" ]] || { echo "Missing config: ${weight_dir}" >&2; exit 1; }
  compgen -G "${weight_dir}/diffusion_pytorch_model.*" >/dev/null || {
    echo "Missing weights: ${weight_dir}" >&2
    exit 1
  }
done

if [[ "${RESUME}" != "1" ]]; then
  [[ ! -e "${PROBE_ROOT}" ]] || { echo "Refusing existing probe root" >&2; exit 2; }
  [[ ! -e "${MANIFEST_ROOT}" ]] || { echo "Refusing existing manifest root" >&2; exit 2; }
  for model in "${models[@]}"; do
    [[ ! -e "${OUTPUT_ROOT}/${model}" ]] || {
      echo "Refusing existing generation output: ${model}" >&2
      exit 2
    }
  done
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
cd "${REPO_DIR}"

mkdir -p "${PROBE_ROOT}" "${MANIFEST_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/audit/generation_kjob.log") 2>&1
echo "[campaign] host=$(hostname) models=${models[*]} seeds=${seeds[*]}"
nvidia-smi

for model in "${models[@]}"; do
  case "${model}" in
    pretrain_raw_s000) weight_dir="${PRETRAIN_WEIGHT_DIR}" ;;
    standard_sft_raw_s150) weight_dir="${SFT_WEIGHT_DIR}" ;;
    *) echo "Unknown model: ${model}" >&2; exit 2 ;;
  esac
  probe="${PROBE_ROOT}/${model}"
  mkdir -p "${probe}"
  if [[ ! -e "${probe}/transformer" ]]; then
    ln -s "${weight_dir}" "${probe}/transformer"
    ln -s "${GAGI}/giga_world_0_video_pretrain/text_encoder" "${probe}/text_encoder"
    ln -s "${GAGI}/giga_world_0_video_pretrain/vae" "${probe}/vae"
  fi

  "${PYTHON}" eveworld/evaluation/eval175_seed_sharded_dispatch.py \
    --model "${model}" \
    --model-dir "${probe}" \
    --data-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${OUTPUT_ROOT}" \
    --chain-name "reference_${model}_seed049_seed062" \
    --seeds "${seeds[@]}" \
    --gpu-count 8 \
    --python "${PYTHON}" \
    --worker-script "${WORKER}" \
    --num-inference-steps 30 \
    --num-frames 93 \
    --fps 16 \
    --height 480 \
    --width 768 \
    --skip-existing

  "${PYTHON}" eveworld/evaluation/eval175_multiseed_prepare.py \
    --generation-root "${OUTPUT_ROOT}/${model}" \
    --input-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${MANIFEST_ROOT}/${model}" \
    --model-name "${model}" \
    --seeds "${seeds[@]}"
  "${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready"] is True and d["manifest_count"] == 252 and not d["errors"]' \
    "${MANIFEST_ROOT}/${model}/prepare_report.json"
done

date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/GENERATION_COMPLETE"
echo "[campaign] Pretrain s000 and SFT s150 seed049/062 generated and audited"
