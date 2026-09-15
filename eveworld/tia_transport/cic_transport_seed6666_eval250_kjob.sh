#!/usr/bin/env bash
#SBATCH --job-name=eve_cic_s6666_eval250
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
CONDA_ENV="${CONDA_ENV:-giga_models}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI}/envs/giga_world_train_venv}"
PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
CAMPAIGN_ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CAMPAIGN_ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250}"
PROBE_ROOT="${PROBE_ROOT:-${CAMPAIGN_ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250_probes}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${CAMPAIGN_ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250_eval}"
PROJECT="${CAMPAIGN_ROOT}/cic_transport_seed6666_s400/experiments"
WORKER="${EVEWORLD_ROOT}/eveworld/tia_transport/eval175_transport_worker.py"
SEED="${SEED:-4}"
RESUME="${RESUME:-0}"

[[ "$(realpath -m "${OUTPUT_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250" ]] || {
  echo "Refusing unexpected OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
}
[[ "$(realpath -m "${PROBE_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250_probes" ]] || {
  echo "Refusing unexpected PROBE_ROOT=${PROBE_ROOT}" >&2
  exit 2
}
[[ "$(realpath -m "${MANIFEST_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_seed004_transport_seed6666_raw_s150_s250_eval" ]] || {
  echo "Refusing unexpected MANIFEST_ROOT=${MANIFEST_ROOT}" >&2
  exit 2
}
[[ "${SEED}" == "4" ]] || { echo "This campaign is frozen to seed004" >&2; exit 2; }
[[ -s "${OUTPUT_ROOT}/audit/PREPARED" ]] || {
  echo "Missing launcher preparation marker" >&2
  exit 2
}

steps=(150 200 250)
models=(
  transport_seed6666_raw_s150
  transport_seed6666_raw_s200
  transport_seed6666_raw_s250
)

if [[ "${RESUME}" != "1" ]]; then
  for model in "${models[@]}"; do
    [[ ! -e "${OUTPUT_ROOT}/${model}" ]] || {
      echo "Refusing existing generation output for ${model}" >&2
      exit 2
    }
    [[ ! -e "${MANIFEST_ROOT}/${model}" ]] || {
      echo "Refusing existing manifest output for ${model}" >&2
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
RUN_LOG="${OUTPUT_ROOT}/audit/generation_kjob.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[campaign] host=$(hostname) train_seed=6666 inference_seed=${SEED} models=${models[*]}"
nvidia-smi

for index in "${!models[@]}"; do
  model="${models[$index]}"
  step="${steps[$index]}"
  checkpoint="$(find "${PROJECT}/models" -maxdepth 1 -type d -name "checkpoint_epoch_*_step_${step}" -print -quit)"
  [[ -n "${checkpoint}" ]] || { echo "Missing ${model} checkpoint" >&2; exit 1; }
  weight_dir="${checkpoint}/transformer"
  [[ -s "${weight_dir}/config.json" ]] || { echo "Missing ${model} config" >&2; exit 1; }
  [[ -s "${weight_dir}/cic_transport_config.json" ]] || {
    echo "Missing ${model} CIC-Transport sidecar" >&2
    exit 1
  }
  compgen -G "${weight_dir}/diffusion_pytorch_model.*" >/dev/null || {
    echo "Missing ${model} weights" >&2
    exit 1
  }

  probe="${PROBE_ROOT}/${model}"
  mkdir -p "${probe}"
  if [[ ! -e "${probe}/transformer" ]]; then
    ln -s "${weight_dir}" "${probe}/transformer"
    ln -s "${GAGI}/giga_world_0_video_pretrain/text_encoder" "${probe}/text_encoder"
    ln -s "${GAGI}/giga_world_0_video_pretrain/vae" "${probe}/vae"
  fi

  echo "[generate] model=${model} inference_seed=${SEED} checkpoint=${checkpoint}"
  "${PYTHON}" eveworld/evaluation/eval175_seed_sharded_dispatch.py \
    --model "${model}" \
    --model-dir "${probe}" \
    --data-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${OUTPUT_ROOT}" \
    --chain-name "seed004_${model}" \
    --seeds "${SEED}" \
    --gpu-count 8 \
    --python "${PYTHON}" \
    --worker-script "${WORKER}" \
    --num-inference-steps 30 \
    --num-frames 93 \
    --fps 16 \
    --height 480 \
    --width 768 \
    --skip-existing

  echo "[audit] model=${model}"
  "${PYTHON}" eveworld/evaluation/eval175_multiseed_prepare.py \
    --generation-root "${OUTPUT_ROOT}/${model}" \
    --input-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${MANIFEST_ROOT}/${model}" \
    --model-name "${model}" \
    --seeds "${SEED}"
  "${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready"] is True and d["manifest_count"] == 126 and not d["errors"]' \
    "${MANIFEST_ROOT}/${model}/prepare_report.json"
done

date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/COMPLETE"
echo "[campaign] seed6666 s150/s200/s250 seed004 generation and audit complete"
