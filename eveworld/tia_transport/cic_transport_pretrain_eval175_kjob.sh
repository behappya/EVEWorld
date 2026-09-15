#!/usr/bin/env bash
#SBATCH --job-name=eve_cic_pretrain_eval175
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${CAMPAIGN_ROOT}/eval175_seed004_pretrain_raw_s000}"
PROBE_ROOT="${PROBE_ROOT:-${CAMPAIGN_ROOT}/eval175_seed004_pretrain_raw_s000_probes}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${CAMPAIGN_ROOT}/eval175_seed004_pretrain_raw_s000_eval}"
SEED="${SEED:-4}"
RESUME="${RESUME:-0}"
MODEL="pretrain_raw_s000"
WEIGHT_DIR="${GAGI}/giga_world_0_video_pretrain/transformer"
STANDARD_WORKER="${EVEWORLD_ROOT}/eveworld/evaluation/eval175_multiseed_worker.py"

[[ "$(realpath -m "${OUTPUT_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_seed004_pretrain_raw_s000" ]] || {
  echo "Refusing unexpected OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
}
[[ "${SEED}" == "4" ]] || { echo "This comparison is frozen to seed004" >&2; exit 2; }
[[ -s "${OUTPUT_ROOT}/audit/PREPARED" ]] || {
  echo "Missing launcher preparation marker" >&2
  exit 2
}
[[ -s "${WEIGHT_DIR}/config.json" ]] || { echo "Missing pretrained config" >&2; exit 1; }
compgen -G "${WEIGHT_DIR}/diffusion_pytorch_model.*" >/dev/null || {
  echo "Missing pretrained weights" >&2
  exit 1
}

if [[ "${RESUME}" != "1" ]]; then
  [[ ! -e "${PROBE_ROOT}" ]] || { echo "Refusing existing probe root" >&2; exit 2; }
  [[ ! -e "${MANIFEST_ROOT}" ]] || { echo "Refusing existing manifest root" >&2; exit 2; }
  [[ ! -e "${OUTPUT_ROOT}/${MODEL}" ]] || {
    echo "Refusing existing generation output for ${MODEL}" >&2
    exit 2
  }
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

echo "[campaign] host=$(hostname) seed=${SEED} model=${MODEL}"
nvidia-smi

probe="${PROBE_ROOT}/${MODEL}"
mkdir -p "${probe}"
if [[ ! -e "${probe}/transformer" ]]; then
  ln -s "${WEIGHT_DIR}" "${probe}/transformer"
  ln -s "${GAGI}/giga_world_0_video_pretrain/text_encoder" "${probe}/text_encoder"
  ln -s "${GAGI}/giga_world_0_video_pretrain/vae" "${probe}/vae"
fi

echo "[generate] model=${MODEL} seed=${SEED} transformer=${WEIGHT_DIR}"
"${PYTHON}" eveworld/evaluation/eval175_seed_sharded_dispatch.py \
  --model "${MODEL}" \
  --model-dir "${probe}" \
  --data-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
  --output-root "${OUTPUT_ROOT}" \
  --chain-name "seed004_${MODEL}" \
  --seeds "${SEED}" \
  --gpu-count 8 \
  --python "${PYTHON}" \
  --worker-script "${STANDARD_WORKER}" \
  --num-inference-steps 30 \
  --num-frames 93 \
  --fps 16 \
  --height 480 \
  --width 768 \
  --skip-existing

echo "[audit] model=${MODEL}"
"${PYTHON}" eveworld/evaluation/eval175_multiseed_prepare.py \
  --generation-root "${OUTPUT_ROOT}/${MODEL}" \
  --input-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
  --output-root "${MANIFEST_ROOT}/${MODEL}" \
  --model-name "${MODEL}" \
  --seeds "${SEED}"
"${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready"] is True and d["manifest_count"] == 126 and not d["errors"]' \
  "${MANIFEST_ROOT}/${MODEL}/prepare_report.json"

date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/GENERATION_COMPLETE"
echo "[campaign] pretrained baseline generated and audited"
