#!/usr/bin/env bash
#SBATCH --job-name=eve_cic_ms70
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing multi-seed generation on the workspace host." >&2
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
TRAIN_VENV="${TRAIN_VENV:-${GAGI}/envs/giga_world_train_venv}"
PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
CAMPAIGN_ROOT="${GAGI}/eve_v2_outputs/eve_cic_transport_v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CAMPAIGN_ROOT}/eval175_transport_raw_s150_s200_multiseed70}"
PROBE_ROOT="${PROBE_ROOT:-${CAMPAIGN_ROOT}/eval175_transport_raw_s150_s200_multiseed70_probes}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${CAMPAIGN_ROOT}/eval175_transport_raw_s150_s200_multiseed70_eval}"
SEED_START="${SEED_START:?SEED_START is required}"
SEED_END="${SEED_END:?SEED_END is required}"
CHAIN_NAME="${CHAIN_NAME:?CHAIN_NAME is required}"
RESUME="${RESUME:-0}"
DISPATCHER="${EVEWORLD_ROOT}/eveworld/tia_transport/cic_transport_multiseed70_dispatch.py"
WORKER="${EVEWORLD_ROOT}/eveworld/tia_transport/eval175_transport_worker.py"

[[ "$(realpath -m "${OUTPUT_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_transport_raw_s150_s200_multiseed70" ]] || {
  echo "Refusing unexpected OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
}
[[ "$(realpath -m "${PROBE_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_transport_raw_s150_s200_multiseed70_probes" ]] || {
  echo "Refusing unexpected PROBE_ROOT=${PROBE_ROOT}" >&2
  exit 2
}
[[ "$(realpath -m "${MANIFEST_ROOT}")" == "${CAMPAIGN_ROOT}/eval175_transport_raw_s150_s200_multiseed70_eval" ]] || {
  echo "Refusing unexpected MANIFEST_ROOT=${MANIFEST_ROOT}" >&2
  exit 2
}
if [[ "${SEED_START}:${SEED_END}:${CHAIN_NAME}" != "1:35:seed001_035" && \
      "${SEED_START}:${SEED_END}:${CHAIN_NAME}" != "36:70:seed036_070" ]]; then
  echo "Only frozen ranges 1-35 and 36-70 are allowed." >&2
  exit 2
fi
[[ -s "${OUTPUT_ROOT}/audit/PREPARED" ]] || {
  echo "Missing launcher preparation marker" >&2
  exit 2
}

models=(transport_raw_s150 transport_raw_s200)
mapfile -t seeds < <(seq "${SEED_START}" "${SEED_END}")
expected_count="$(( ${#seeds[@]} * 126 ))"

if [[ "${RESUME}" != "1" ]]; then
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
      seed_dir="${OUTPUT_ROOT}/${model}/seed$(printf '%03d' "${seed}")"
      [[ ! -e "${seed_dir}" ]] || {
        echo "Refusing existing assigned seed output: ${seed_dir}" >&2
        exit 2
      }
    done
    [[ ! -e "${MANIFEST_ROOT}/${model}_${CHAIN_NAME}" ]] || {
      echo "Refusing existing manifest range for ${model}_${CHAIN_NAME}" >&2
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

mkdir -p "${OUTPUT_ROOT}/audit" "${MANIFEST_ROOT}"
RUN_LOG="${OUTPUT_ROOT}/audit/generation_${CHAIN_NAME}_kjob.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[campaign] host=$(hostname) chain=${CHAIN_NAME} seeds=${SEED_START}-${SEED_END}"
echo "[campaign] serial_models=${models[*]}"
nvidia-smi

for model in "${models[@]}"; do
  probe="${PROBE_ROOT}/${model}"
  [[ -s "${probe}/transformer/config.json" ]] || { echo "Missing ${model} probe" >&2; exit 1; }
  [[ -s "${probe}/transformer/cic_transport_config.json" ]] || {
    echo "Missing ${model} CIC-Transport sidecar" >&2
    exit 1
  }

  export CIC_TRANSPORT_MODEL_NAME="${model}"
  export CIC_TRANSPORT_MODEL_DIR="${probe}"
  echo "[generate] model=${model} seeds=${SEED_START}-${SEED_END} probe=${probe}"
  "${PYTHON}" "${DISPATCHER}" \
    --model "${model}" \
    --data-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${OUTPUT_ROOT}" \
    --chain-name "${CHAIN_NAME}" \
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

  audit_root="${MANIFEST_ROOT}/${model}_${CHAIN_NAME}"
  echo "[audit] model=${model} seeds=${SEED_START}-${SEED_END}"
  "${PYTHON}" eveworld/evaluation/eval175_multiseed_prepare.py \
    --generation-root "${OUTPUT_ROOT}/${model}" \
    --input-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${audit_root}" \
    --model-name "${model}" \
    --seeds "${seeds[@]}"
  "${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready"] is True and d["manifest_count"] == int(sys.argv[2]) and not d["errors"]' \
    "${audit_root}/prepare_report.json" "${expected_count}"
done

date -u '+%Y-%m-%dT%H:%M:%SZ' >"${OUTPUT_ROOT}/audit/${CHAIN_NAME^^}_COMPLETE"
echo "[campaign] ${CHAIN_NAME} completed s150 then s200 with media audit"
