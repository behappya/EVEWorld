#!/usr/bin/env bash
#SBATCH --job-name=eval175_qwen
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
DREAMGEN_REPO="${DREAMGEN_REPO:-/home/jovyan/gagibench/GR00T-Dreams}"
EVAL_PYTHON="${EVAL_PYTHON:-/data/datasets/gagi/envs/dreamgenbench_eval_venv/bin/python}"
MANIFEST="${MANIFEST:?Set MANIFEST to one audited model manifest}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen2.5-VL-7B-Instruct}"
METRICS="${METRICS:-qwen_if,pa_i}"
SEED="${SEED:-0}"
START_OFFSET="${START_OFFSET:-0}"
LIMIT="${LIMIT:-0}"
RUN_LOG="${RUN_LOG:-${OUTPUT_DIR}/qwen.log}"
GPU_IDS="${GPU_IDS:-0}"

export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/datasets/gagi/.cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${DREAMGEN_REPO}:${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS// /,}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${RUN_LOG}")"
exec > >(tee -a "${RUN_LOG}") 2>&1

if [[ ! -x "${EVAL_PYTHON}" ]]; then
  echo "Missing eval Python: ${EVAL_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${DREAMGEN_REPO}/dreamgenbench" ]]; then
  echo "Missing DreamGenBench repo: ${DREAMGEN_REPO}" >&2
  exit 1
fi

echo "EVAL-175 official Qwen evaluation"
echo "Manifest:   ${MANIFEST}"
echo "Output:     ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Metrics:    ${METRICS}"
echo "Seed:       ${SEED}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

"${EVAL_PYTHON}" "${EVEWORLD_ROOT}/eveworld/evaluation/eval175_qwen_local.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --metrics "${METRICS}" \
  --device cuda \
  --seed "${SEED}" \
  --start-offset "${START_OFFSET}" \
  --limit "${LIMIT}"

