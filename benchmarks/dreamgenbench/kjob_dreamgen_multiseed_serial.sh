#!/usr/bin/env bash
#SBATCH --job-name=dreamgen_multiseed
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host without GPUs." >&2
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

TASK_ID="${TASK_ID:-task001}"
REQUEST_ID="${REQUEST_ID:-gr1_behavior_23}"
SOURCE_JSON="${SOURCE_JSON:-${GAGI}/gr1_dreamgen_eval/giga_input/eval175_gr1_behavior.json}"
SOURCE_VIDEO="${SOURCE_VIDEO:-${GAGI}/eve_v2_outputs/eval175_gen/t4g_wmapA_pre_s250/gr1_behavior/generated_only/23_Use_the_right_hand_to_pick_up_long_reach_lighter_to_light_the_candle.mp4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/20seed/dreamgenbench}"
MODELS="${MODELS:-t4g_wmapA_pre_seed42_s150 t4g_wmapA_pre_seed42_s250}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61}"

export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}/${TASK_ID}"
RUN_LOG="${OUTPUT_ROOT}/${TASK_ID}/kjob.log"

read -r -a MODEL_ARRAY <<< "${MODELS}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
SKIP_ARGS=()
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  SKIP_ARGS=(--skip-existing)
fi

{
  echo "host=$(hostname) task=${TASK_ID} request=${REQUEST_ID}"
  echo "models=${MODELS}"
  echo "seeds=${SEEDS}"
  echo "output=${OUTPUT_ROOT}/${TASK_ID}"
  nvidia-smi
  "${PYTHON}" eveworld/evaluation/dreamgen_multiseed_dispatch.py \
    --task-id "${TASK_ID}" \
    --request-id "${REQUEST_ID}" \
    --source-json "${SOURCE_JSON}" \
    --source-video "${SOURCE_VIDEO}" \
    --output-root "${OUTPUT_ROOT}" \
    --models "${MODEL_ARRAY[@]}" \
    --seeds "${SEED_ARRAY[@]}" \
    --gpu-count 8 \
    --python "${PYTHON}" \
    --worker-script "${EVEWORLD_ROOT}/eveworld/evaluation/dreamgen_multiseed_worker.py" \
    --num-inference-steps "${NUM_INFERENCE_STEPS:-30}" \
    --num-frames "${NUM_FRAMES:-93}" \
    --fps "${FPS:-16}" \
    --height "${HEIGHT:-480}" \
    --width "${WIDTH:-768}" \
    "${SKIP_ARGS[@]}"
} 2>&1 | tee -a "${RUN_LOG}"
