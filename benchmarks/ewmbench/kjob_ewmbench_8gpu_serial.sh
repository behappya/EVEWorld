#!/usr/bin/env bash
#SBATCH --job-name=ewmbench_serial
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# EWMBench 生成:5 个 GW-0 典型模型 x 3 seeds(=3 samples/episode) x 21 episodes,
# 单节点 8 卡串行链。输出 side-by-side mp4, 由后处理脚本转 EWMBench 帧序列布局。

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
GAGI="${GAGI:-/data/datasets/gagi}"

MODELS="${MODELS:-pretrain round0 t4g_wmapA_pre_seed42_s250 t4g_wmapA_pre_noaug_s50 t4g_wmaponly_s150}"
SEEDS="${SEEDS:-42 43 44}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/ewmbench_gen}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/ewmbench_it2v.json}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
NUM_FRAMES="${NUM_FRAMES:-93}"
FPS="${FPS:-16}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.nv/ComputeCache}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"

model_dir_for() {
  case "$1" in
    pretrain) echo "${GAGI}/giga_world_0_video_pretrain" ;;
    round0) echo "${GAGI}/eve_v2_outputs/anchor_models/round0_ema_st" ;;
    t4g_wmapA_pre_seed42_s250) echo "${GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s250" ;;
    t4g_wmapA_pre_noaug_s50) echo "${GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s50" ;;
    t4g_wmaponly_s150) echo "${GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmaponly_s150" ;;
    t4g_*|anmix*|dpo_*|gr1_2b_dreamgen) echo "${GAGI}/eve_v2_outputs/anchor_models/probe_$1" ;;
    ewm_vanilla_*) echo "${GAGI}/agibot_ewm_vanilla/probes/$1" ;;
    ewm_apre_*) echo "${GAGI}/agibot_ewm_apre/probes/$1" ;;
    ewm_wmaponly_*) echo "${GAGI}/eve_v2_outputs/agibot_ewm_apre/probes/$1" ;;
    ewm_froms250_*) echo "${GAGI}/eve_v2_outputs/agibot_ewm_from_s250/probes/$1" ;;
    ewm_froms150_*) echo "${GAGI}/eve_v2_outputs/agibot_ewm_from_s150/probes/$1" ;;
    ewm_fullp02_*) echo "${GAGI}/eve_v2_outputs/agibot_ewm_full/probes/$1" ;;
    ewm_fullp50_*) echo "${GAGI}/eve_v2_outputs/agibot_ewm_full_paug50/probes/$1" ;;
    *) echo "" ;;
  esac
}

mkdir -p "${OUTPUT_ROOT}"
JOB_LOG="${OUTPUT_ROOT}/ewmbench_serial_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"
echo "host=$(hostname) models=${MODELS} seeds=${SEEDS}" | tee -a "${JOB_LOG}"

overall_rc=0
for model in ${MODELS}; do
  MODEL_DIR="$(model_dir_for "${model}")"
  if [[ -z "${MODEL_DIR}" ]]; then
    echo "Unknown model name: ${model}" | tee -a "${JOB_LOG}" >&2
    overall_rc=1
    continue
  fi
  if [[ ! -f "${MODEL_DIR}/transformer/config.json" ]]; then
    echo "Missing transformer config for ${model}: ${MODEL_DIR}" | tee -a "${JOB_LOG}" >&2
    overall_rc=1
    continue
  fi
  for seed in ${SEEDS}; do
    SAVE_DIR="${OUTPUT_ROOT}/${model}/seed${seed}"
    SUMMARY_PATH="${SAVE_DIR}/generation_summary.json"
    if [[ -f "${SUMMARY_PATH}" ]]; then
      echo "SKIP ${model}/seed${seed}: summary exists" | tee -a "${JOB_LOG}"
      continue
    fi
    mkdir -p "${SAVE_DIR}"
    echo "===== $(date '+%F %T') START ${model}/seed${seed}" | tee -a "${JOB_LOG}"
    python scripts/inference.py \
      --data-path "${DATA_PATH}" \
      --save-dir "${SAVE_DIR}" \
      --transformer-model-path "${MODEL_DIR}/transformer" \
      --text-encoder-model-path "${MODEL_DIR}/text_encoder" \
      --vae-model-path "${MODEL_DIR}/vae" \
      --gpu-ids 0 1 2 3 4 5 6 7 \
      --num-inference-steps "${NUM_INFERENCE_STEPS}" \
      --fps "${FPS}" \
      --num-frames "${NUM_FRAMES}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --seed "${seed}" \
      --summary-path "${SUMMARY_PATH}" >>"${SAVE_DIR}/run.log" 2>&1
    rc=$?
    n_mp4=$(find "${SAVE_DIR}" -maxdepth 1 -name '*.mp4' | wc -l)
    echo "===== $(date '+%F %T') DONE ${model}/seed${seed}: rc=${rc} mp4=${n_mp4}/21" | tee -a "${JOB_LOG}"
    if [[ "${rc}" != "0" ]]; then
      overall_rc=1
    fi
  done
done

echo "===== $(date '+%F %T') ALL DONE rc=${overall_rc}" | tee -a "${JOB_LOG}"
exit "${overall_rc}"
