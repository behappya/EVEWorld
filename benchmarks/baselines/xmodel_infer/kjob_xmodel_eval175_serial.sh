#!/usr/bin/env bash
#SBATCH --job-name=xmodel_eval175
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# 跨模型 DreamGenBench eval175(126题) 串行生成:一个 8 卡节点内,
# 3 个 xmodel x 3 split 逐个跑(模型间串行, 卡内 python mp 数据并行)。
# 输入为 eval175_gr1_<split>_xmodel.json(request_id 已改为 <idx>_<prompt> 形式,
# 兼容 eval175_prepare.py 的 INDEX_RE)。断点续传:已存在 mp4 跳过;
# split 的 generation_summary.json 存在则整段跳过。

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  case "$(hostname)" in
    coder-workspace-*)
      echo "Refusing to run on workspace host (no GPU)." >&2
      exit 2
      ;;
  esac
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg} (pass overrides as KEY=VALUE)" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
INFER_DIR="${INFER_DIR:-${EVEWORLD_ROOT}/benchmarks/baselines/xmodel_infer}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_world1}"
GAGI="${GAGI:-/data/datasets/gagi}"

INPUT_ROOT="${INPUT_ROOT:-${GAGI}/gr1_dreamgen_eval/giga_input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/gr1_dreamgen_eval/xmodel_eval175}"
XMODELS="${XMODELS:-wan22_ti2v_5b cogvideox15_5b_i2v wan22_i2v_a14b}"
SPLITS="${SPLITS:-gr1_env gr1_object gr1_behavior}"

NUM_FRAMES="${NUM_FRAMES:-93}"
FPS="${FPS:-16}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SEED="${SEED:-42}"
DATA_LIMIT="${DATA_LIMIT:-0}"
DTYPE="${DTYPE:-bf16}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
unset CUDA_VISIBLE_DEVICES

export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

family_for() {
  case "$1" in
    wan22_ti2v_5b) echo "wan_ti2v" ;;
    cogvideox15_5b_i2v) echo "cogvideox" ;;
    wan22_i2v_a14b) echo "wan" ;;
    cosmos_predict25_2b) echo "cosmos" ;;
    cosmos_predict2_2b_v2w) echo "cosmos" ;;
    *) echo "" ;;
  esac
}

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${INFER_DIR}"

mkdir -p "${OUTPUT_ROOT}"
JOB_LOG="${OUTPUT_ROOT}/xmodel_eval175_serial_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"
echo "host=$(hostname) xmodels=${XMODELS} splits=${SPLITS} seed=${SEED} frames=${NUM_FRAMES}" | tee -a "${JOB_LOG}"
nvidia-smi -L 2>&1 | tee -a "${JOB_LOG}"

overall_rc=0
for model in ${XMODELS}; do
  FAMILY="$(family_for "${model}")"
  MODEL_PATH="${GAGI}/xmodels/${model}"
  if [[ -z "${FAMILY}" ]]; then
    echo "Unknown xmodel: ${model}" | tee -a "${JOB_LOG}" >&2
    overall_rc=1
    continue
  fi
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing model path: ${MODEL_PATH}" | tee -a "${JOB_LOG}" >&2
    overall_rc=1
    continue
  fi
  for split in ${SPLITS}; do
    DATA_PATH="${INPUT_ROOT}/eval175_${split}_xmodel.json"
    SAVE_DIR="${OUTPUT_ROOT}/${model}/${split}/generated_only"
    SUMMARY_PATH="${OUTPUT_ROOT}/${model}/${split}/generation_summary.json"
    if [[ -f "${SUMMARY_PATH}" ]]; then
      echo "SKIP ${model}/${split}: summary exists" | tee -a "${JOB_LOG}"
      continue
    fi
    mkdir -p "${SAVE_DIR}"
    echo "===== $(date '+%F %T') START ${model}/${split}" | tee -a "${JOB_LOG}"
    python xmodel_dreamgen_infer.py \
      --model-family "${FAMILY}" \
      --model-path "${MODEL_PATH}" \
      --data-path "${DATA_PATH}" \
      --save-dir "${SAVE_DIR}" \
      --gpu-ids "${GPU_IDS}" \
      --num-frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS}" \
      --guidance-scale "${GUIDANCE_SCALE}" \
      --seed "${SEED}" \
      --data-limit "${DATA_LIMIT}" \
      --dtype "${DTYPE}" \
      --summary-path "${SUMMARY_PATH}" >>"${OUTPUT_ROOT}/${model}/${split}/run.log" 2>&1
    rc=$?
    n_mp4=$(find "${SAVE_DIR}" -maxdepth 1 -name '*.mp4' | wc -l)
    echo "===== $(date '+%F %T') DONE ${model}/${split}: rc=${rc} mp4=${n_mp4}" | tee -a "${JOB_LOG}"
    if [[ "${rc}" != "0" ]]; then
      overall_rc=1
    fi
  done
done

echo "===== $(date '+%F %T') ALL DONE rc=${overall_rc}" | tee -a "${JOB_LOG}"
exit "${overall_rc}"
