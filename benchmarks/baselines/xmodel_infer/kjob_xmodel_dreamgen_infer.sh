#!/usr/bin/env bash
#SBATCH --job-name=xmodel_dreamgen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

# 跨模型 DreamGen 批量 I2V 推理 payload（在 GPU kjob pod 内运行）。
# 由 launch_xmodel_dreamgen_infer_kjob.sh 提交；本机（无 GPU）会拒绝直接运行。
# 覆盖参数以 KEY=VALUE 传入。

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host (no GPU). Submit via launch_xmodel_dreamgen_infer_kjob.sh" >&2
  exit 2
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
CONDA_ENV="${CONDA_ENV:-giga_world1}"   # diffusers 0.39.0，四个 I2V pipeline 齐全
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- 必填：模型与输出 ----
MODEL_FAMILY="${MODEL_FAMILY:?Set MODEL_FAMILY=wan|wan_ti2v|cogvideox|cosmos}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH=/data/.../xmodels/<model>}"
XMODEL_EVAL_ROOT="${XMODEL_EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/xmodel_eval}"
DATA_PATH="${DATA_PATH:-/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
RUN_NAME="${RUN_NAME:-${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${XMODEL_EVAL_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/run.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${SAVE_DIR}/generation_summary.json}"

# ---- 生成参数（时长档：93=5.8s / 157=9.8s / 253=15.8s @16fps）----
NUM_FRAMES="${NUM_FRAMES:-93}"
FPS="${FPS:-16}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SEED="${SEED:-6666}"
DATA_LIMIT="${DATA_LIMIT:-0}"          # smoke 时设 4
DTYPE="${DTYPE:-bf16}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# 8 卡数据并行(92条按卡切分)。python 内部用 --gpu-ids 绑定各进程到各卡。
# 主动 unset CUDA_VISIBLE_DEVICES：否则若外部环境/透传把它设成单卡, 会遮住其余 7 张,
# 导致 spawn 的进程绑不到 cuda:1~7。想限制可见卡请改用 GPU_IDS。
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
unset CUDA_VISIBLE_DEVICES

export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${SAVE_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${INFER_DIR}"

echo "============================================"
echo "xmodel DreamGen infer"
echo "  host        = $(hostname)"
echo "  family      = ${MODEL_FAMILY}"
echo "  model_path  = ${MODEL_PATH}"
echo "  data_path   = ${DATA_PATH}"
echo "  save_dir    = ${SAVE_DIR}"
echo "  frames/fps  = ${NUM_FRAMES}/${FPS}   size=${WIDTH}x${HEIGHT}"
echo "  steps/cfg   = ${NUM_INFERENCE_STEPS}/${GUIDANCE_SCALE}  seed=${SEED} dtype=${DTYPE}"
echo "  data_limit  = ${DATA_LIMIT}"
echo "  gpu_ids     = ${GPU_IDS}"
echo "============================================"
nvidia-smi -L 2>&1 || true

"${PYTHON_BIN}" xmodel_dreamgen_infer.py \
  --model-family "${MODEL_FAMILY}" \
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
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --summary-path "${SUMMARY_PATH}"

echo "Inference done. Outputs in ${SAVE_DIR}"
