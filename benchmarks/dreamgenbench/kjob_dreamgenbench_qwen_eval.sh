#!/usr/bin/env bash
#SBATCH --job-name=dreamgen_qwen_eval
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  cat >&2 <<'EOF'
Refusing to run DreamGenBench Qwen eval on the workspace host.

This payload loads Qwen2.5-VL and should run inside a GPU kjob.
Submit it with:
  ./benchmarks/dreamgenbench/launch_dreamgenbench_qwen_eval_kjob.sh
EOF
  exit 2
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
DREAMGEN_EVAL_PYTHON="${DREAMGEN_EVAL_PYTHON:-/data/datasets/gagi/envs/dreamgenbench_eval_venv/bin/python}"
DREAMGEN_REPO="${DREAMGEN_REPO:-/home/jovyan/gagibench/GR00T-Dreams}"
VIDEO_DIR="${VIDEO_DIR:?Set VIDEO_DIR to generated-only DreamGenBench videos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/eval_outputs}"
RUN_NAME="${RUN_NAME:-dreamgen_qwen_eval_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/kjob_logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
ENV_FILE="${ENV_FILE:-${LOG_DIR}/${RUN_NAME}.env}"
GPU_IDS_RAW="${GPU_IDS:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
GPU_MEMORY_SAMPLES="${GPU_MEMORY_SAMPLES:-${OUTPUT_ROOT}/${RUN_NAME}_gpu_memory_samples.csv}"
GPU_MEMORY_PEAK="${GPU_MEMORY_PEAK:-${OUTPUT_ROOT}/${RUN_NAME}_gpu_memory_peak.json}"
GPU_MONITOR_PID=""

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/datasets/gagi/.cache}"
export TORCH_HOME="${TORCH_HOME:-/data/datasets/gagi/torch_cache}"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

IFS=' ,' read -r -a REQUESTED_GPU_IDS_ARGS <<< "${GPU_IDS_RAW}"
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${REQUESTED_GPU_IDS_ARGS[*]}")"
export CUDA_VISIBLE_DEVICES

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${HF_HOME}" "${HF_XET_CACHE}" "${TORCH_HOME}"
touch "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Persistent log: ${RUN_LOG}"

start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; GPU memory monitor disabled."
    return 0
  fi
  mkdir -p "$(dirname "${GPU_MEMORY_SAMPLES}")" "$(dirname "${GPU_MEMORY_PEAK}")"
  "${DREAMGEN_EVAL_PYTHON}" "${REPO_DIR}/scripts/gpu_memory_monitor.py" \
    --samples-csv "${GPU_MEMORY_SAMPLES}" \
    --peak-json "${GPU_MEMORY_PEAK}" \
    --interval-sec "${GPU_MONITOR_INTERVAL}" \
    --label "dreamgenbench_qwen_eval_${RUN_NAME}" &
  GPU_MONITOR_PID="$!"
  echo "GPU memory monitor pid: ${GPU_MONITOR_PID}"
  echo "GPU memory samples:     ${GPU_MEMORY_SAMPLES}"
  echo "GPU memory peak:        ${GPU_MEMORY_PEAK}"
}

if [[ ! -x "${DREAMGEN_EVAL_PYTHON}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  DREAMGEN_EVAL_PYTHON="$(command -v python)"
fi

if [[ ! -x "${DREAMGEN_EVAL_PYTHON}" ]]; then
  echo "Missing DreamGenBench eval python: ${DREAMGEN_EVAL_PYTHON}" >&2
  echo "Run ./benchmarks/dreamgenbench/setup_dreamgenbench_eval_venv.sh first." >&2
  exit 1
fi

if [[ ! -d "${DREAMGEN_REPO}/dreamgenbench" ]]; then
  echo "Missing DreamGenBench repo: ${DREAMGEN_REPO}" >&2
  echo "Run ./benchmarks/dreamgenbench/download_dreamgenbench_code.sh first." >&2
  exit 1
fi
if [[ ! -d "${VIDEO_DIR}" ]]; then
  echo "Missing VIDEO_DIR: ${VIDEO_DIR}" >&2
  exit 1
fi

LOCAL_IP="$(hostname -I | awk '{print $1}')"
cat > "${ENV_FILE}" <<EOF
REPO_DIR=${REPO_DIR}
CONDA_ENV=${CONDA_ENV}
DREAMGEN_EVAL_PYTHON=${DREAMGEN_EVAL_PYTHON}
DREAMGEN_REPO=${DREAMGEN_REPO}
VIDEO_DIR=${VIDEO_DIR}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_NAME=${RUN_NAME}
RUN_LOG=${RUN_LOG}
ENV_FILE=${ENV_FILE}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}
GPU_MEMORY_SAMPLES=${GPU_MEMORY_SAMPLES}
GPU_MEMORY_PEAK=${GPU_MEMORY_PEAK}
HF_HOME=${HF_HOME}
HF_XET_CACHE=${HF_XET_CACHE}
LOCAL_IP=${LOCAL_IP}
GPU_IDS=${GPU_IDS_RAW}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EOF

echo "============================================"
echo "DreamGenBench Qwen eval"
echo "Host:                 $(hostname)"
echo "Video dir:            ${VIDEO_DIR}"
echo "DreamGen repo:        ${DREAMGEN_REPO}"
echo "Eval python:          ${DREAMGEN_EVAL_PYTHON}"
echo "Output root:          ${OUTPUT_ROOT}"
echo "Run log:              ${RUN_LOG}"
echo "GPU memory samples:   ${GPU_MEMORY_SAMPLES}"
echo "GPU memory peak:      ${GPU_MEMORY_PEAK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "============================================"
nvidia-smi || true
start_gpu_monitor

cd "${DREAMGEN_REPO}"
export PYTHONPATH="${DREAMGEN_REPO}:${PYTHONPATH:-}"

"${DREAMGEN_EVAL_PYTHON}" - <<'PY'
from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
import qwen_vl_utils  # noqa: F401
import decord  # noqa: F401
import cv2  # noqa: F401
print("DreamGenBench Qwen eval imports: ok", flush=True)
PY

QWEN_IF_CSV="${OUTPUT_ROOT}/${RUN_NAME}_qwen_if.csv"
PA_I_CSV="${OUTPUT_ROOT}/${RUN_NAME}_pa_i.csv"

"${DREAMGEN_EVAL_PYTHON}" -m dreamgenbench.eval_sr_qwen_whole \
  --video_dir "${VIDEO_DIR}" \
  --output_csv "${QWEN_IF_CSV}" \
  --device cuda

"${DREAMGEN_EVAL_PYTHON}" -m dreamgenbench.eval_qwen_pa \
  --video_dir "${VIDEO_DIR}" \
  --output_csv "${PA_I_CSV}" \
  --device cuda

"${DREAMGEN_EVAL_PYTHON}" "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/summarize_dreamgenbench_csv.py" \
  --qwen-if-csv "${QWEN_IF_CSV}" \
  --pa-i-csv "${PA_I_CSV}" \
  --gpu-peak-json "${GPU_MEMORY_PEAK}" \
  --output-json "${OUTPUT_ROOT}/${RUN_NAME}_summary.json"

echo "DreamGenBench Qwen eval finished."
echo "Qwen-IF CSV: ${QWEN_IF_CSV}"
echo "PA-I CSV:    ${PA_I_CSV}"
echo "Summary:     ${OUTPUT_ROOT}/${RUN_NAME}_summary.json"
echo "GPU peak:    ${GPU_MEMORY_PEAK}"
