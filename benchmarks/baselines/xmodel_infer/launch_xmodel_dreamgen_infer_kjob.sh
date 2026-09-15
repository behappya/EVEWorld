#!/usr/bin/env bash
set -euo pipefail

# 提交跨模型 DreamGen I2V 推理到 GPU kjob（单卡）。
# 复用 giga-world-0 的 submit_gigaworld0_kjob.sh 提交器。
#
# 用法（smoke，先跑 4 条确认不崩）：
#   MODEL_FAMILY=wan_ti2v \
#   MODEL_PATH=/data/datasets/gagi/xmodels/wan22_ti2v_5b \
#   RUN_NAME=wan22_ti2v_5b_5p8s_smoke DATA_LIMIT=4 NUM_FRAMES=93 \
#   bash launch_xmodel_dreamgen_infer_kjob.sh
#
# 用法（全 92 条，某时长档）：
#   MODEL_FAMILY=cogvideox \
#   MODEL_PATH=/data/datasets/gagi/xmodels/cogvideox15_5b_i2v \
#   RUN_NAME=cogvideox15_5b_9p8s NUM_FRAMES=157 \
#   bash launch_xmodel_dreamgen_infer_kjob.sh
#
# 时长档 -> NUM_FRAMES @16fps: 5.8s=93  9.8s=157  15.8s=253

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RL_DIR="${RL_DIR:-/home/jovyan/new_rl/rl}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/kjob_xmodel_dreamgen_infer.sh}"

: "${MODEL_FAMILY:?Set MODEL_FAMILY=wan|wan_ti2v|cogvideox|cosmos}"
: "${MODEL_PATH:?Set MODEL_PATH=/data/.../xmodels/<model>}"

if [[ ! -d "${RL_DIR}" ]]; then echo "Missing RL_DIR: ${RL_DIR}" >&2; exit 1; fi
if [[ ! -f "${JOB_SCRIPT}" ]]; then echo "Missing JOB_SCRIPT: ${JOB_SCRIPT}" >&2; exit 1; fi
if [[ ! -d "${MODEL_PATH}" ]]; then echo "Missing MODEL_PATH: ${MODEL_PATH}（先下权重）" >&2; exit 1; fi

SCRIPT_ARGS=("REPO_DIR=${REPO_DIR}")
maybe_export() { local n="$1"; if [[ -n "${!n:-}" ]]; then SCRIPT_ARGS+=("${n}=${!n}"); fi; }

for name in \
  INFER_DIR CONDA_SH CONDA_ENV PYTHON_BIN \
  MODEL_FAMILY MODEL_PATH XMODEL_EVAL_ROOT DATA_PATH \
  RUN_NAME SAVE_DIR LOG_FILE SUMMARY_PATH \
  NUM_FRAMES FPS HEIGHT WIDTH NUM_INFERENCE_STEPS GUIDANCE_SCALE SEED \
  DATA_LIMIT DTYPE NEGATIVE_PROMPT \
  GPU_IDS CUDA_VISIBLE_DEVICES HF_HOME ; do
  maybe_export "${name}"
done

echo "Submitting xmodel DreamGen infer kjob"
echo "  family:     ${MODEL_FAMILY}"
echo "  model_path: ${MODEL_PATH}"
echo "  job_script: ${JOB_SCRIPT}"
echo "  run_name:   ${RUN_NAME:-<auto>}"
echo "  data_limit: ${DATA_LIMIT:-0}   frames: ${NUM_FRAMES:-93}"

if command -v kjobctl >/dev/null 2>&1; then
  KJOB_CMD=(kjobctl)
elif command -v pixi >/dev/null 2>&1; then
  cd "${RL_DIR}"; KJOB_CMD=(pixi run kjobctl)
else
  echo "Neither kjobctl nor pixi available." >&2; exit 1
fi

exec "${KJOB_CMD[@]}" create slurm \
  --pod-template-label vllm-metrics=true \
  -- \
  --chdir "${REPO_DIR}" \
  "${JOB_SCRIPT}" \
  "${SCRIPT_ARGS[@]}" \
  "$@"
