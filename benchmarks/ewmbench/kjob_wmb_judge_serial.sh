#!/usr/bin/env bash
#SBATCH --job-name=wmb_judge
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# WorldModelBench 判分:VILA judge(vila-ewm-qwen2-1.5b)对 5 个 GW-0 模型的
# robotics 50 题视频逐模型判分, 单节点串行(judge 单卡, 但独占节点避免碎片化)。
# 非 robotics 的 300 题无视频, evaluation.py 会 warning 跳过, 结果即 robotics 域分。

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

WMB_DIR="${WMB_DIR:-/home/jovyan/gagibench/WorldModelBench}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vila}"
GAGI="${GAGI:-/data/datasets/gagi}"

MODELS="${MODELS:-pretrain round0 t4g_wmapA_pre_seed42_s250 t4g_wmapA_pre_noaug_s50 t4g_wmaponly_s150}"
VIDEO_ROOT="${VIDEO_ROOT:-${GAGI}/eve_v2_outputs/wmb_robotics_gen/eval_videos}"
JUDGE_PATH="${JUDGE_PATH:-${GAGI}/xmodels/judge_vila_ewm_qwen2_1p5b}"
SAVE_ROOT="${SAVE_ROOT:-${GAGI}/eve_v2_outputs/wmb_robotics_eval}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# 尽早建日志并全程 set -x, 便于定位 activate/cd 阶段的早期失败
mkdir -p "${SAVE_ROOT}"
JOB_LOG="${SAVE_ROOT}/wmb_judge_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"
exec > >(tee -a "${JOB_LOG}") 2>&1
set -x
echo "host=$(hostname) models=${MODELS} judge=${JUDGE_PATH}"

# vila env 的 activate.d 钩子引用未定义变量, set -u 下会炸, 激活期间临时关掉
set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u
cd "${WMB_DIR}"

overall_rc=0
for model in ${MODELS}; do
  VDIR="${VIDEO_ROOT}/${model}"
  n=$(find "${VDIR}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
  echo "===== $(date '+%F %T') START ${model} (videos=${n}/50)" | tee -a "${JOB_LOG}"
  if [[ "${n}" != "50" ]]; then
    echo "SKIP ${model}: 视频不足 50" | tee -a "${JOB_LOG}" >&2
    overall_rc=1
    continue
  fi
  python evaluation.py \
    --model_name "${model}" \
    --video_dir "${VDIR}" \
    --judge "${JUDGE_PATH}" \
    --save_name "${SAVE_ROOT}/${model}_results" >>"${SAVE_ROOT}/${model}_judge.log" 2>&1
  rc=$?
  echo "===== $(date '+%F %T') DONE ${model}: rc=${rc}" | tee -a "${JOB_LOG}"
  if [[ "${rc}" != "0" ]]; then
    overall_rc=1
  fi
done

echo "===== $(date '+%F %T') ALL DONE rc=${overall_rc}" | tee -a "${JOB_LOG}"
exit "${overall_rc}"
