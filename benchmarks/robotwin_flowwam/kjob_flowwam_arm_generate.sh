#!/usr/bin/env bash
#SBATCH --job-name=flowwam_arm_gen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-flowwam}"
FLOWWAM_ROOT="${FLOWWAM_ROOT:-/home/jovyan/FlowWAM}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/datasets/gagi/flowwam/train_runs}"
OUT_ROOT="${OUT_ROOT:-/data/datasets/gagi/flowwam/arm_eval}"
RUN_TAG="${RUN_TAG:-arm_v5}"
CKPT_NAME="${CKPT_NAME:-step-200.safetensors}"   # 空格分隔可多个(逐 epoch 扫描)
ARMS="${ARMS:-control eve}"
LIMIT="${LIMIT:-50}"
N_GPU="${N_GPU:-8}"
MANIFEST="${MANIFEST:?set MANIFEST to the episode manifest}"
FLOW_COND="${FLOW_COND:-none}"
CFG_SCALE="${CFG_SCALE:-1.0}"
GEN_STEPS="${GEN_STEPS:-25}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
TIA_INJECT="${TIA_INJECT:-on}"
GEN_OFFSET="${GEN_OFFSET:-0}"
FULL_TRAJ="${FULL_TRAJ:-off}"  # off|on|direct (direct is an experimental one-pass path)
EPISODES="${EPISODES:-}"
RESUME_FROM="${RESUME_FROM:-}"
GEN_SEED="${GEN_SEED:-42}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${FLOWWAM_ROOT}/training"
export PYTHONUNBUFFERED=1 FLOWWAM_ROOT N_GPU
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

cond_tag=""
[[ "${FLOW_COND}" != "none" ]] && cond_tag="_${FLOW_COND}"

overall_rc=0
for arm in ${ARMS}; do
  for ckpt_name in ${CKPT_NAME}; do
    ckpt="${TRAIN_ROOT}/${RUN_TAG}_${arm}/${ckpt_name}"
    tag="${ckpt_name%.safetensors}"
    out="${OUT_ROOT}/${RUN_TAG}_${arm}_${tag}${cond_tag}${OUT_SUFFIX}"
    if [[ ! -s "${ckpt}" ]]; then
      echo "MISSING ckpt: ${ckpt}" >&2
      overall_rc=1
      continue
    fi
    echo "START arm=${arm} ckpt=${ckpt_name} cond=${FLOW_COND} limit=${LIMIT} out=${out}"
    python "${REPO_DIR}/eveworld/flowwam_port/arm_generate_dispatch.py" \
      --arm-ckpt "${ckpt}" \
      --out "${out}" \
      --limit "${LIMIT}" \
      --manifest "${MANIFEST}" \
      --flow-cond "${FLOW_COND}" \
      --cfg-scale "${CFG_SCALE}" \
      --steps "${GEN_STEPS}" \
      --tia-inject "${TIA_INJECT}" \
      --offset "${GEN_OFFSET}" \
      --full-traj "${FULL_TRAJ}" \
      --episodes "${EPISODES}" \
      --resume-from-video "${RESUME_FROM}" \
      --seed "${GEN_SEED}" \
      >"${OUT_ROOT}/gen_${RUN_TAG}_${arm}_${tag}${cond_tag}${OUT_SUFFIX}.off${GEN_OFFSET}.log" 2>&1
    rc=$?
    echo "DONE arm=${arm} ckpt=${ckpt_name} rc=${rc}"
    [[ "${rc}" != "0" ]] && overall_rc=1
  done
done
exit "${overall_rc}"
