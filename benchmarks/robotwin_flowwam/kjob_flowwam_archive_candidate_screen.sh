#!/usr/bin/env bash
#SBATCH --job-name=flowwam_archive_screen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -Eeuo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

TAG="${TAG:?TAG is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
CAMPAIGN="${CAMPAIGN:-/data/datasets/gagi/flowwam/checkpoint_screen_v1}"
REPO_ROOT="${REPO_ROOT:-${EVEWORLD_ROOT}}"
CODE_ROOT="${REPO_ROOT}/giga-world-0"
FLOWWAM_ROOT="${FLOWWAM_ROOT:-/home/jovyan/FlowWAM}"
FLOWWAM_PYTHON="${FLOWWAM_PYTHON:-/home/jovyan/miniconda/envs/flowwam/bin/python}"
METRIC_PYTHON="${METRIC_PYTHON:-/home/jovyan/miniconda/envs/giga_models/bin/python}"
DEV_MANIFEST="${CAMPAIGN}/manifests/dev_episode45.json"
VIDEOS="${CAMPAIGN}/videos/${TAG}"
METRICS="${CAMPAIGN}/metrics/${TAG}"

[[ -s "${CHECKPOINT}" && -s "${DEV_MANIFEST}" ]]
mkdir -p "${VIDEOS}" "${METRICS}" "${CAMPAIGN}/logs" "${CAMPAIGN}/status"
exec > >(tee -a "${CAMPAIGN}/logs/archive_${TAG}.log") 2>&1

source /home/jovyan/miniconda/etc/profile.d/conda.sh
conda activate flowwam
export PYTHONUNBUFFERED=1 FLOWWAM_ROOT N_GPU=8
export PYTHONPATH="${CODE_ROOT}:${FLOWWAM_ROOT}:${FLOWWAM_ROOT}/inference:${FLOWWAM_ROOT}/training:${PYTHONPATH:-}"

count="$(find "${VIDEOS}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
if [[ "${count}" != "50" ]]; then
  cd "${FLOWWAM_ROOT}/training"
  "${FLOWWAM_PYTHON}" "${CODE_ROOT}/eveworld/flowwam_port/arm_generate_dispatch.py" \
    --arm-ckpt "${CHECKPOINT}" --out "${VIDEOS}" --manifest "${DEV_MANIFEST}" \
    --flow-cond robot_only --cfg-scale 5.0 --steps 40 --tia-inject off \
    --full-traj direct --seed 42 \
    >"${CAMPAIGN}/logs/archive_generate_${TAG}.log" 2>&1
fi
[[ "$(find "${VIDEOS}" -maxdepth 1 -type f -name '*.mp4' | wc -l)" == "50" ]]

"${METRIC_PYTHON}" "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_fidelity_metrics.py" \
  --manifest "${DEV_MANIFEST}" --variant "${TAG}=${VIDEOS}" \
  --output "${METRICS}/psnr_ssim.json" --workers 32 --expected-count 50
printf 'tag=%s\ncheckpoint=%s\nutc=%s\n' "${TAG}" "${CHECKPOINT}" "$(date -u +%FT%TZ)" \
  >"${CAMPAIGN}/status/${TAG}.complete"
