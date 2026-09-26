#!/usr/bin/env bash
#SBATCH --job-name=flowwam_ckpt_screen
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

ROLE="${ROLE:?ROLE=A or ROLE=B is required}"
[[ "${ROLE}" == "A" || "${ROLE}" == "B" ]] || { echo "invalid ROLE=${ROLE}" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT:-${EVEWORLD_ROOT}}"
CODE_ROOT="${REPO_ROOT}/giga-world-0"
FLOWWAM_ROOT="${FLOWWAM_ROOT:-/home/jovyan/FlowWAM}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
FLOWWAM_PYTHON="${FLOWWAM_PYTHON:-/home/jovyan/miniconda/envs/flowwam/bin/python}"
METRIC_PYTHON="${METRIC_PYTHON:-/home/jovyan/miniconda/envs/EVEWorld/bin/python}"
CAMPAIGN="${CAMPAIGN:-/data/datasets/gagi/flowwam/checkpoint_screen_v1}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/data/datasets/gagi/flowwam/five_row_r250_v1/manifest.json}"
DEV_MANIFEST="${CAMPAIGN}/manifests/dev_episode45.json"
TEST_MANIFEST="${CAMPAIGN}/manifests/test_episode46_49.json"
N_GPU="${N_GPU:-8}"

mkdir -p "${CAMPAIGN}/logs" "${CAMPAIGN}/status" "${CAMPAIGN}/videos" \
  "${CAMPAIGN}/metrics" "${CAMPAIGN}/aggregate" "${CAMPAIGN}/manifests"
exec > >(tee -a "${CAMPAIGN}/logs/node_${ROLE,,}.log") 2>&1

CURRENT_STAGE=initialization
on_error() {
  local rc=$?
  printf 'role=%s\nstage=%s\nrc=%s\nutc=%s\n' \
    "${ROLE}" "${CURRENT_STAGE}" "${rc}" "$(date -u +%FT%TZ)" \
    >"${CAMPAIGN}/status/node_${ROLE,,}.failed"
  exit "${rc}"
}
trap on_error ERR

if [[ ! -s "${DEV_MANIFEST}" || ! -s "${TEST_MANIFEST}" ]]; then
  "${METRIC_PYTHON}" "${CODE_ROOT}/benchmarks/robotwin_flowwam/build_flowwam_checkpoint_screen_manifest.py" \
    --input "${SOURCE_MANIFEST}" --dev-output "${DEV_MANIFEST}" \
    --test-output "${TEST_MANIFEST}"
fi

if [[ "${ROLE}" == "A" ]]; then
  CANDIDATES=(
    "sft_e0|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_sft/epoch-0.safetensors"
    "sft_e1|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_sft/epoch-1.safetensors"
    "sft_e2|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_sft/epoch-2.safetensors"
    "igr_e0|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_igr/epoch-0.safetensors"
    "igr_e1|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_igr/epoch-1.safetensors"
    "igr_e2|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_igr/epoch-2.safetensors"
  )
else
  CANDIDATES=(
    "tia_e0|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_tia/epoch-0.safetensors"
    "tia_e1|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_tia/epoch-1.safetensors"
    "tia_e2|/data/datasets/gagi/flowwam/five_row_r250_v1/checkpoints/five_row_tia/epoch-2.safetensors"
    "eve_s100|/data/datasets/gagi/flowwam/train_runs/arm_eve/step-100.safetensors"
    "eve_s200|/data/datasets/gagi/flowwam/train_runs/arm_eve/step-200.safetensors"
    "eve_e0|/data/datasets/gagi/flowwam/train_runs/arm_eve/epoch-0.safetensors"
  )
fi

source "${CONDA_SH}"
conda activate flowwam
export PYTHONUNBUFFERED=1 FLOWWAM_ROOT N_GPU
export PYTHONPATH="${CODE_ROOT}:${FLOWWAM_ROOT}:${FLOWWAM_ROOT}/inference:${FLOWWAM_ROOT}/training:${PYTHONPATH:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

for spec in "${CANDIDATES[@]}"; do
  IFS='|' read -r tag checkpoint <<<"${spec}"
  [[ -s "${checkpoint}" ]]
  videos="${CAMPAIGN}/videos/${tag}"
  metric_dir="${CAMPAIGN}/metrics/${tag}"
  mkdir -p "${videos}" "${metric_dir}"

  CURRENT_STAGE="generate_${tag}"
  count="$(find "${videos}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  if [[ "${count}" != "50" ]]; then
    cd "${FLOWWAM_ROOT}/training"
    "${FLOWWAM_PYTHON}" "${CODE_ROOT}/eveworld/flowwam_port/arm_generate_dispatch.py" \
      --arm-ckpt "${checkpoint}" --out "${videos}" --manifest "${DEV_MANIFEST}" \
      --flow-cond robot_only --cfg-scale 5.0 --steps 40 --tia-inject off \
      --full-traj direct --seed 42 \
      >"${CAMPAIGN}/logs/generate_${tag}.log" 2>&1
  fi
  [[ "$(find "${videos}" -maxdepth 1 -type f -name '*.mp4' | wc -l)" == "50" ]]

  CURRENT_STAGE="evaluate_${tag}"
  "${METRIC_PYTHON}" "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_fidelity_metrics.py" \
    --manifest "${DEV_MANIFEST}" --variant "${tag}=${videos}" \
    --output "${metric_dir}/psnr_ssim.json" --workers 32 --expected-count 50
  printf 'utc=%s\ncheckpoint=%s\n' "$(date -u +%FT%TZ)" "${checkpoint}" \
    >"${CAMPAIGN}/status/${tag}.complete"
done

CURRENT_STAGE=aggregate
"${METRIC_PYTHON}" "${CODE_ROOT}/benchmarks/robotwin_flowwam/aggregate_flowwam_checkpoint_screen.py" \
  --metrics-root "${CAMPAIGN}/metrics" \
  --output-json "${CAMPAIGN}/aggregate/dev50_ranking.json" \
  --output-csv "${CAMPAIGN}/aggregate/dev50_ranking.csv"
printf 'role=%s\nutc=%s\n' "${ROLE}" "$(date -u +%FT%TZ)" \
  >"${CAMPAIGN}/status/node_${ROLE,,}.complete"
