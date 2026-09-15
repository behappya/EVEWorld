#!/usr/bin/env bash
#SBATCH --job-name=flowwam_five_row
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
if [[ "${ROLE}" != "A" && "${ROLE}" != "B" ]]; then
  echo "ROLE must be A or B, got ${ROLE}" >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-${EVEWORLD_ROOT}}"
CODE_ROOT="${REPO_ROOT}/giga-world-0"
FLOWWAM_ROOT="${FLOWWAM_ROOT:-/home/jovyan/FlowWAM}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-flowwam}"
METRIC_PYTHON="${METRIC_PYTHON:-/home/jovyan/miniconda/envs/giga_models/bin/python}"
MLR_PYTHON="${MLR_PYTHON:-/home/jovyan/miniconda/envs/flowwam/bin/python}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/data/datasets/gagi/flowwam/five_row_r250_v1}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/data/datasets/gagi/flowwam/heldout_r250_v1/manifest.json}"
STAGE1="${STAGE1:-/data/datasets/gagi/flowwam/checkpoints/flowwam_worldarena_stage1.safetensors}"
EXISTING_EVE_VIDEOS="${EXISTING_EVE_VIDEOS:-/data/datasets/gagi/flowwam/heldout_r250_v1/arm_eve_final_robot_only}"
N_GPU="${N_GPU:-8}"

MANIFEST="${CAMPAIGN_ROOT}/manifest.json"
CHECKPOINT_ROOT="${CAMPAIGN_ROOT}/checkpoints"
VIDEO_ROOT="${CAMPAIGN_ROOT}/videos"
METRIC_ROOT="${CAMPAIGN_ROOT}/metrics"
STATUS_ROOT="${CAMPAIGN_ROOT}/status"
LOG_ROOT="${CAMPAIGN_ROOT}/logs"

mkdir -p "${CHECKPOINT_ROOT}" "${VIDEO_ROOT}" "${METRIC_ROOT}" \
  "${STATUS_ROOT}" "${LOG_ROOT}" "${CAMPAIGN_ROOT}/aggregate"
exec > >(tee -a "${LOG_ROOT}/node_${ROLE,,}.log") 2>&1

CURRENT_STAGE="initialization"
on_error() {
  local rc=$?
  echo "FAILED role=${ROLE} stage=${CURRENT_STAGE} rc=${rc} utc=$(date -u +%FT%TZ)"
  printf '%s\n' "role=${ROLE}" "stage=${CURRENT_STAGE}" "rc=${rc}" \
    "utc=$(date -u +%FT%TZ)" >"${STATUS_ROOT}/node_${ROLE,,}.failed"
  exit "${rc}"
}
trap on_error ERR

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1 FLOWWAM_ROOT N_GPU
export PYTHONPATH="${CODE_ROOT}:${FLOWWAM_ROOT}:${FLOWWAM_ROOT}/inference:${FLOWWAM_ROOT}/training:${CODE_ROOT}/eveworld/pipeline:${PYTHONPATH:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ ! -s "${MANIFEST}" ]]; then
  cp "${SOURCE_MANIFEST}" "${MANIFEST}"
fi
cmp "${SOURCE_MANIFEST}" "${MANIFEST}"
python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" manifest "${MANIFEST}"
[[ "$(sha256sum "${STAGE1}" | cut -c1-24)" == "e211e32b6b79b293f7dec1a7" ]]

python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" protocol \
  "${CAMPAIGN_ROOT}/protocol.node_${ROLE,,}.json" "${MANIFEST}" "${STAGE1}"

mark_stage() {
  CURRENT_STAGE="$1"
  printf '%s\n' "${CURRENT_STAGE}" >"${STATUS_ROOT}/node_${ROLE,,}.stage"
  echo "STAGE role=${ROLE} name=${CURRENT_STAGE} utc=$(date -u +%FT%TZ)"
}

train_variant() {
  local variant="$1"
  local out="${CHECKPOINT_ROOT}/five_row_${variant}"
  local marker="${STATUS_ROOT}/${variant}.train.complete"
  if [[ -s "${marker}" && -s "${out}/final.safetensors" ]]; then
    echo "SKIP completed training variant=${variant}"
    return
  fi
  mark_stage "train_${variant}"
  mkdir -p "${out}"
  cd "${FLOWWAM_ROOT}/training"
  accelerate launch --num_processes "${N_GPU}" --mixed_precision bf16 \
    "${CODE_ROOT}/eveworld/flowwam_port/eve_flowwam_train.py" \
    --arm "${variant}" \
    --output-path "${out}" \
    --num-epochs 4 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --save-steps 999999 \
    --flow-mode robot_only \
    --full-offset off \
    --t-lat-win 8 \
    --paste-prob 0.5 \
    --l-star 12 \
    --tia-loss-weight 0.1 \
    --lora-rank 32 \
    >"${out}/train.log" 2>&1
  [[ -s "${out}/final.safetensors" ]]
  python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" train \
    "${out}/train_args.json" "${variant}"
  printf '%s\n' "utc=$(date -u +%FT%TZ)" "checkpoint=${out}/final.safetensors" >"${marker}"
}

smoke_variant() {
  local variant="$1"
  local ckpt="${CHECKPOINT_ROOT}/smoke_${variant}"
  local videos="${VIDEO_ROOT}/smoke_${variant}"
  local marker="${STATUS_ROOT}/${variant}.smoke.complete"
  if [[ -s "${marker}" ]]; then
    echo "SKIP completed smoke variant=${variant}"
    return
  fi
  mark_stage "smoke_train_${variant}"
  mkdir -p "${ckpt}" "${videos}"
  cd "${FLOWWAM_ROOT}/training"
  accelerate launch --num_processes "${N_GPU}" --mixed_precision bf16 \
    "${CODE_ROOT}/eveworld/flowwam_port/eve_flowwam_train.py" \
    --arm "${variant}" --output-path "${ckpt}" \
    --num-epochs 1 --samples-per-epoch 8 --lr-max-steps 1 \
    --log-every 1 \
    --learning-rate 1e-4 --save-steps 999999 --flow-mode robot_only \
    --full-offset off --t-lat-win 8 --paste-prob 0.5 \
    --l-star 12 --tia-loss-weight 0.1 --lora-rank 32 \
    >"${ckpt}/train.log" 2>&1
  [[ -s "${ckpt}/final.safetensors" ]]
  if [[ "${variant}" == "igr" ]]; then
    ! grep -q 'tia\[out_norm=' "${ckpt}/train.log"
  elif [[ "${variant}" == "tia" ]]; then
    grep -q 'tia\[out_norm=' "${ckpt}/train.log"
    grep -q 'pasted=0' "${ckpt}/train.log"
  fi
  mark_stage "smoke_generate_${variant}"
  cd "${FLOWWAM_ROOT}/training"
  N_GPU=2 python "${CODE_ROOT}/eveworld/flowwam_port/arm_generate_dispatch.py" \
    --arm-ckpt "${ckpt}/final.safetensors" --out "${videos}" \
    --manifest "${MANIFEST}" --limit 2 --flow-cond robot_only \
    --cfg-scale 5.0 --steps 40 --tia-inject off --full-traj direct --seed 42 \
    >"${LOG_ROOT}/smoke_generate_${variant}.log" 2>&1
  [[ "$(find "${videos}" -maxdepth 1 -type f -name '*.mp4' | wc -l)" == "2" ]]
  printf '%s\n' "utc=$(date -u +%FT%TZ)" >"${marker}"
}

smoke_stage1() {
  local videos="${VIDEO_ROOT}/smoke_stage1"
  local marker="${STATUS_ROOT}/stage1.smoke.complete"
  if [[ -s "${marker}" ]]; then
    echo "SKIP completed Stage-1 smoke"
    return
  fi
  mark_stage "smoke_generate_stage1"
  mkdir -p "${videos}"
  cd "${FLOWWAM_ROOT}/training"
  N_GPU=1 python "${CODE_ROOT}/eveworld/flowwam_port/arm_generate_dispatch.py" \
    --out "${videos}" --manifest "${MANIFEST}" --limit 1 \
    --flow-cond robot_only --cfg-scale 5.0 --steps 40 --tia-inject off \
    --full-traj direct --seed 42 >"${LOG_ROOT}/smoke_generate_stage1.log" 2>&1
  [[ "$(find "${videos}" -maxdepth 1 -type f -name '*.mp4' | wc -l)" == "1" ]]
  printf '%s\n' "utc=$(date -u +%FT%TZ)" >"${marker}"
}

generate_variant() {
  local variant="$1"
  local ckpt="${2:-}"
  local videos="${VIDEO_ROOT}/${variant}"
  local marker="${STATUS_ROOT}/${variant}.generate.complete"
  if [[ -s "${marker}" ]]; then
    python "${CODE_ROOT}/benchmarks/robotwin_flowwam/audit_flowwam_variant.py" \
      --manifest "${MANIFEST}" --video-dir "${videos}" \
      --output "${STATUS_ROOT}/${variant}.media.json" >/dev/null
    echo "SKIP completed generation variant=${variant}"
    return
  fi
  mark_stage "generate_${variant}"
  mkdir -p "${videos}"
  cd "${FLOWWAM_ROOT}/training"
  local ckpt_args=()
  if [[ -n "${ckpt}" ]]; then
    [[ -s "${ckpt}" ]]
    ckpt_args=(--arm-ckpt "${ckpt}")
  fi
  N_GPU="${N_GPU}" python "${CODE_ROOT}/eveworld/flowwam_port/arm_generate_dispatch.py" \
    "${ckpt_args[@]}" --out "${videos}" --manifest "${MANIFEST}" \
    --flow-cond robot_only --cfg-scale 5.0 --steps 40 --tia-inject off \
    --full-traj direct --seed 42 \
    >"${LOG_ROOT}/generate_${variant}.log" 2>&1
  python "${CODE_ROOT}/benchmarks/robotwin_flowwam/audit_flowwam_variant.py" \
    --manifest "${MANIFEST}" --video-dir "${videos}" \
    --output "${STATUS_ROOT}/${variant}.media.json"
  printf '%s\n' "utc=$(date -u +%FT%TZ)" "videos=${videos}" >"${marker}"
}

evaluate_variant() {
  local variant="$1"
  local videos="${VIDEO_ROOT}/${variant}"
  local out="${METRIC_ROOT}/${variant}"
  local marker="${STATUS_ROOT}/${variant}.eval.complete"
  if [[ -s "${marker}" ]]; then
    echo "SKIP completed evaluation variant=${variant}"
    return
  fi
  mark_stage "evaluate_${variant}_psnr_ssim"
  mkdir -p "${out}"
  if [[ -s "${out}/psnr_ssim.json" ]] && \
     python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" psnr \
       "${out}/psnr_ssim.json" "${variant}" >/dev/null 2>&1; then
    echo "SKIP completed PSNR/SSIM variant=${variant}"
  else
    "${METRIC_PYTHON}" "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_fidelity_metrics.py" \
      --manifest "${MANIFEST}" --variant "${variant}=${videos}" \
      --output "${out}/psnr_ssim.json" --workers 32
    python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" psnr \
      "${out}/psnr_ssim.json" "${variant}"
  fi

  mark_stage "evaluate_${variant}_lpips_epe"
  if [[ -s "${out}/lpips_flow_epe/summary.json" ]] && \
     python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" lpips \
       "${out}/lpips_flow_epe/summary.json" "${variant}" >/dev/null 2>&1; then
    echo "SKIP completed LPIPS/Flow-EPE variant=${variant}"
  else
    "${METRIC_PYTHON}" "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_lpips_flow_epe.py" \
      --manifest "${MANIFEST}" --variant "${variant}=${videos}" \
      --output-dir "${out}/lpips_flow_epe" --num-shards 8
    python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" lpips \
      "${out}/lpips_flow_epe/summary.json" "${variant}"
  fi

  mark_stage "evaluate_${variant}_mlr"
  if [[ -s "${out}/mlr/summary.json" ]] && \
     python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" mlr \
       "${out}/mlr/summary.json" "${variant}" >/dev/null 2>&1; then
    echo "SKIP completed MLR variant=${variant}"
  else
    "${MLR_PYTHON}" "${CODE_ROOT}/benchmarks/worldarena/mlr_flowwam_variants.py" \
      --manifest "${MANIFEST}" --variant "${variant}=${videos}" \
      --output-dir "${out}/mlr" --num-shards 8
    python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" mlr \
      "${out}/mlr/summary.json" "${variant}"
  fi
  printf '%s\n' "utc=$(date -u +%FT%TZ)" "metrics=${out}" >"${marker}"
}

evaluate_mlr_only() {
  local variant="$1"
  local videos="$2"
  local out="${METRIC_ROOT}/${variant}/mlr"
  mark_stage "evaluate_${variant}_mlr"
  mkdir -p "${out}"
  if [[ -s "${out}/summary.json" ]] && \
     python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" mlr \
       "${out}/summary.json" "${variant}" >/dev/null 2>&1; then
    echo "SKIP completed MLR variant=${variant}"
    return
  fi
  "${MLR_PYTHON}" "${CODE_ROOT}/benchmarks/worldarena/mlr_flowwam_variants.py" \
    --manifest "${MANIFEST}" --variant "${variant}=${videos}" \
    --output-dir "${out}" --num-shards 8
  python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_campaign_checks.py" mlr \
    "${out}/summary.json" "${variant}"
  printf '%s\n' "utc=$(date -u +%FT%TZ)" "metrics=${out}" \
    >"${STATUS_ROOT}/${variant}.mlr.complete"
}

try_aggregate() {
  mark_stage "aggregate_if_ready"
  if python "${CODE_ROOT}/benchmarks/robotwin_flowwam/flowwam_five_row_aggregate.py" \
      --campaign-root "${CAMPAIGN_ROOT}"; then
    printf '%s\n' "utc=$(date -u +%FT%TZ)" \
      "json=${CAMPAIGN_ROOT}/aggregate/table5.json" \
      "csv=${CAMPAIGN_ROOT}/aggregate/table5.csv" \
      >"${STATUS_ROOT}/aggregate.complete"
    echo "AGGREGATE_COMPLETE"
  else
    echo "AGGREGATE_PENDING other node outputs"
  fi
}

echo "NODE_START role=${ROLE} host=$(hostname) utc=$(date -u +%FT%TZ)"
if [[ -s "${STATUS_ROOT}/node_${ROLE,,}.failed" ]]; then
  mv "${STATUS_ROOT}/node_${ROLE,,}.failed" \
    "${STATUS_ROOT}/node_${ROLE,,}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ "${ROLE}" == "A" ]]; then
  smoke_variant igr
  train_variant igr
  generate_variant igr "${CHECKPOINT_ROOT}/five_row_igr/final.safetensors"
  train_variant sft
  generate_variant sft "${CHECKPOINT_ROOT}/five_row_sft/final.safetensors"
  evaluate_variant igr
  evaluate_variant sft
else
  smoke_variant tia
  smoke_stage1
  train_variant tia
  generate_variant tia "${CHECKPOINT_ROOT}/five_row_tia/final.safetensors"
  generate_variant stage1
  evaluate_variant tia
  evaluate_variant stage1
  evaluate_mlr_only eve "${EXISTING_EVE_VIDEOS}"
fi

printf '%s\n' "utc=$(date -u +%FT%TZ)" >"${STATUS_ROOT}/node_${ROLE,,}.complete"
try_aggregate
mark_stage "complete"
echo "NODE_COMPLETE role=${ROLE} utc=$(date -u +%FT%TZ)"
