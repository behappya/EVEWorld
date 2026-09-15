#!/usr/bin/env bash
#SBATCH --job-name=wa1_eval
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

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-WorldArena}"
WA1_ROOT="${WA1_ROOT:-/data/datasets/gagi/worldarena1}"
WORLDARENA_ROOT="${WORLDARENA_ROOT:-/home/jovyan/gagibench/WorldArena_WA1_EVAL}"
MANIFEST="${MANIFEST:-${WA1_ROOT}/manifests/track1_it2v.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/data/datasets/gagi/eve_v2_outputs/worldarena_eval_videos}"
EVAL_ROOT="${EVAL_ROOT:-${WA1_ROOT}/evaluation}"
CONFIG="${CONFIG:-${EVEWORLD_ROOT}/benchmarks/worldarena/eval_config.yaml}"
MODELS="${MODELS:-pretrain round0 t4g_wmapA_pre_seed42_s250}"
METRICS="${METRICS:-image_quality aesthetic_quality dynamic_degree flow_score motion_smoothness subject_consistency background_consistency photometric_smoothness}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1

overall_rc=0
for model in ${MODELS}; do
  video_dir="${VIDEO_ROOT}/${model}_test"
  output_dir="${EVAL_ROOT}/${model}/core"
  mkdir -p "${output_dir}"
  for metric in ${METRICS}; do
    output="${output_dir}/${metric}.json"
    if [[ -s "${output}" ]]; then
      echo "SKIP model=${model} metric=${metric} output=${output}"
      continue
    fi
    echo "START model=${model} metric=${metric} count=${EXPECTED_COUNT} ranks=${NPROC_PER_NODE}"
    torchrun --standalone --nproc-per-node="${NPROC_PER_NODE}" \
      benchmarks/worldarena/local_metric_eval.py \
      --worldarena-root "${WORLDARENA_ROOT}" \
      --config "${CONFIG}" \
      --manifest "${MANIFEST}" \
      --expected-count "${EXPECTED_COUNT}" \
      --video-dir "${video_dir}" \
      --metric "${metric}" \
      --output "${output}" \
      >>"${output_dir}/${metric}.log" 2>&1
    rc=$?
    echo "DONE model=${model} metric=${metric} rc=${rc}"
    [[ "${rc}" != "0" ]] && overall_rc=1
  done
done
exit "${overall_rc}"
