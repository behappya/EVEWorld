#!/usr/bin/env bash
#SBATCH --job-name=xmodel_wa1
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  case "$(hostname)" in
    coder-workspace-*)
      echo "Refusing to run on the workspace host without GPUs." >&2
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
GEN_CONDA_ENV="${GEN_CONDA_ENV:-giga_world1}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-WorldArena}"
MLR_CONDA_ENV="${MLR_CONDA_ENV:-EVEWorld}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
GAGI="${GAGI:-/data/datasets/gagi}"
WA1_ROOT="${WA1_ROOT:-${GAGI}/worldarena1}"

MODELS="${MODELS:-wan22_ti2v_5b wan22_i2v_a14b}"
CHAIN_NAME="${CHAIN_NAME:-xmodel_chain}"
DATA_PATH="${DATA_PATH:-${WA1_ROOT}/manifests/track1_it2v.json}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WA1_ROOT}/xmodel_generation}"
VIDEO_ROOT="${VIDEO_ROOT:-${GAGI}/eve_v2_outputs/worldarena_eval_videos}"
EVAL_ROOT="${EVAL_ROOT:-${WA1_ROOT}/evaluation}"
MLR_ROOT="${MLR_ROOT:-${WA1_ROOT}/mlr_xmodels}"

NUM_FRAMES="${NUM_FRAMES:-125}"
FINAL_FRAMES="${FINAL_FRAMES:-121}"
FPS="${FPS:-24}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
NORMALIZE_WORKERS="${NORMALIZE_WORKERS:-8}"
RUN_CORE="${RUN_CORE:-1}"
RUN_MLR="${RUN_MLR:-1}"

export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

family_for() {
  case "$1" in
    wan22_ti2v_5b) echo "wan_ti2v" ;;
    wan22_i2v_a14b) echo "wan" ;;
    cogvideox15_5b_i2v) echo "cogvideox" ;;
    cosmos_predict2_2b_v2w) echo "cosmos" ;;
    *) echo "" ;;
  esac
}

mkdir -p "${OUTPUT_ROOT}" "${VIDEO_ROOT}" "${EVAL_ROOT}" "${MLR_ROOT}"
JOB_LOG="${OUTPUT_ROOT}/worldarena1_${CHAIN_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"
echo "host=$(hostname) chain=${CHAIN_NAME} models=${MODELS} expected=${EXPECTED_COUNT}" | tee -a "${JOB_LOG}"
nvidia-smi -L 2>&1 | tee -a "${JOB_LOG}"

overall_rc=0
for model in ${MODELS}; do
  family="$(family_for "${model}")"
  model_path="${GAGI}/xmodels/${model}"
  raw_dir="${OUTPUT_ROOT}/${model}/raw"
  final_dir="${VIDEO_ROOT}/${model}_test"
  summary_path="${OUTPUT_ROOT}/${model}/generation_summary.json"
  run_log="${OUTPUT_ROOT}/${model}/run.log"
  normalize_report="${OUTPUT_ROOT}/${model}/normalization.json"
  mkdir -p "${raw_dir}" "${final_dir}" "${OUTPUT_ROOT}/${model}"

  if [[ -z "${family}" || ! -d "${model_path}" ]]; then
    echo "invalid model=${model} family=${family} path=${model_path}" | tee -a "${JOB_LOG}" >&2
    overall_rc=1
    continue
  fi

  source "${CONDA_SH}"
  conda activate "${GEN_CONDA_ENV}"
  generation_complete=0
  if [[ -s "${summary_path}" ]]; then
    generation_complete=$(python -c \
      'import json,sys; x=json.load(open(sys.argv[1])); print(int(x.get("total")==int(sys.argv[2]) and x.get("ok")==int(sys.argv[2])))' \
      "${summary_path}" "${EXPECTED_COUNT}")
  fi
  if [[ "${generation_complete}" == "1" ]]; then
    echo "===== $(date '+%F %T') SKIP generation ${model}: complete summary" | tee -a "${JOB_LOG}"
    generation_rc=0
  else
    echo "===== $(date '+%F %T') START generation ${model}" | tee -a "${JOB_LOG}"
    cd "${INFER_DIR}"
    python xmodel_dreamgen_infer.py \
      --model-family "${family}" \
      --model-path "${model_path}" \
      --data-path "${DATA_PATH}" \
      --save-dir "${raw_dir}" \
      --gpu-ids "${GPU_IDS}" \
      --num-frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS}" \
      --guidance-scale "${GUIDANCE_SCALE}" \
      --seed "${SEED}" \
      --dtype "${DTYPE}" \
      --summary-path "${summary_path}" >>"${run_log}" 2>&1
    generation_rc=$?
  fi
  echo "===== $(date '+%F %T') DONE generation ${model} rc=${generation_rc}" | tee -a "${JOB_LOG}"
  if [[ "${generation_rc}" != "0" ]]; then
    overall_rc=1
    continue
  fi

  cd "${REPO_DIR}"
  python benchmarks/worldarena/normalize_xmodel_videos.py \
    --manifest "${DATA_PATH}" \
    --input-dir "${raw_dir}" \
    --output-dir "${final_dir}" \
    --expected-count "${EXPECTED_COUNT}" \
    --width "${WIDTH}" --height "${HEIGHT}" \
    --frames "${FINAL_FRAMES}" --fps "${FPS}" \
    --workers "${NORMALIZE_WORKERS}" \
    --report "${normalize_report}" >>"${run_log}" 2>&1
  normalize_rc=$?
  audit_rc=0
  echo "===== $(date '+%F %T') FORMAT ${model} normalize_rc=${normalize_rc}" | tee -a "${JOB_LOG}"
  if [[ "${normalize_rc}" != "0" ]]; then
    overall_rc=1
    continue
  fi

  if [[ "${RUN_CORE}" == "1" ]]; then
    echo "===== $(date '+%F %T') START core metrics ${model}" | tee -a "${JOB_LOG}"
    bash "${EVEWORLD_ROOT}/benchmarks/worldarena/kjob_worldarena1_eval_serial.sh" \
      "MODELS=${model}" \
      "VIDEO_ROOT=${VIDEO_ROOT}" \
      "EVAL_ROOT=${EVAL_ROOT}" \
      "MANIFEST=${DATA_PATH}" \
      "EXPECTED_COUNT=${EXPECTED_COUNT}" \
      "NPROC_PER_NODE=8" >>"${run_log}" 2>&1
    core_rc=$?
    echo "===== $(date '+%F %T') DONE core metrics ${model} rc=${core_rc}" | tee -a "${JOB_LOG}"
    [[ "${core_rc}" != "0" ]] && overall_rc=1
  fi

  if [[ "${RUN_MLR}" == "1" ]]; then
    echo "===== $(date '+%F %T') START MLR ${model}" | tee -a "${JOB_LOG}"
    source "${CONDA_SH}"
    conda activate "${MLR_CONDA_ENV}"
    condition_dir="${MLR_ROOT}/${model}/condition_inventory"
    video_first_path="${MLR_ROOT}/${model}/video_first_summary.json"
    export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
    "${TRAIN_PYTHON}" benchmarks/worldarena/mlr_dispatch.py \
      --manifest "${DATA_PATH}" \
      --video-root "${VIDEO_ROOT}" \
      --models "${model}" \
      --output-dir "${condition_dir}" \
      --python "${TRAIN_PYTHON}" \
      --num-shards 8 >>"${run_log}" 2>&1
    mlr_rc=$?
    if [[ "${mlr_rc}" == "0" ]]; then
      "${TRAIN_PYTHON}" benchmarks/worldarena/recompute_video_first_mlr.py \
        --input "${condition_dir}/summary.json" \
        --output "${video_first_path}" \
        --models "${model}" >>"${run_log}" 2>&1
      mlr_rc=$?
    fi
    echo "===== $(date '+%F %T') DONE MLR ${model} rc=${mlr_rc}" | tee -a "${JOB_LOG}"
    [[ "${mlr_rc}" != "0" ]] && overall_rc=1
  fi
done

if [[ "${RUN_CORE}" == "1" ]]; then
  source "${CONDA_SH}"
  conda activate "${EVAL_CONDA_ENV}"
  python "${EVEWORLD_ROOT}/benchmarks/worldarena/aggregate_core_scores.py" \
    --eval-root "${EVAL_ROOT}" \
    --models ${MODELS} \
    --output-dir "${WA1_ROOT}/comparison_${CHAIN_NAME}" \
    --manifest "${DATA_PATH}" >>"${JOB_LOG}" 2>&1
  aggregate_rc=$?
  [[ "${aggregate_rc}" != "0" ]] && overall_rc=1
fi

echo "===== $(date '+%F %T') ALL DONE chain=${CHAIN_NAME} rc=${overall_rc}" | tee -a "${JOB_LOG}"
exit "${overall_rc}"
