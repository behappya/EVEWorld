#!/usr/bin/env bash
# Serial controller for the leakage-safe held-out main experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

PHASE="${PHASE:?Set PHASE=train, PHASE=validate, or PHASE=generate}"
RUN_TAG="${RUN_TAG:-heldout_main_$(date +%Y%m%d)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/eve/heldout_main}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${EVEWORLD_ROOT}/eveworld/data_curation/splits/frontier_20260715/train.jsonl}"
TRAINING_SEEDS="${TRAINING_SEEDS:-20260716 20260717 20260718}"
METHODS="${METHODS:-joint_lora frontier_only eve}"
POLL_SEC="${POLL_SEC:-120}"
STARTUP_SEC="${STARTUP_SEC:-60}"
OUTPUT_WAIT_SEC="${OUTPUT_WAIT_SEC:-600}"
OUTPUT_POLL_SEC="${OUTPUT_POLL_SEC:-15}"
TRAIN_LAUNCHER="${SCRIPT_DIR}/launch_heldout_train_kjob.sh"
GEN_LAUNCHER="${SCRIPT_DIR}/launch_heldout_generate_kjob.sh"
STATE_DIR="${OUTPUT_ROOT}/${RUN_TAG}/controller"
ACTIVE_JOB_FILE="${STATE_DIR}/active_job.tsv"
mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/controller.lock"
if ! flock -n 9; then
  echo "Another held-out controller is already running for RUN_TAG=${RUN_TAG}" >&2
  exit 2
fi

job_alive() {
  local job="$1"
  timeout 40 kubectl get pods 2>/dev/null | grep -E "${job}" | grep -qiE "Running|Pending|ContainerCreating"
}

submit_and_track() {
  local label="$1"
  shift
  local output job
  output="$("$@" 2>&1)"
  echo "${output}" | tail -8
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  job="$(echo "${output}" | grep -oE "slurm-profile-slurm-[a-z0-9]+" | head -1)"
  if [[ -z "${job}" ]]; then
    echo "[heldout-chain] failed to capture job id for ${label}" >&2
    return 1
  fi
  echo "[heldout-chain] ${label} -> ${job}"
  printf '%s\t%s\n' "${job}" "${label}" > "${ACTIVE_JOB_FILE}"
  sleep "${STARTUP_SEC}"
  while job_alive "${job}"; do
    sleep "${POLL_SEC}"
  done
  rm -f "${ACTIVE_JOB_FILE}"
  echo "[heldout-chain] ${label} left the active queue"
}

recover_active_job() {
  local job label
  [[ -s "${ACTIVE_JOB_FILE}" ]] || return 0
  IFS=$'\t' read -r job label < "${ACTIVE_JOB_FILE}"
  if job_alive "${job}"; then
    echo "[heldout-chain] recovering active job ${job}: ${label}"
    while job_alive "${job}"; do
      sleep "${POLL_SEC}"
    done
  fi
  rm -f "${ACTIVE_JOB_FILE}"
}

recover_active_job

checkpoint_ready() {
  local project="$1" step="$2"
  find "${project}/models" -type d \
    -path "*/checkpoint_*_step_${step}/transformer" \
    -print -quit 2>/dev/null | grep -q .
}

wait_for_checkpoint() {
  local project="$1" step="$2" deadline
  deadline=$((SECONDS + OUTPUT_WAIT_SEC))
  while ! checkpoint_ready "${project}" "${step}"; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    echo "[heldout-chain] cluster job ended; waiting for checkpoint step ${step} on shared storage"
    sleep "${OUTPUT_POLL_SEC}"
  done
}

if [[ "${PHASE}" == "train" ]]; then
  CANDIDATES="${CANDIDATES:-${OUTPUT_ROOT}/${RUN_TAG}/checkpoint_candidates.tsv}"
  mkdir -p "$(dirname "${CANDIDATES}")"
  TRAIN_COUNT="$(python3 eveworld/data_curation/scripts/heldout_manifest_tool.py count \
    --manifest "${TRAIN_MANIFEST}" --expected-split train)"
  STEPS_PER_EPOCH=$(( (TRAIN_COUNT + 7) / 8 ))
  EXPECTED_MAX_STEPS="${MAX_STEPS:-$((STEPS_PER_EPOCH * 50))}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    CANDIDATES_TMP="$(mktemp /tmp/eve_heldout_candidates.XXXXXX)"
  else
    CANDIDATES_TMP="${CANDIDATES}.tmp.$$"
  fi
  trap 'rm -f "${CANDIDATES_TMP}"' EXIT
  printf 'method\tpipeline\ttraining_seed\tcheckpoint_step\tlora_path\n' > "${CANDIDATES_TMP}"
  for training_seed in ${TRAINING_SEEDS}; do
    for method in ${METHODS}; do
      project_base="${OUTPUT_ROOT}/${RUN_TAG}/train/${method}/seed_${training_seed}"
      project=""
      while IFS= read -r candidate_project; do
        if checkpoint_ready "${candidate_project}" "${EXPECTED_MAX_STEPS}"; then
          project="${candidate_project}"
        fi
      done < <(find "$(dirname "${project_base}")" -maxdepth 1 -type d \
        -name "$(basename "${project_base}")*" 2>/dev/null | sort -V)

      if [[ -n "${project}" ]]; then
        echo "[heldout-chain] reuse completed ${method} seed=${training_seed}: ${project}"
      else
        project="${project_base}"
        retry_index=0
        if [[ -d "${project}" ]] && find "${project}" -mindepth 1 -print -quit | grep -q .; then
          retry_index=1
          while [[ -e "${project_base}_retry${retry_index}" ]]; do
            retry_index=$((retry_index + 1))
          done
          project="${project_base}_retry${retry_index}"
          echo "[heldout-chain] incomplete prior run preserved; restarting in ${project}"
        fi
        run_name="${RUN_TAG}_${method}_trainseed${training_seed}"
        if [[ "${retry_index}" -gt 0 ]]; then
          run_name="${run_name}_retry${retry_index}"
        fi
        submit_and_track "train ${method} seed=${training_seed}" \
          bash "${TRAIN_LAUNCHER}" \
            "METHOD=${method}" \
            "TRAINING_SEED=${training_seed}" \
            "TRAIN_MANIFEST=${TRAIN_MANIFEST}" \
            "RUN_TAG=${RUN_TAG}" \
            "RUN_NAME=${run_name}" \
            "TRAIN_PROJECT_DIR=${project}" \
            "OUTPUT_ROOT=${OUTPUT_ROOT}" \
            "MAX_STEPS=${EXPECTED_MAX_STEPS}" \
            "DRY_RUN=${DRY_RUN:-0}"
      fi

      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
      fi
      if ! wait_for_checkpoint "${project}" "${EXPECTED_MAX_STEPS}"; then
        echo "Training did not reach step ${EXPECTED_MAX_STEPS}: ${project}" >&2
        exit 1
      fi
      mapfile -t checkpoints < <(find "${project}/models" -mindepth 2 -maxdepth 2 \
        -type d -name transformer | sort -V)
      if [[ ${#checkpoints[@]} -eq 0 ]]; then
        echo "No checkpoint produced for ${method} seed=${training_seed}: ${project}" >&2
        exit 1
      fi
      pipeline="joint"
      [[ "${method}" == "frontier_only" || "${method}" == "eve" ]] && pipeline="frontier"
      for checkpoint in "${checkpoints[@]}"; do
        parent="$(basename "$(dirname "${checkpoint}")")"
        step="${parent##*_step_}"
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "${method}" "${pipeline}" "${training_seed}" "${step}" "${checkpoint}" >> "${CANDIDATES_TMP}"
      done
    done
  done
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[heldout-chain] dry-run complete; candidate table was not modified"
    exit 0
  fi
  mv "${CANDIDATES_TMP}" "${CANDIDATES}"
  trap - EXIT
  echo "[heldout-chain] training complete; validation candidates: ${CANDIDATES}"
  echo "Select exactly one checkpoint per method/training seed on validation data, then create selected_checkpoints.tsv."
  exit 0
fi

if [[ "${PHASE}" == "validate" ]]; then
  CANDIDATES="${CANDIDATES:-${OUTPUT_ROOT}/${RUN_TAG}/checkpoint_candidates.tsv}"
  VAL_DATA_PATH="${VAL_DATA_PATH:?VAL_DATA_PATH is required}"
  VAL_MANIFEST="${VAL_MANIFEST:?VAL_MANIFEST is required}"
  EXPECTED_VAL_SPLIT="${EXPECTED_VAL_SPLIT:-val}"
  VAL_ROOT="${VAL_ROOT:-${OUTPUT_ROOT}/${RUN_TAG}/validation}"
  QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
  EVAL_PYTHON="${EVAL_PYTHON:-/data/datasets/gagi/envs/giga_world_train_venv/bin/python}"
  VAL_SCORE_ROOT="${VAL_SCORE_ROOT:-${VAL_ROOT}/scores}"
  SELECTED_OUT="${SELECTED_OUT:-${OUTPUT_ROOT}/${RUN_TAG}/selected_checkpoints.tsv}"
  VAL_COUNT="$(python3 eveworld/data_curation/scripts/heldout_manifest_tool.py count \
    --manifest "${VAL_MANIFEST}" --expected-split "${EXPECTED_VAL_SPLIT}")"
  mkdir -p "${VAL_SCORE_ROOT}/qwen" "${VAL_SCORE_ROOT}/tea"
  while IFS=$'\t' read -r method pipeline training_seed checkpoint_step lora_path; do
    [[ "${method}" == "method" || -z "${method}" ]] && continue
    candidate_method="${method}_step${checkpoint_step}"
    submit_and_track "validate ${method} train_seed=${training_seed} step=${checkpoint_step}" \
      bash "${GEN_LAUNCHER}" \
        "PIPELINE=${pipeline}" \
        "METHOD=${candidate_method}" \
        "TRAINING_SEED=${training_seed}" \
        "LORA=${lora_path}" \
        "DATA_PATH=${VAL_DATA_PATH}" \
        "SPLIT_MANIFEST=${VAL_MANIFEST}" \
        "EXPECTED_SPLIT=${EXPECTED_VAL_SPLIT}" \
        "OUT_ROOT=${VAL_ROOT}" \
        "GENERATION_SEEDS=${GENERATION_SEEDS:-6666 1234}" \
        "DRY_RUN=${DRY_RUN:-0}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      continue
    fi
    for generation_seed in ${GENERATION_SEEDS:-6666 1234}; do
      run_name="${method}_train${training_seed}_step${checkpoint_step}_gen${generation_seed}"
      video_dir="${VAL_ROOT}/${candidate_method}/train_seed_${training_seed}/gen_seed_${generation_seed}/generated_only"
      "${EVAL_PYTHON}" eveworld/evaluation/tea/qwen_laziness.py \
        --video-dir "${video_dir}" --out-root "${VAL_SCORE_ROOT}/qwen" \
        --run-name "${run_name}_B" --qwen-base "${QWEN_BASE}" \
        --judge b --frame-offset 0.5 --concurrency "${QWEN_CONCURRENCY:-64}"
      "${EVAL_PYTHON}" eveworld/evaluation/tea/ncm.py score \
        --video-dir "${video_dir}" --out "${VAL_SCORE_ROOT}/tea/${run_name}.json" \
        --tag "${run_name}"
    done
  done < "${CANDIDATES}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    read -r -a VALIDATION_SEEDS <<< "${GENERATION_SEEDS:-6666 1234}"
    "${EVAL_PYTHON}" eveworld/method/scripts/select_validation_checkpoints.py \
      --candidates "${CANDIDATES}" \
      --score-root "${VAL_SCORE_ROOT}/qwen" \
      --generation-seeds "${VALIDATION_SEEDS[@]}" \
      --expected-per-seed "${VAL_COUNT}" \
      --out "${SELECTED_OUT}"
  fi
  echo "[heldout-chain] frozen checkpoint selection: ${SELECTED_OUT}"
  exit 0
fi

if [[ "${PHASE}" != "generate" ]]; then
  echo "PHASE must be train, validate, or generate" >&2
  exit 2
fi

SELECTED_CHECKPOINTS="${SELECTED_CHECKPOINTS:?Set SELECTED_CHECKPOINTS to the frozen TSV}"
TEST_DATA_PATH="${TEST_DATA_PATH:?TEST_DATA_PATH is required}"
TEST_MANIFEST="${TEST_MANIFEST:?TEST_MANIFEST is required}"
EXPECTED_TEST_SPLIT="${EXPECTED_TEST_SPLIT:-test}"
GEN_ROOT="${GEN_ROOT:-${OUTPUT_ROOT}/${RUN_TAG}/generation}"

read -r -a REQUIRED_METHOD_ARRAY <<< "${METHODS}"
read -r -a REQUIRED_SEED_ARRAY <<< "${TRAINING_SEEDS}"
python3 eveworld/method/scripts/validate_selected_checkpoints.py \
  --selected "${SELECTED_CHECKPOINTS}" \
  --methods "${REQUIRED_METHOD_ARRAY[@]}" \
  --training-seeds "${REQUIRED_SEED_ARRAY[@]}"

submit_and_track "generate joint_base" \
  bash "${GEN_LAUNCHER}" \
    "PIPELINE=joint" \
    "METHOD=joint_base" \
    "TRAINING_SEED=base" \
    "LORA=NONE" \
    "DATA_PATH=${TEST_DATA_PATH}" \
    "SPLIT_MANIFEST=${TEST_MANIFEST}" \
    "EXPECTED_SPLIT=${EXPECTED_TEST_SPLIT}" \
    "OUT_ROOT=${GEN_ROOT}" \
    "GENERATION_SEEDS=${GENERATION_SEEDS:-6666 1234}" \
    "DRY_RUN=${DRY_RUN:-0}"

if [[ "${DRY_RUN:-0}" != "1" && ! -f "${GEN_ROOT}/joint_base/train_seed_base/dispatch_summary.json" ]]; then
  echo "joint_base generation did not produce a dispatch summary" >&2
  exit 1
fi

while IFS=$'\t' read -r method pipeline training_seed checkpoint_step lora_path; do
  [[ "${method}" == "method" || -z "${method}" ]] && continue
  if [[ ! -d "${lora_path}" ]]; then
    echo "Missing selected checkpoint: ${lora_path}" >&2
    exit 1
  fi
  submit_and_track "generate ${method} train_seed=${training_seed} step=${checkpoint_step}" \
    bash "${GEN_LAUNCHER}" \
      "PIPELINE=${pipeline}" \
      "METHOD=${method}" \
      "TRAINING_SEED=${training_seed}" \
      "LORA=${lora_path}" \
      "DATA_PATH=${TEST_DATA_PATH}" \
      "SPLIT_MANIFEST=${TEST_MANIFEST}" \
      "EXPECTED_SPLIT=${EXPECTED_TEST_SPLIT}" \
      "OUT_ROOT=${GEN_ROOT}" \
      "GENERATION_SEEDS=${GENERATION_SEEDS:-6666 1234}" \
      "DRY_RUN=${DRY_RUN:-0}"
  if [[ "${DRY_RUN:-0}" != "1" && ! -f "${GEN_ROOT}/${method}/train_seed_${training_seed}/dispatch_summary.json" ]]; then
    echo "generation did not produce a dispatch summary for ${method}/${training_seed}" >&2
    exit 1
  fi
done < "${SELECTED_CHECKPOINTS}"

echo "[heldout-chain] all held-out generation jobs completed: ${GEN_ROOT}"
echo "Next: SELECTED_CHECKPOINTS=${SELECTED_CHECKPOINTS} GEN_ROOT=${GEN_ROOT} bash ${SCRIPT_DIR}/score_heldout_main.sh"
