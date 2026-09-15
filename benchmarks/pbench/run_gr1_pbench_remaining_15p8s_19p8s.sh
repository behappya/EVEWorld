#!/usr/bin/env bash
set -euo pipefail

# Finish only the remaining GR1/SFT PBench full generations:
#   - 15.8s / 253 frames
#   - 19.8s / 317 frames
#
# After each generation completes, this script optionally runs Qwen Domain/VQA
# with the current 32000-token + thinking configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${MODE:-full}"
SWEEP_ID="${SWEEP_ID:-gr1_pbench_len_sweep_full_8gpu}"
DATA_LIMIT="${DATA_LIMIT:-0}"
EXPECTED_COUNT="${EXPECTED_COUNT:-174}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE:-1}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-640}"
SEED="${SEED:-6666}"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi/giga_world_0_outputs}"
GEN_OUTPUT_ROOT="${GEN_OUTPUT_ROOT:-${EVAL_ROOT}/gr1_pbench_robot_length_sweep}"
DOMAIN_EVAL_ROOT="${DOMAIN_EVAL_ROOT:-${EVAL_ROOT}/gr1_pbench_robot_qwen_vqa_eval}"
DATA_PATH="${DATA_PATH:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"

GR1_CHECKPOINT_DIR="${GR1_CHECKPOINT_DIR:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
GR1_TRANSFORMER_SUBDIR="${GR1_TRANSFORMER_SUBDIR:-transformer_ema}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
MODEL_DIR="${MODEL_DIR:-${GEN_OUTPUT_ROOT}/model_gr1_checkpoint_epoch_100_step_200_${GR1_TRANSFORMER_SUBDIR}}"

RUN_DOMAIN="${RUN_DOMAIN:-1}"
DOMAIN_QWEN_BASE="${DOMAIN_QWEN_BASE:-http://127.0.0.1:8000/v1}"
DOMAIN_QWEN_TAG="${DOMAIN_QWEN_TAG:-58_think32000}"
DOMAIN_QWEN_MODEL="${DOMAIN_QWEN_MODEL:-auto}"
DOMAIN_CONCURRENCY="${DOMAIN_CONCURRENCY:-100}"
DOMAIN_MAX_INFLIGHT="${DOMAIN_MAX_INFLIGHT:-100}"
DOMAIN_FRAME_COUNT="${DOMAIN_FRAME_COUNT:-8}"
DOMAIN_MAX_IMAGE_SIDE="${DOMAIN_MAX_IMAGE_SIDE:-512}"
DOMAIN_MODEL_MAX_TOKENS="${DOMAIN_MODEL_MAX_TOKENS:-32000}"
DOMAIN_MODEL_TIMEOUT="${DOMAIN_MODEL_TIMEOUT:-600}"
DOMAIN_MODEL_RETRIES="${DOMAIN_MODEL_RETRIES:-3}"
DOMAIN_DISABLE_THINKING="${DOMAIN_DISABLE_THINKING:-0}"
DOMAIN_REQUIRE_ZERO_ERRORS="${DOMAIN_REQUIRE_ZERO_ERRORS:-0}"

ARCHIVE_INCOMPLETE="${ARCHIVE_INCOMPLETE:-1}"
ARCHIVE_BAD_EVAL="${ARCHIVE_BAD_EVAL:-1}"
RECENT_MINUTES="${RECENT_MINUTES:-20}"
STATUS_ONLY="${STATUS_ONLY:-0}"

LENGTH_LABELS=("15p8s" "19p8s")
LENGTH_FRAMES=("253" "317")

mkdir -p "${GEN_OUTPUT_ROOT}" "${DOMAIN_EVAL_ROOT}"

json_value() {
  local path="$1"
  local key="$2"
  python3 - "$path" "$key" <<'PY'
import json
import sys
path, key = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get(key, ""))
PY
}

mp4_count() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name 'robot_*.mp4' 2>/dev/null | wc -l
}

generation_done() {
  local save_dir="$1"
  local summary_path="${save_dir}/generation_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  local generated request_count
  generated="$(json_value "${summary_path}" generated_count)"
  request_count="$(json_value "${summary_path}" request_count)"
  [[ "${generated}" == "${EXPECTED_COUNT}" && "${request_count}" == "${EXPECTED_COUNT}" ]]
}

domain_done() {
  local eval_dir="$1"
  local summary_path="${eval_dir}/qwen_vqa_summary.json"
  [[ -f "${summary_path}" ]] || return 1
  python3 - "$summary_path" "$EXPECTED_COUNT" "$DOMAIN_REQUIRE_ZERO_ERRORS" <<'PY'
import json
import sys
path, expected, require_zero = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("sample_count") == expected
if require_zero:
    ok = ok and int(data.get("error_count") or 0) == 0
raise SystemExit(0 if ok else 1)
PY
}

recent_generation_activity() {
  local save_dir="$1"
  find "${save_dir}" -maxdepth 1 -type f \
    \( -name 'run.log' -o -name 'gpu_memory_samples.csv' -o -name 'robot_*.mp4' -o -name 'generation_summary.json' \) \
    -mmin "-${RECENT_MINUTES}" 2>/dev/null | grep -q .
}

prepare_model_dir() {
  local transformer_src="${GR1_CHECKPOINT_DIR}/${GR1_TRANSFORMER_SUBDIR}"
  [[ -d "${transformer_src}" ]] || { echo "Missing GR1 transformer dir: ${transformer_src}" >&2; exit 1; }
  [[ -f "${transformer_src}/config.json" ]] || { echo "Missing transformer config: ${transformer_src}/config.json" >&2; exit 1; }
  if [[ ! -f "${transformer_src}/diffusion_pytorch_model.safetensors" && ! -f "${transformer_src}/diffusion_pytorch_model.bin" ]]; then
    echo "Missing transformer weights: ${transformer_src}/diffusion_pytorch_model.{safetensors,bin}" >&2
    exit 1
  fi
  [[ -d "${PRETRAIN_DIR}/text_encoder" ]] || { echo "Missing text_encoder: ${PRETRAIN_DIR}/text_encoder" >&2; exit 1; }
  [[ -d "${PRETRAIN_DIR}/vae" ]] || { echo "Missing vae: ${PRETRAIN_DIR}/vae" >&2; exit 1; }

  mkdir -p "${MODEL_DIR}"
  ln -sfn "${transformer_src}" "${MODEL_DIR}/transformer"
  ln -sfn "${PRETRAIN_DIR}/text_encoder" "${MODEL_DIR}/text_encoder"
  ln -sfn "${PRETRAIN_DIR}/vae" "${MODEL_DIR}/vae"
  cat > "${MODEL_DIR}/model_manifest.json" <<EOF
{
  "model": "gr1_finetuned_gigaworld0",
  "checkpoint_dir": "${GR1_CHECKPOINT_DIR}",
  "transformer_src": "${transformer_src}",
  "pretrain_dir": "${PRETRAIN_DIR}",
  "text_encoder_src": "${PRETRAIN_DIR}/text_encoder",
  "vae_src": "${PRETRAIN_DIR}/vae"
}
EOF
}

archive_incomplete_generation_if_needed() {
  local save_dir="$1"
  [[ -d "${save_dir}" ]] || return 0
  generation_done "${save_dir}" && return 0
  if recent_generation_activity "${save_dir}"; then
    echo "Found recent generation activity; will wait instead of archiving: ${save_dir}"
    return 0
  fi
  if [[ "${ARCHIVE_INCOMPLETE}" != "1" ]]; then
    echo "Incomplete generation exists and ARCHIVE_INCOMPLETE=0: ${save_dir}" >&2
    exit 1
  fi
  local archived="${save_dir}_incomplete_$(date +%Y%m%d_%H%M%S)"
  echo "Archive incomplete generation dir: ${save_dir} -> ${archived}"
  mv "${save_dir}" "${archived}"
}

archive_bad_eval_if_needed() {
  local eval_dir="$1"
  [[ -d "${eval_dir}" ]] || return 0
  domain_done "${eval_dir}" && return 0
  if [[ "${ARCHIVE_BAD_EVAL}" != "1" ]]; then
    echo "Incomplete/error eval exists and ARCHIVE_BAD_EVAL=0: ${eval_dir}" >&2
    exit 1
  fi
  local archived="${eval_dir}_bad_$(date +%Y%m%d_%H%M%S)"
  echo "Archive incomplete/error eval dir: ${eval_dir} -> ${archived}"
  mv "${eval_dir}" "${archived}"
}

wait_for_generation() {
  local run_name="$1"
  local save_dir="$2"
  local log_path="${save_dir}/run.log"
  echo "Waiting for generation: ${run_name}"
  while true; do
    if [[ -f "${log_path}" ]] && rg -q "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}"; then
      echo "Generation appears to have failed. Check log: ${log_path}" >&2
      rg -n "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}" -S | tail -n 80 >&2
      exit 1
    fi
    if generation_done "${save_dir}"; then
      echo "Generation complete: ${EXPECTED_COUNT}/${EXPECTED_COUNT}"
      break
    fi
    echo "  robot mp4 count so far: $(mp4_count "${save_dir}")/${EXPECTED_COUNT}; sleeping 60s"
    sleep 60
  done
}

launch_generation_if_needed() {
  local label="$1"
  local frames="$2"
  local run_name="gr1_pbench_robot_${label}_full_${SWEEP_ID}"
  local save_dir="${GEN_OUTPUT_ROOT}/${run_name}"

  echo
  echo "============================================================"
  echo "GR1 PBench ${label}: NUM_FRAMES=${frames}, FPS=${FPS}"
  echo "Run: ${run_name}"
  echo "Save dir: ${save_dir}"
  echo "============================================================"

  if generation_done "${save_dir}"; then
    echo "Generation already complete: ${run_name}"
  else
    if [[ "${STATUS_ONLY}" == "1" ]]; then
      if [[ -d "${save_dir}" ]]; then
        if recent_generation_activity "${save_dir}"; then
          echo "Would wait for existing active generation: ${run_name}"
        else
          echo "Would archive stale incomplete generation dir: ${save_dir}"
          echo "Would submit generation: ${run_name}"
        fi
      else
        echo "Would submit generation: ${run_name}"
      fi
      return
    fi
    archive_incomplete_generation_if_needed "${save_dir}"
    if [[ "${STATUS_ONLY}" == "1" ]]; then
      echo "Would submit generation: ${run_name}"
      return
    fi
    if [[ -d "${save_dir}" ]] && recent_generation_activity "${save_dir}"; then
      wait_for_generation "${run_name}" "${save_dir}"
    else
      MODE="${MODE}" \
      RUN_NAME="${run_name}" \
      OUTPUT_ROOT="${GEN_OUTPUT_ROOT}" \
      SAVE_DIR="${save_dir}" \
      MODEL_DIR="${MODEL_DIR}" \
      DATA_PATH="${DATA_PATH}" \
      DATA_LIMIT="${DATA_LIMIT}" \
      NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
      GPU_IDS="${GPU_IDS}" \
      ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE}" \
      NUM_FRAMES="${frames}" \
      FPS="${FPS}" \
      HEIGHT="${HEIGHT}" \
      WIDTH="${WIDTH}" \
      SEED="${SEED}" \
      GPU_MEMORY_SAMPLES="${save_dir}/gpu_memory_samples.csv" \
      GPU_MEMORY_PEAK="${save_dir}/gpu_memory_peak.json" \
      "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_kjob.sh"
      wait_for_generation "${run_name}" "${save_dir}"
    fi
  fi

  run_domain_if_needed "${run_name}" "${save_dir}"
}

run_domain_if_needed() {
  local run_name="$1"
  local save_dir="$2"
  local eval_dir="${DOMAIN_EVAL_ROOT}/${run_name}_domain_qwen36vl_${DOMAIN_QWEN_TAG}"
  [[ "${RUN_DOMAIN}" == "1" ]] || return 0
  if domain_done "${eval_dir}"; then
    echo "Domain already complete: ${eval_dir}"
    return
  fi
  archive_bad_eval_if_needed "${eval_dir}"
  if [[ "${STATUS_ONLY}" == "1" ]]; then
    echo "Would run Domain/VQA: ${eval_dir}"
    return
  fi
  echo "Running PBench Domain/VQA: ${run_name}"
  VIDEO_DIR="${save_dir}" \
  EVAL_DIR="${eval_dir}" \
  METADATA_JSONL="${METADATA_JSONL}" \
  QWEN_BASE="${DOMAIN_QWEN_BASE}" \
  QWEN_MODEL="${DOMAIN_QWEN_MODEL}" \
  LIMIT="${DATA_LIMIT}" \
  CONCURRENCY="${DOMAIN_CONCURRENCY}" \
  MAX_INFLIGHT="${DOMAIN_MAX_INFLIGHT}" \
  FRAME_COUNT="${DOMAIN_FRAME_COUNT}" \
  MAX_IMAGE_SIDE="${DOMAIN_MAX_IMAGE_SIDE}" \
  MODEL_MAX_TOKENS="${DOMAIN_MODEL_MAX_TOKENS}" \
  MODEL_TIMEOUT="${DOMAIN_MODEL_TIMEOUT}" \
  MODEL_RETRIES="${DOMAIN_MODEL_RETRIES}" \
  DISABLE_THINKING="${DOMAIN_DISABLE_THINKING}" \
  RERUN_ERRORS=1 \
  bash "${EVEWORLD_ROOT}/benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh"
  if ! domain_done "${eval_dir}"; then
    echo "Domain/VQA finished but still incomplete: ${eval_dir}" >&2
    exit 1
  fi
}

print_status() {
  python3 - "${GEN_OUTPUT_ROOT}" "${DOMAIN_EVAL_ROOT}" "${SWEEP_ID}" "${DOMAIN_QWEN_TAG}" <<'PY'
import json
import sys
from pathlib import Path
gen_root, eval_root, sweep_id, tag = sys.argv[1:5]
gen_root = Path(gen_root)
eval_root = Path(eval_root)
for label in ("15p8s", "19p8s"):
    run = f"gr1_pbench_robot_{label}_full_{sweep_id}"
    gen_dir = gen_root / run
    eval_dir = eval_root / f"{run}_domain_qwen36vl_{tag}"
    summary = {}
    peak = {}
    qwen = {}
    for path, out in ((gen_dir / "generation_summary.json", summary), (gen_dir / "gpu_memory_peak.json", peak), (eval_dir / "qwen_vqa_summary.json", qwen)):
        if path.exists():
            out.update(json.loads(path.read_text()))
    print(
        f"{label}: generated={summary.get('generated_count')}/{summary.get('request_count')} "
        f"mp4={len(list(gen_dir.glob('robot_*.mp4'))) if gen_dir.exists() else 0} "
        f"wall={summary.get('wall_time_sec')} mean={summary.get('model_elapsed_mean_sec')} "
        f"peak_gib={peak.get('peak_memory_used_gib')} "
        f"domain={qwen.get('domain_score_like')} errors={qwen.get('error_count')}"
    )
PY
}

cd "${REPO_DIR}"
prepare_model_dir

echo "GR1 PBench remaining full sweep"
echo "Repo:                  ${REPO_DIR}"
echo "Sweep id:              ${SWEEP_ID}"
echo "Lengths:               ${LENGTH_LABELS[*]}"
echo "Checkpoint dir:        ${GR1_CHECKPOINT_DIR}"
echo "Prepared model dir:    ${MODEL_DIR}"
echo "Data limit/expected:   ${DATA_LIMIT}/${EXPECTED_COUNT}"
echo "GPU ids:               ${GPU_IDS}"
echo "Run Domain:            ${RUN_DOMAIN}"
echo "Domain Qwen:           ${DOMAIN_QWEN_BASE}, tag=${DOMAIN_QWEN_TAG}"
echo "Domain max tokens:     ${DOMAIN_MODEL_MAX_TOKENS}"
echo "Domain disable think:  ${DOMAIN_DISABLE_THINKING}"
echo "Domain require 0 err:  ${DOMAIN_REQUIRE_ZERO_ERRORS}"
echo "Status only:           ${STATUS_ONLY}"

for idx in "${!LENGTH_LABELS[@]}"; do
  launch_generation_if_needed "${LENGTH_LABELS[$idx]}" "${LENGTH_FRAMES[$idx]}"
done

echo
print_status
echo "Done."
