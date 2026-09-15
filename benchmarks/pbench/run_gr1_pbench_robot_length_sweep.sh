#!/usr/bin/env bash
set -euo pipefail

# Run PBench Robotics length sweep with the GR1 fine-tuned transformer and the
# original GigaWorld-0 text_encoder/VAE. The generation output is separate from
# the pretrain/base PBench sweep so scores can be compared side by side.

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
QUALITY_OUTPUT_ROOT_BASE="${QUALITY_OUTPUT_ROOT_BASE:-${EVAL_ROOT}/gr1_pbench_robot_vbench_quality_length_sweep}"
DATA_PATH="${DATA_PATH:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"

GR1_CHECKPOINT_DIR="${GR1_CHECKPOINT_DIR:-/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
GR1_TRANSFORMER_SUBDIR="${GR1_TRANSFORMER_SUBDIR:-transformer_ema}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/data/datasets/gagi/giga_world_0_video_pretrain}"
MODEL_DIR="${MODEL_DIR:-${GEN_OUTPUT_ROOT}/model_gr1_checkpoint_epoch_100_step_200_${GR1_TRANSFORMER_SUBDIR}}"

RUN_DOMAIN="${RUN_DOMAIN:-1}"
DOMAIN_QWEN_BASE="${DOMAIN_QWEN_BASE:-http://127.0.0.1:8000/v1}"
DOMAIN_QWEN_TAG="${DOMAIN_QWEN_TAG:-58}"
DOMAIN_QWEN_MODEL="${DOMAIN_QWEN_MODEL:-auto}"
DOMAIN_CONCURRENCY="${DOMAIN_CONCURRENCY:-100}"
DOMAIN_MAX_INFLIGHT="${DOMAIN_MAX_INFLIGHT:-100}"
DOMAIN_FRAME_COUNT="${DOMAIN_FRAME_COUNT:-8}"
DOMAIN_MAX_IMAGE_SIDE="${DOMAIN_MAX_IMAGE_SIDE:-512}"
DOMAIN_MODEL_MAX_TOKENS="${DOMAIN_MODEL_MAX_TOKENS:-256}"
DOMAIN_MODEL_TIMEOUT="${DOMAIN_MODEL_TIMEOUT:-300}"
DOMAIN_MODEL_RETRIES="${DOMAIN_MODEL_RETRIES:-3}"
DOMAIN_DISABLE_THINKING="${DOMAIN_DISABLE_THINKING:-1}"

RUN_QUALITY="${RUN_QUALITY:-0}"
QUALITY_GPU_IDS="${QUALITY_GPU_IDS:-0}"
QUALITY_LIMIT="${QUALITY_LIMIT:-${DATA_LIMIT}}"
QUALITY_DIMENSIONS="${QUALITY_DIMENSIONS:-i2v_subject i2v_background aesthetic_quality imaging_quality background_consistency motion_smoothness subject_consistency overall_consistency}"

PRETRAIN_GEN_ROOT="${PRETRAIN_GEN_ROOT:-${EVAL_ROOT}/pbench_robot_length_sweep}"
PRETRAIN_SWEEP_ID="${PRETRAIN_SWEEP_ID:-pbench_len_sweep_full_8gpu}"
PRETRAIN_DOMAIN_ROOT="${PRETRAIN_DOMAIN_ROOT:-${EVAL_ROOT}/pbench_robot_qwen_vqa_eval}"
PRETRAIN_DOMAIN_TAG="${PRETRAIN_DOMAIN_TAG:-${DOMAIN_QWEN_TAG}}"

SUMMARY_CSV="${SUMMARY_CSV:-${GEN_OUTPUT_ROOT}/${SWEEP_ID}_summary.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${GEN_OUTPUT_ROOT}/${SWEEP_ID}_summary.json}"
COMPARE_CSV="${COMPARE_CSV:-${GEN_OUTPUT_ROOT}/${SWEEP_ID}_vs_${PRETRAIN_SWEEP_ID}_qwen${DOMAIN_QWEN_TAG}_summary.csv}"
COMPARE_JSON="${COMPARE_JSON:-${GEN_OUTPUT_ROOT}/${SWEEP_ID}_vs_${PRETRAIN_SWEEP_ID}_qwen${DOMAIN_QWEN_TAG}_summary.json}"

LENGTH_LABELS=("3p8s" "5p8s" "9p8s" "15p8s" "19p8s")
LENGTH_FRAMES=("61" "93" "157" "253" "317")

mkdir -p "${GEN_OUTPUT_ROOT}"

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
  python3 - "$summary_path" "$EXPECTED_COUNT" <<'PY'
import json
import sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ok = data.get("sample_count") == expected and int(data.get("error_count") or 0) == 0
raise SystemExit(0 if ok else 1)
PY
}

quality_done() {
  local output_root="$1"
  [[ -f "${output_root}/quality_eval/pbench_robot_quality_overall_summary.json" ]]
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

run_domain_if_needed() {
  local run_name="$1"
  local save_dir="$2"
  local eval_dir="${DOMAIN_EVAL_ROOT}/${run_name}_domain_qwen36vl_${DOMAIN_QWEN_TAG}"
  [[ "${RUN_DOMAIN}" == "1" ]] || return 0
  if domain_done "${eval_dir}"; then
    echo "Domain already complete: ${eval_dir}"
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
}

wait_for_quality() {
  local run_name="$1"
  local output_root="$2"
  local run_log="${output_root}/kjob_logs/${run_name}_quality.log"
  echo "Waiting for Quality/VBench: ${run_name}"
  while true; do
    if [[ -f "${run_log}" ]] && rg -q "Traceback|RuntimeError|CUDA out|Killed|Error" "${run_log}"; then
      echo "Quality appears to have failed. Check log: ${run_log}" >&2
      rg -n "Traceback|RuntimeError|CUDA out|Killed|Error" "${run_log}" -S | tail -n 80 >&2
      exit 1
    fi
    if quality_done "${output_root}"; then
      echo "Quality complete: ${output_root}/quality_eval/pbench_robot_quality_overall_summary.json"
      break
    fi
    echo "  quality summary not ready; sleeping 60s"
    sleep 60
  done
}

run_quality_if_needed() {
  local run_name="$1"
  local save_dir="$2"
  local domain_summary="${DOMAIN_EVAL_ROOT}/${run_name}_domain_qwen36vl_${DOMAIN_QWEN_TAG}/qwen_vqa_summary.json"
  local output_root="${QUALITY_OUTPUT_ROOT_BASE}/${run_name}"
  local quality_run_name="${run_name}_quality"
  [[ "${RUN_QUALITY}" == "1" ]] || return 0
  if [[ ! -f "${domain_summary}" ]]; then
    echo "Skip Quality: missing domain summary ${domain_summary}"
    return
  fi
  if quality_done "${output_root}"; then
    echo "Quality already complete: ${output_root}"
    return
  fi
  echo "Submitting PBench Quality/VBench: ${run_name}"
  OUTPUT_ROOT="${output_root}" \
  RUN_NAME="${quality_run_name}" \
  SOURCE_VIDEO_DIR="${save_dir}" \
  DOMAIN_SUMMARY="${domain_summary}" \
  LIMIT="${QUALITY_LIMIT}" \
  DIMENSIONS="${QUALITY_DIMENSIONS}" \
  GPU_IDS="${QUALITY_GPU_IDS}" \
  RUN_LOG="${output_root}/kjob_logs/${quality_run_name}.log" \
  GPU_MEMORY_SAMPLES="${output_root}/kjob_logs/${quality_run_name}_gpu_memory_samples.csv" \
  GPU_MEMORY_PEAK="${output_root}/kjob_logs/${quality_run_name}_gpu_memory_peak.json" \
  "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_vbench_quality_kjob.sh"
  wait_for_quality "${run_name}" "${output_root}"
}

write_summary() {
  python3 - \
    "${GEN_OUTPUT_ROOT}" \
    "${DOMAIN_EVAL_ROOT}" \
    "${QUALITY_OUTPUT_ROOT_BASE}" \
    "${SWEEP_ID}" \
    "${DOMAIN_QWEN_TAG}" \
    "${PRETRAIN_GEN_ROOT}" \
    "${PRETRAIN_DOMAIN_ROOT}" \
    "${PRETRAIN_SWEEP_ID}" \
    "${PRETRAIN_DOMAIN_TAG}" \
    "${SUMMARY_CSV}" \
    "${SUMMARY_JSON}" \
    "${COMPARE_CSV}" \
    "${COMPARE_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    gr1_gen_root,
    gr1_domain_root,
    gr1_quality_root,
    gr1_sweep_id,
    gr1_domain_tag,
    pre_gen_root,
    pre_domain_root,
    pre_sweep_id,
    pre_domain_tag,
    summary_csv,
    summary_json,
    compare_csv,
    compare_json,
) = sys.argv[1:14]
gr1_gen_root = Path(gr1_gen_root)
gr1_domain_root = Path(gr1_domain_root)
gr1_quality_root = Path(gr1_quality_root)
pre_gen_root = Path(pre_gen_root)
pre_domain_root = Path(pre_domain_root)
summary_csv = Path(summary_csv)
summary_json = Path(summary_json)
compare_csv = Path(compare_csv)
compare_json = Path(compare_json)

labels = [("3.8s", "3p8s", 61), ("5.8s", "5p8s", 93), ("9.8s", "9p8s", 157), ("15.8s", "15p8s", 253), ("19.8s", "19p8s", 317)]

def load(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def delta(new, old):
    if new is None or old is None:
        return None
    try:
        return round(float(new) - float(old), 6)
    except Exception:
        return None

def fmt(value):
    return "-" if value is None else f"{float(value):.6f}"

summary_rows = []
compare_rows = []
for target, label, frames in labels:
    gr1_run = f"gr1_pbench_robot_{label}_full_{gr1_sweep_id}"
    pre_run = f"pbench_robot_{label}_full_{pre_sweep_id}"
    gr1_gen_dir = gr1_gen_root / gr1_run
    gr1_domain_dir = gr1_domain_root / f"{gr1_run}_domain_qwen36vl_{gr1_domain_tag}"
    gr1_quality_dir = gr1_quality_root / gr1_run
    pre_gen_dir = pre_gen_root / pre_run
    pre_domain_dir = pre_domain_root / f"{pre_run}_domain_qwen36vl_{pre_domain_tag}"
    gr1_gen = load(gr1_gen_dir / "generation_summary.json")
    gr1_peak = load(gr1_gen_dir / "gpu_memory_peak.json")
    gr1_domain = load(gr1_domain_dir / "qwen_vqa_summary.json")
    gr1_config = load(gr1_domain_dir / "run_config.json")
    gr1_quality = load(gr1_quality_dir / "quality_eval" / "pbench_robot_quality_overall_summary.json")
    gr1_quality_peak = load(gr1_quality_dir / "kjob_logs" / f"{gr1_run}_quality_gpu_memory_peak.json")
    pre_gen = load(pre_gen_dir / "generation_summary.json")
    pre_peak = load(pre_gen_dir / "gpu_memory_peak.json")
    pre_domain = load(pre_domain_dir / "qwen_vqa_summary.json")
    summary_rows.append({
        "target": target,
        "num_frames": frames,
        "fps": 16,
        "actual_sec": frames / 16.0,
        "run_name": gr1_run,
        "model": "gr1_finetuned",
        "generated_count": gr1_gen.get("generated_count"),
        "request_count": gr1_gen.get("request_count"),
        "generation_wall_time_sec": gr1_gen.get("wall_time_sec"),
        "generation_mean_sec": gr1_gen.get("model_elapsed_mean_sec"),
        "generation_median_sec": gr1_gen.get("model_elapsed_median_sec"),
        "generation_gpu_peak_mib": gr1_peak.get("peak_memory_used_mib"),
        "generation_gpu_peak_gib": gr1_peak.get("peak_memory_used_gib"),
        "generation_gpu_peak_name": gr1_peak.get("peak_gpu_name"),
        "domain_eval_dir": str(gr1_domain_dir),
        "domain_model_max_tokens": gr1_config.get("model_max_tokens"),
        "domain_disable_thinking": gr1_config.get("disable_thinking"),
        "domain_thinking_enabled": gr1_config.get("thinking_enabled"),
        "domain_sample_count": gr1_domain.get("sample_count"),
        "domain_error_count": gr1_domain.get("error_count"),
        "domain_score_like": gr1_domain.get("domain_score_like"),
        "domain_question_micro_accuracy": gr1_domain.get("question_micro_accuracy"),
        "domain_sample_macro_accuracy": gr1_domain.get("sample_macro_accuracy"),
        "quality_score": gr1_quality.get("quality_score"),
        "overall_score_like": gr1_quality.get("overall_score_like"),
        "quality_gpu_peak_mib": gr1_quality_peak.get("peak_memory_used_mib"),
        "quality_gpu_peak_gib": gr1_quality_peak.get("peak_memory_used_gib"),
    })
    compare_rows.append({
        "target": target,
        "num_frames": frames,
        "actual_sec": frames / 16.0,
        "gr1_run_name": gr1_run,
        "pretrain_run_name": pre_run,
        "gr1_generated_count": gr1_gen.get("generated_count"),
        "pretrain_generated_count": pre_gen.get("generated_count"),
        "gr1_generation_wall_time_sec": gr1_gen.get("wall_time_sec"),
        "pretrain_generation_wall_time_sec": pre_gen.get("wall_time_sec"),
        "generation_wall_time_delta_sec": delta(gr1_gen.get("wall_time_sec"), pre_gen.get("wall_time_sec")),
        "gr1_generation_gpu_peak_gib": gr1_peak.get("peak_memory_used_gib"),
        "pretrain_generation_gpu_peak_gib": pre_peak.get("peak_memory_used_gib"),
        "generation_gpu_peak_delta_gib": delta(gr1_peak.get("peak_memory_used_gib"), pre_peak.get("peak_memory_used_gib")),
        "gr1_domain_sample_count": gr1_domain.get("sample_count"),
        "pretrain_domain_sample_count": pre_domain.get("sample_count"),
        "gr1_domain_error_count": gr1_domain.get("error_count"),
        "pretrain_domain_error_count": pre_domain.get("error_count"),
        "gr1_domain_score_like": gr1_domain.get("domain_score_like"),
        "pretrain_domain_score_like": pre_domain.get("domain_score_like"),
        "domain_score_like_delta": delta(gr1_domain.get("domain_score_like"), pre_domain.get("domain_score_like")),
        "gr1_question_micro_accuracy": gr1_domain.get("question_micro_accuracy"),
        "pretrain_question_micro_accuracy": pre_domain.get("question_micro_accuracy"),
        "question_micro_accuracy_delta": delta(gr1_domain.get("question_micro_accuracy"), pre_domain.get("question_micro_accuracy")),
    })

for path, rows in ((summary_csv, summary_rows), (compare_csv, compare_rows)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

summary_json.write_text(json.dumps({"rows": summary_rows, "summary_csv": str(summary_csv)}, ensure_ascii=False, indent=2), encoding="utf-8")
compare_json.write_text(json.dumps({"rows": compare_rows, "compare_csv": str(compare_csv)}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"GR1 summary CSV:  {summary_csv}")
print(f"GR1 summary JSON: {summary_json}")
print(f"Compare CSV:      {compare_csv}")
print(f"Compare JSON:     {compare_json}")
print("| Target | GR1 Domain | Pretrain Domain | Delta | GR1 gen peak GiB | Pretrain gen peak GiB |")
print("| ---: | ---: | ---: | ---: | ---: | ---: |")
for row in compare_rows:
    print(
        f"| {row['target']} | "
        f"{fmt(row['gr1_domain_score_like'])} | "
        f"{fmt(row['pretrain_domain_score_like'])} | "
        f"{fmt(row['domain_score_like_delta'])} | "
        f"{fmt(row['gr1_generation_gpu_peak_gib'])} | "
        f"{fmt(row['pretrain_generation_gpu_peak_gib'])} |"
    )
PY
}

cd "${REPO_DIR}"
prepare_model_dir

echo "GR1 PBench Robot length sweep"
echo "Repo:                  ${REPO_DIR}"
echo "Sweep id:              ${SWEEP_ID}"
echo "Checkpoint dir:        ${GR1_CHECKPOINT_DIR}"
echo "Transformer subdir:    ${GR1_TRANSFORMER_SUBDIR}"
echo "Prepared model dir:    ${MODEL_DIR}"
echo "Pretrain aux dir:      ${PRETRAIN_DIR}"
echo "Data path:             ${DATA_PATH}"
echo "Output root:           ${GEN_OUTPUT_ROOT}"
echo "Domain root:           ${DOMAIN_EVAL_ROOT}"
echo "Data limit/expected:   ${DATA_LIMIT}/${EXPECTED_COUNT}"
echo "Steps:                 ${NUM_INFERENCE_STEPS}"
echo "GPU ids:               ${GPU_IDS}"
echo "Domain Qwen:           ${DOMAIN_QWEN_BASE}, tag=${DOMAIN_QWEN_TAG}"
echo "Domain max tokens:     ${DOMAIN_MODEL_MAX_TOKENS}"
echo "Domain disable think:  ${DOMAIN_DISABLE_THINKING}"
echo "Run Quality:           ${RUN_QUALITY}"
echo

for idx in "${!LENGTH_LABELS[@]}"; do
  label="${LENGTH_LABELS[$idx]}"
  frames="${LENGTH_FRAMES[$idx]}"
  run_name="gr1_pbench_robot_${label}_full_${SWEEP_ID}"
  save_dir="${GEN_OUTPUT_ROOT}/${run_name}"

  echo "============================================================"
  echo "GR1 PBench ${label}: NUM_FRAMES=${frames}, FPS=${FPS}, actual=$(python3 - <<PY
print(round(${frames} / ${FPS}, 4))
PY
)s"
  echo "Run: ${run_name}"
  echo "============================================================"

  if generation_done "${save_dir}"; then
    echo "Generation already complete: ${run_name}"
  elif [[ -d "${save_dir}" ]] && [[ -f "${save_dir}/run.log" || -f "${save_dir}/submit.log" ]]; then
    echo "Found existing generation state; waiting: ${save_dir}"
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

  run_domain_if_needed "${run_name}" "${save_dir}"
  run_quality_if_needed "${run_name}" "${save_dir}"
done

write_summary

echo
echo "Done."
