#!/usr/bin/env bash
set -euo pipefail

# 补跑短时长档的「纯推理」驱动（不含任何评测 / judge / PA-II / Domain）。
#
# 单节点 8 卡；串行：一个档生成完再跑下一个。
# 断点续跑：已有完整 generation_summary.json 的档自动跳过；
#          若发现某档已在跑（有 run.log/submit.log 但没跑完），则等待而不重复提交。
#
# 覆盖 4 条线（缺哪个档补哪个，其余自动跳过）：
#   1. PBench Pretrain      期望补 7.8s（3.8/5.8/9.8… 已有）
#   2. PBench GR1/SFT       期望补 7.8s
#   3. DreamGen Pretrain    期望补 3.8s / 7.8s
#   4. DreamGen GR1/SFT     期望补 3.8s / 7.8s
#
# 帧数（16fps）：3.8s=61，7.8s=125。均满足 VAE 4k+1 且 <= max_frames=128（域内，时间维不外推）。
#
# 用法：
#   ./benchmarks/dreamgenbench/run_short_length_generation_only.sh                # 跑全部四条线
#   TASKS="pbench_pretrain pbench_sft" ./benchmarks/dreamgenbench/run_short_length_generation_only.sh
#   LENGTHS="3p8s:61 7p8s:125" ./benchmarks/dreamgenbench/run_short_length_generation_only.sh
#   DRY_RUN=1 ./benchmarks/dreamgenbench/run_short_length_generation_only.sh      # 只打印将要做什么，不提交

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- 全局可覆盖参数（默认对齐已有 sweep 产物，保证结果可并入现有曲线）----
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
SEED="${SEED:-6666}"
DRY_RUN="${DRY_RUN:-0}"

# 要补的时长档（label:frames，空格分隔）。DreamGen 用全部；PBench 只取其中缺的档。
LENGTHS="${LENGTHS:-3p8s:61 7p8s:125}"
# PBench 只补 7.8s（3.8s 已存在）；如需也补 3.8s 自行加入。
PBENCH_LENGTHS="${PBENCH_LENGTHS:-7p8s:125}"

# 要跑哪些线（默认：DreamGen 两条线优先，PBench 两条线在后）
TASKS="${TASKS:-dreamgen_pretrain dreamgen_sft pbench_pretrain pbench_sft}"

EVAL_ROOT="${EVAL_ROOT:-/data/datasets/gagi}"

# ---- PBench 相关 ----
PBENCH_DATA_PATH="${PBENCH_DATA_PATH:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json}"
PBENCH_EXPECTED="${PBENCH_EXPECTED:-174}"
PBENCH_HEIGHT="${PBENCH_HEIGHT:-480}"
PBENCH_WIDTH="${PBENCH_WIDTH:-640}"
PBENCH_PRETRAIN_GEN_ROOT="${PBENCH_PRETRAIN_GEN_ROOT:-${EVAL_ROOT}/giga_world_0_outputs/pbench_robot_length_sweep}"
PBENCH_PRETRAIN_SWEEP_ID="${PBENCH_PRETRAIN_SWEEP_ID:-pbench_len_sweep_full_8gpu}"
PBENCH_SFT_GEN_ROOT="${PBENCH_SFT_GEN_ROOT:-${EVAL_ROOT}/giga_world_0_outputs/gr1_pbench_robot_length_sweep}"
PBENCH_SFT_SWEEP_ID="${PBENCH_SFT_SWEEP_ID:-gr1_pbench_len_sweep_full_8gpu}"

# SFT checkpoint（PBench SFT 与 DreamGen SFT 共用）
SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-${EVAL_ROOT}/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
GR1_TRANSFORMER_SUBDIR="${GR1_TRANSFORMER_SUBDIR:-transformer_ema}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${EVAL_ROOT}/giga_world_0_video_pretrain}"
# PBench SFT 用的合成 model 目录（transformer_ema + 复用 pretrain 的 text_encoder/vae 软链）
PBENCH_SFT_MODEL_DIR="${PBENCH_SFT_MODEL_DIR:-${PBENCH_SFT_GEN_ROOT}/model_gr1_checkpoint_epoch_100_step_200_${GR1_TRANSFORMER_SUBDIR}}"

# ---- DreamGen 相关 ----
DREAMGEN_EVAL_ROOT="${DREAMGEN_EVAL_ROOT:-${EVAL_ROOT}/gr1_dreamgen_eval}"
DREAMGEN_OUTPUT_ROOT="${DREAMGEN_OUTPUT_ROOT:-${DREAMGEN_EVAL_ROOT}/generated_side_by_side}"
DREAMGEN_EXPECTED="${DREAMGEN_EXPECTED:-92}"
DREAMGEN_HEIGHT="${DREAMGEN_HEIGHT:-480}"
DREAMGEN_WIDTH="${DREAMGEN_WIDTH:-768}"
# 固定 SWEEP_ID（非时间戳），保证 run_name 确定、可续跑
DREAMGEN_SWEEP_ID="${DREAMGEN_SWEEP_ID:-short_3p8_7p8}"

# --------------------------------------------------------------------------

json_value() {
  local path="$1" key="$2"
  python3 - "$path" "$key" <<'PY'
import json, sys
path, key = sys.argv[1:3]
try:
    with open(path, "r", encoding="utf-8") as f:
        print(json.load(f).get(key, ""))
except Exception:
    print("")
PY
}

generation_done() {  # $1=save_dir  $2=expected
  local summary="$1/generation_summary.json"
  [[ -f "${summary}" ]] || return 1
  local gen req
  gen="$(json_value "${summary}" generated_count)"
  req="$(json_value "${summary}" request_count)"
  [[ "${gen}" == "$2" && "${req}" == "$2" ]]
}

mp4_count() { find "$1" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l; }

wait_for_generation() {  # $1=run_name  $2=save_dir  $3=expected
  local run_name="$1" save_dir="$2" expected="$3"
  local log_path="${save_dir}/run.log"
  echo "  等待生成: ${run_name}"
  while true; do
    if [[ -f "${log_path}" ]] && rg -q "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}"; then
      echo "  生成疑似失败，检查日志: ${log_path}" >&2
      rg -n "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}" -S | tail -n 60 >&2
      exit 1
    fi
    if generation_done "${save_dir}" "${expected}"; then
      echo "  生成完成: ${expected}/${expected}"
      break
    fi
    echo "    当前 mp4 数: $(mp4_count "${save_dir}")/${expected}；60s 后再查"
    sleep 60
  done
}

prepare_sft_model_dir() {  # 幂等：为 PBench SFT 合成 model 目录（软链）
  local src="${SFT_CHECKPOINT_DIR}/${GR1_TRANSFORMER_SUBDIR}"
  [[ -d "${src}" ]] || { echo "缺 SFT transformer 目录: ${src}" >&2; exit 1; }
  [[ -f "${src}/config.json" ]] || { echo "缺 transformer config: ${src}/config.json" >&2; exit 1; }
  [[ -d "${PRETRAIN_DIR}/text_encoder" ]] || { echo "缺 text_encoder: ${PRETRAIN_DIR}/text_encoder" >&2; exit 1; }
  [[ -d "${PRETRAIN_DIR}/vae" ]] || { echo "缺 vae: ${PRETRAIN_DIR}/vae" >&2; exit 1; }
  mkdir -p "${PBENCH_SFT_MODEL_DIR}"
  ln -sfn "${src}" "${PBENCH_SFT_MODEL_DIR}/transformer"
  ln -sfn "${PRETRAIN_DIR}/text_encoder" "${PBENCH_SFT_MODEL_DIR}/text_encoder"
  ln -sfn "${PRETRAIN_DIR}/vae" "${PBENCH_SFT_MODEL_DIR}/vae"
}

# 跑一个 PBench 档（pretrain 或 sft）
run_pbench_one() {  # $1=run_name  $2=save_dir  $3=frames  $4=model_dir(空=pretrain)
  local run_name="$1" save_dir="$2" frames="$3" model_dir="${4:-}"
  echo "------------------------------------------------------------"
  echo "PBench 生成: ${run_name}  (NUM_FRAMES=${frames}, $(python3 -c "print(round(${frames}/${FPS},4))")s)"
  if generation_done "${save_dir}" "${PBENCH_EXPECTED}"; then
    echo "  已完成，跳过。"; return
  fi
  if [[ -d "${save_dir}" && ( -f "${save_dir}/run.log" || -f "${save_dir}/submit.log" ) ]]; then
    echo "  发现进行中的任务，等待而非重复提交。"
    wait_for_generation "${run_name}" "${save_dir}" "${PBENCH_EXPECTED}"; return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DRY_RUN] 将提交: MODEL_DIR='${model_dir}' NUM_FRAMES=${frames} SAVE_DIR=${save_dir}"; return
  fi
  MODE="full" \
  RUN_NAME="${run_name}" \
  OUTPUT_ROOT="$(dirname "${save_dir}")" \
  SAVE_DIR="${save_dir}" \
  MODEL_DIR="${model_dir}" \
  DATA_PATH="${PBENCH_DATA_PATH}" \
  DATA_LIMIT="0" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  GPU_IDS="${GPU_IDS}" \
  ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE}" \
  NUM_FRAMES="${frames}" \
  FPS="${FPS}" \
  HEIGHT="${PBENCH_HEIGHT}" \
  WIDTH="${PBENCH_WIDTH}" \
  SEED="${SEED}" \
  GPU_MEMORY_SAMPLES="${save_dir}/gpu_memory_samples.csv" \
  GPU_MEMORY_PEAK="${save_dir}/gpu_memory_peak.json" \
  "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_kjob.sh"
  wait_for_generation "${run_name}" "${save_dir}" "${PBENCH_EXPECTED}"
}

# 跑一个 DreamGen 档（pretrain 或 sft）
run_dreamgen_one() {  # $1=model_key(sft|pretrain)  $2=checkpoint_dir  $3=use_ema  $4=label  $5=frames
  local model_key="$1" ckpt="$2" use_ema="$3" label="$4" frames="$5"
  local run_name="${model_key}_dreamgen_8gpu_${label}_full_${DREAMGEN_SWEEP_ID}"
  local save_dir="${DREAMGEN_OUTPUT_ROOT}/${run_name}"
  echo "------------------------------------------------------------"
  echo "DreamGen 生成: ${run_name}  (NUM_FRAMES=${frames}, $(python3 -c "print(round(${frames}/${FPS},4))")s)"
  if generation_done "${save_dir}" "${DREAMGEN_EXPECTED}"; then
    echo "  已完成，跳过。"; return
  fi
  if [[ -d "${save_dir}" && ( -f "${save_dir}/run.log" || -f "${save_dir}/submit.log" ) ]]; then
    echo "  发现进行中的任务，等待而非重复提交。"
    wait_for_generation "${run_name}" "${save_dir}" "${DREAMGEN_EXPECTED}"; return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DRY_RUN] 将提交: CHECKPOINT_DIR=${ckpt} USE_EMA=${use_ema} NUM_FRAMES=${frames} SAVE_DIR=${save_dir}"; return
  fi
  CHECKPOINT_DIR="${ckpt}" \
  PRETRAIN_DIR="${PRETRAIN_DIR}" \
  USE_EMA="${use_ema}" \
  EVAL_ROOT="${DREAMGEN_EVAL_ROOT}" \
  OUTPUT_ROOT="${DREAMGEN_OUTPUT_ROOT}" \
  RUN_NAME="${run_name}" \
  SAVE_DIR="${save_dir}" \
  SUMMARY_PATH="${save_dir}/generation_summary.json" \
  GPU_IDS="${GPU_IDS}" \
  NUM_FRAMES="${frames}" \
  FPS="${FPS}" \
  HEIGHT="${DREAMGEN_HEIGHT}" \
  WIDTH="${DREAMGEN_WIDTH}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  SEED="${SEED}" \
  "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_dreamgen_generation_kjob.sh"
  wait_for_generation "${run_name}" "${save_dir}" "${DREAMGEN_EXPECTED}"
}

# --------------------------------------------------------------------------
cd "${REPO_DIR}"

echo "============================================================"
echo "短时长档纯推理补跑（无 judge）"
echo "TASKS:            ${TASKS}"
echo "DreamGen 档:      ${LENGTHS}"
echo "PBench 档:        ${PBENCH_LENGTHS}"
echo "GPU_IDS:          ${GPU_IDS}"
echo "Steps/FPS/Seed:   ${NUM_INFERENCE_STEPS} / ${FPS} / ${SEED}"
echo "DRY_RUN:          ${DRY_RUN}"
echo "============================================================"

for task in ${TASKS}; do
  case "${task}" in
    pbench_pretrain)
      echo; echo "### 任务: PBench Pretrain"
      for item in ${PBENCH_LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_name="pbench_robot_${label}_full_${PBENCH_PRETRAIN_SWEEP_ID}"
        run_pbench_one "${run_name}" "${PBENCH_PRETRAIN_GEN_ROOT}/${run_name}" "${frames}" ""
      done
      ;;
    pbench_sft)
      echo; echo "### 任务: PBench GR1/SFT"
      prepare_sft_model_dir
      for item in ${PBENCH_LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_name="gr1_pbench_robot_${label}_full_${PBENCH_SFT_SWEEP_ID}"
        run_pbench_one "${run_name}" "${PBENCH_SFT_GEN_ROOT}/${run_name}" "${frames}" "${PBENCH_SFT_MODEL_DIR}"
      done
      ;;
    dreamgen_pretrain)
      echo; echo "### 任务: DreamGen Pretrain"
      for item in ${LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_dreamgen_one "pretrain" "${PRETRAIN_DIR}" "0" "${label}" "${frames}"
      done
      ;;
    dreamgen_sft)
      echo; echo "### 任务: DreamGen GR1/SFT"
      for item in ${LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_dreamgen_one "sft" "${SFT_CHECKPOINT_DIR}" "1" "${label}" "${frames}"
      done
      ;;
    *)
      echo "未知 TASK: ${task}（可选: pbench_pretrain pbench_sft dreamgen_pretrain dreamgen_sft）" >&2
      exit 1
      ;;
  esac
done

echo
echo "============================================================"
echo "全部指定任务已结束。生成产物位置："
echo "  PBench Pretrain: ${PBENCH_PRETRAIN_GEN_ROOT}/pbench_robot_<档>_full_${PBENCH_PRETRAIN_SWEEP_ID}"
echo "  PBench SFT:      ${PBENCH_SFT_GEN_ROOT}/gr1_pbench_robot_<档>_full_${PBENCH_SFT_SWEEP_ID}"
echo "  DreamGen:        ${DREAMGEN_OUTPUT_ROOT}/{pretrain,sft}_dreamgen_8gpu_<档>_full_${DREAMGEN_SWEEP_ID}"
echo "============================================================"
