#!/usr/bin/env bash
set -euo pipefail
# EVE · LAD 自监督预训(集群 kjob, 单卡)。
# 两步: Wan VAE 离线编码 GR1 latent -> 自监督训 LAD + go/no-go(transition_error 区分偷懒)。
#
# 用法:
#   bash eveworld/method/scripts/launch_lam_pretrain_kjob.sh
#   STEPS=12000 WINDOW=16 bash eveworld/method/scripts/launch_lam_pretrain_kjob.sh
#   ENCODE_LIMIT=8 STEPS=500 bash ...   # 小规模冒烟(先编8条, 训500步)
#   DRY_RUN=1 bash ...                  # 只打印提交计划
#
# 产物: ${GAGI}/eve_outputs/latents/gr1_real.pt (latent 缓存)
#       ${GAGI}/eve_outputs/lam/lam_gr1.pt      (LAD checkpoint)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = giga-world-0

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_eve_lam_pretrain.sh}"

PASS=(
  "REPO_DIR=${REPO_DIR}"
  "GAGI_ROOT=${GAGI_ROOT:-/data/datasets/gagi}"
  "TRAIN_VENV=${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
  "LAM_VIDEO_DIR=${LAM_VIDEO_DIR:-/data/datasets/gagi/gr1_finetune_data/raw_hf/gr1}"
  "LAT_CACHE=${LAT_CACHE:-/data/datasets/gagi/eve_outputs/latents/gr1_real.pt}"
  "LAM_OUT=${LAM_OUT:-/data/datasets/gagi/eve_outputs/lam/lam_gr1.pt}"
  "NUM_FRAMES=${NUM_FRAMES:-49}"
  "HEIGHT=${HEIGHT:-480}"
  "WIDTH=${WIDTH:-768}"
  "STEPS=${STEPS:-8000}"
  "WINDOW=${WINDOW:-12}"
  "BATCH=${BATCH:-8}"
)
[[ -n "${ENCODE_LIMIT:-}" ]] && PASS+=("ENCODE_LIMIT=${ENCODE_LIMIT}")
[[ -n "${FORCE_ENCODE:-}" ]] && PASS+=("FORCE_ENCODE=${FORCE_ENCODE}")

echo "[EVE] 提交 LAD 预训 kjob (单卡)"
echo "  JOB_SCRIPT = ${JOB_SCRIPT}"
echo "  steps=${STEPS:-8000} window=${WINDOW:-12} batch=${BATCH:-8} frames=${NUM_FRAMES:-49}"
echo "  latent 缓存 -> ${LAT_CACHE:-.../eve_outputs/latents/gr1_real.pt}"
echo "  LAD ckpt   -> ${LAM_OUT:-.../eve_outputs/lam/lam_gr1.pt}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] 将执行:"
  echo "  ${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh \\"
  printf '    %s \\\n' "${PASS[@]}"
  exit 0
fi

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" "${PASS[@]}"
