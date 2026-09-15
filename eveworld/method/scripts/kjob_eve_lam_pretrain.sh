#!/usr/bin/env bash
#SBATCH --job-name=eve_lam_pretrain
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# EVE · LAD 预训 kjob payload(单卡足够: latent 离线, LAD 仅 ~0.4M 参数)。
# 两步: (1) 若 latent 缓存不存在, 用 Wan VAE 离线编码; (2) 自监督训 LAD + go/no-go 验证。
set -euo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host (no GPU). Submit via launch_eve_lam_kjob.sh" >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
VIDEO_DIR="${LAM_VIDEO_DIR:-${GAGI}/gr1_finetune_data/raw_hf/gr1}"
VAE_PATH="${VAE_PATH:-${GAGI}/giga_world_0_video_pretrain/vae}"
LAT_CACHE="${LAT_CACHE:-${GAGI}/eve_outputs/latents/gr1_real.pt}"
LAM_OUT="${LAM_OUT:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"

NUM_FRAMES="${NUM_FRAMES:-49}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"
STEPS="${STEPS:-8000}"; WINDOW="${WINDOW:-12}"; BATCH="${BATCH:-8}"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/eve:${PYTHONPATH:-}"
cd "${REPO_DIR}"

echo "=========================================="
echo " EVE LAM pretrain kjob"
echo "  video_dir  = ${VIDEO_DIR}"
echo "  lat_cache  = ${LAT_CACHE}"
echo "  lam_out    = ${LAM_OUT}"
echo "  frames=${NUM_FRAMES} hw=${HEIGHT}x${WIDTH} steps=${STEPS} window=${WINDOW} batch=${BATCH}"
echo "  python     = ${TRAIN_PYTHON}"
echo "=========================================="

# (1) 编码 latent。跳过条件: 缓存存在 且 视频条数 >= MIN_CACHE_N 且 未强制。
#     防止复用小冒烟缓存(如 8 条)导致正式训练数据不足。
MIN_CACHE_N="${MIN_CACHE_N:-50}"
need_encode=1
if [[ -f "${LAT_CACHE}" && "${FORCE_ENCODE:-0}" != "1" ]]; then
  cache_n=$("${TRAIN_PYTHON}" -c "import torch;print(len(torch.load('${LAT_CACHE}',map_location='cpu')['latents']))" 2>/dev/null || echo 0)
  if [[ "${cache_n}" -ge "${MIN_CACHE_N}" ]]; then
    echo "[lam] latent 缓存已存在且足量(${cache_n}条 >= ${MIN_CACHE_N}), 跳过编码: ${LAT_CACHE}"
    need_encode=0
  else
    echo "[lam] latent 缓存仅 ${cache_n} 条 (< ${MIN_CACHE_N}), 判定为冒烟缓存 -> 重新编码全量"
  fi
fi
if [[ "${need_encode}" == "1" ]]; then
  echo "[lam] 编码 latent -> ${LAT_CACHE}"
  "${TRAIN_PYTHON}" eveworld/method/scripts/encode_latents.py \
    --video-dir "${VIDEO_DIR}" --out "${LAT_CACHE}" \
    --vae-path "${VAE_PATH}" --num-frames "${NUM_FRAMES}" \
    --height "${HEIGHT}" --width "${WIDTH}" ${ENCODE_LIMIT:+--limit ${ENCODE_LIMIT}}
fi

# (2) 训 LAD + go/no-go
echo "[lam] 训练 LAD + go/no-go 验证"
"${TRAIN_PYTHON}" eveworld/method/scripts/train_lam.py \
  --latents "${LAT_CACHE}" --out "${LAM_OUT}" \
  --steps "${STEPS}" --window "${WINDOW}" --batch "${BATCH}"

echo "[lam] DONE. checkpoint: ${LAM_OUT}"
