#!/usr/bin/env bash
#SBATCH --job-name=t4g_probe
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# EVE · Track4Gen 特征可追踪性探针 kjob payload (单节点, 默认 8 卡视频分片并行)。
# 纯探针: 导 28 层特征 + 算指标 (动组可追踪性 / 静组稳定性), 不做任何训练。
# 多卡编排在 Python 里 (t4g_probe.py --dispatch-gpus N); kjobctl 的 slurm 解释器不支持
# bash 后台作业语法, 故不用 & / wait。NGPU=1 时退化为单卡跑全部视频。
set -uo pipefail

# 禁止在无 GPU 的工作机误跑 (纪律: 不在本机跑 GPU)。
if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host (no GPU). Submit via launch_t4g_probe_kjob.sh" >&2
  exit 2
fi

# 透传: 尾随 KEY=VALUE 逐个 export (与 kjob_eve_eag_generate.sh 同机制)。
for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
# 探针默认底座 = anmix_s200 (当前最优, 对应 loss 训练底座候选)。
MODEL_DIR="${MODEL_DIR:-${GAGI}/eve_v2_outputs/anchor_models/probe_anmix_s200}"
TRANSFORMER="${TRANSFORMER:-${MODEL_DIR}/transformer}"
VAE="${VAE:-${MODEL_DIR}/vae}"
TEXT_ENCODER="${TEXT_ENCODER:-${MODEL_DIR}/text_encoder}"

VIDEO_ROOT="${VIDEO_ROOT:-${GAGI}/gr1_finetune_data/raw_data}"
# 12-20 条真实 GT, 含 shelf 垂直搬运任务 (13/32/76)。
VIDEO_IDS="${VIDEO_IDS:-13,32,76,14,15,16,17,18,19,20,21,23,24,25,26,27}"
SIGMAS="${SIGMAS:-0.25,0.7,2.0,5.0}"      # 低/中/高噪声级 (Track4Gen: 特征质量随噪声变)
NUM_FRAMES="${NUM_FRAMES:-93}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"; FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"
NGPU="${NGPU:-8}"                          # 单节点卡数 (1-8), 视频分片并行
EPE_GO="${EPE_GO:-2.0}"                    # 判据: 端点误差 < 2 cell
STAB_GO="${STAB_GO:-0.90}"                 # 判据: 静组自相似 > 0.9
P_HI="${P_HI:-85}"; P_LO="${P_LO:-40}"
OUT_DIR="${OUT_DIR:-${GAGI}/eve_v2_outputs/track4gen_probe/anmix_s200}"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"

echo "=========================================="
echo " Track4Gen 探针  host=$(hostname)"
echo "  model    = ${TRANSFORMER}"
echo "  videos   = ${VIDEO_IDS}"
echo "  sigmas   = ${SIGMAS}   frames=${NUM_FRAMES} ${HEIGHT}x${WIDTH}"
echo "  ngpu     = ${NGPU}   dtype=${DTYPE}"
echo "  judge    = EPE<${EPE_GO} cell & STAB>${STAB_GO}"
echo "  out      = ${OUT_DIR}"
echo "=========================================="
nvidia-smi || true

"${TRAIN_PYTHON}" t4g_probe.py \
  --transformer "${TRANSFORMER}" --vae "${VAE}" --text-encoder "${TEXT_ENCODER}" \
  --video-root "${VIDEO_ROOT}" --video-ids "${VIDEO_IDS}" \
  --sigmas "${SIGMAS}" \
  --num-frames "${NUM_FRAMES}" --height "${HEIGHT}" --width "${WIDTH}" --fps "${FPS}" \
  --dtype "${DTYPE}" --device cuda \
  --p-hi "${P_HI}" --p-lo "${P_LO}" --epe-go "${EPE_GO}" --stab-go "${STAB_GO}" \
  --out-dir "${OUT_DIR}" \
  --dispatch-gpus "${NGPU}"
RC=$?

echo "[t4g] payload done rc=${RC} -> ${OUT_DIR}/t4g_probe_summary.json"
exit "${RC}"
