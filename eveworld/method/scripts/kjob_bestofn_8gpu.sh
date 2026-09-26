#!/usr/bin/env bash
#SBATCH --job-name=eve_bestofn_8gpu
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# EVE · best-of-N 生成 —— 单节点 8 卡并行(每 seed 占 1 卡, 卡内串行跑 92 条)。
# 取代原来"每 seed 1 个 kjob"的碎片化做法(8 个 job 散在 7 个节点各占 1 卡)。
# 8 个 seed 并行分发到 GPU 0..7(Python 分发器编排)。补齐模式(--skip-existing)不重跑已有 mp4。
set -uo pipefail

if [[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]]; then
  echo "Refusing to run on workspace host (no GPU)." >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
MODEL_DIR="${MODEL_DIR:-${GAGI}/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_outputs/bestofn}"

SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
EAG_WEIGHT="${EAG_WEIGHT:-0}"          # 0 = baseline 采样(best-of-N 用)
NUM_FRAMES="${NUM_FRAMES:-93}"
STEPS="${STEPS:-30}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"; FPS="${FPS:-16}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"    # 1 = 补齐模式(默认); 0 = 全部重生成

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

read -r -a SEED_ARR <<< "${SEEDS}"
echo "=========================================="
echo " EVE best-of-N 单节点8卡  host=$(hostname)"
echo "  seeds   = ${SEEDS}  (共 ${#SEED_ARR[@]} 个 -> GPU 0..$(( ${#SEED_ARR[@]} - 1 )), Python分发器编排)"
echo "  eag_w=${EAG_WEIGHT} frames=${NUM_FRAMES} steps=${STEPS} limit=${LIMIT} skip_existing=${SKIP_EXISTING}"
echo "  out     = ${OUT_ROOT}"
echo "=========================================="
nvidia-smi || true

SKIP_ARG=(); [[ "${SKIP_EXISTING}" == "1" ]] && SKIP_ARG=(--skip-existing)

# 多卡编排放进 Python 分发器(kjobctl 的 slurm 解释器不支持 bash 后台作业语法)。
"${TRAIN_PYTHON}" eveworld/method/scripts/bestofn_dispatch.py \
  --seeds ${SEEDS} \
  --data-path "${DATA_PATH}" --out-root "${OUT_ROOT}" \
  --transformer "${MODEL_DIR}/transformer" \
  --text-encoder "${MODEL_DIR}/text_encoder" \
  --vae "${MODEL_DIR}/vae" \
  --lam "${LAM}" \
  --python "${TRAIN_PYTHON}" \
  --gen-script "${EVEWORLD_ROOT}/eveworld/method/scripts/generate_eag.py" \
  --eag-weight "${EAG_WEIGHT}" \
  --num-frames "${NUM_FRAMES}" --steps "${STEPS}" \
  --height "${HEIGHT}" --width "${WIDTH}" --fps "${FPS}" \
  --limit "${LIMIT}" "${SKIP_ARG[@]}"
exit "$?"
