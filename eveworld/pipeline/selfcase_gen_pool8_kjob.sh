#!/usr/bin/env bash
#SBATCH --job-name=selfcase_gen_pool8
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# 重考池生成 · 8 卡满载版 (4 协议 seed x 2 题目半片)。
# 协议与 kjob_bestofn_8gpu 完全一致 (generate_eag, EAG_WEIGHT=0, 480x768/30步/93f/fps16),
# 仅把"每 seed 1 卡"换成"seed x half 钉 8 卡"。题目半片须先用不相交切分生成。
set -uo pipefail
[[ "${ALLOW_LOCAL_RUN:-0}" != "1" && "$(hostname)" == coder-workspace-* ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }; done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"
GAGI="${GAGI_ROOT:-/data/datasets/gagi}"

MODEL_DIR="${MODEL_DIR:?必填: transformer/vae/text_encoder 合成目录}"
OUT_ROOT="${OUT_ROOT:?必填: 池输出目录}"
# 分片列表: DATA_HALF0..DATA_HALF7 按序收集, 至少 1 片; seeds x 分片数 <= 8
DATA_PATHS=()
for i in 0 1 2 3 4 5 6 7; do
  v="DATA_HALF${i}"
  [[ -n "${!v:-}" ]] && DATA_PATHS+=("${!v}")
done
[[ ${#DATA_PATHS[@]} -ge 1 ]] || { echo "缺 DATA_HALF0" >&2; exit 1; }
SEEDS="${SEEDS:-42 314 777 999}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"

source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate "${CONDA_ENV:-EVEWorld}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

echo "== 重考池 8 卡满载生成: seeds=${SEEDS} x ${#DATA_PATHS[@]} shard -> ${OUT_ROOT} =="
nvidia-smi || true

"${TRAIN_PYTHON}" eveworld/pipeline/selfcase_gen_pool8_dispatch.py \
  --seeds ${SEEDS} \
  --data-paths "${DATA_PATHS[@]}" \
  --out-root "${OUT_ROOT}" \
  --transformer "${MODEL_DIR}/transformer" \
  --text-encoder "${MODEL_DIR}/text_encoder" \
  --vae "${MODEL_DIR}/vae" \
  --lam "${LAM}" \
  --python "${TRAIN_PYTHON}" \
  --gen-script "${EVEWORLD_ROOT}/eveworld/method/scripts/generate_eag.py" \
  --eag-weight 0 --num-frames 93 --steps 30 --height 480 --width 768 --fps 16
echo KJOB_DONE
