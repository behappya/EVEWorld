#!/usr/bin/env bash
#SBATCH --job-name=t4g_ichD_gen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# ICH-D 兑现测试池生成: 92 prompt x 4 seed x 93f (pool_pretrain_f93 同协议)。
# 与 kjob_bestofn_8gpu.sh 差异: --gen-script 换成 t4g_ichD_generate.py (D 臂推理
# 包装: 恢复 ich_d.* + 注册 hook, 否则裸 transformer = 白测), transformer 指向
# D 臂 step350 EMA 档。grep 'ichd-infer' seed 日志 = ICH 活跃证据。
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
CKPT_TRANSFORMER="${CKPT_TRANSFORMER:-${GAGI}/eve_v2_outputs/t4g_ich_D/experiments/models/checkpoint_epoch_175_step_350/transformer_ema}"
PRETRAIN="${PRETRAIN:-${GAGI}/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_v2_outputs/selfcase/pool_ichD_s350_f93}"
SEEDS="${SEEDS:-42 314 777 999}"
# GEN_SCRIPT 可覆盖: t4g_ichD_generate.py (ICH 开, 门控由 T4G_ICH_SIGMA_GATE /
# T4G_ICH_M_GATE env 决定) 或 eveworld/method/scripts/generate_eag.py (裸 backbone,
# ich_d.* 被 from_pretrained 丢弃 = ICH 全关隔离臂)。
GEN_SCRIPT="${GEN_SCRIPT:-${EVEWORLD_ROOT}/eveworld/pipeline/t4g_ichD_generate.py}"
export T4G_ICH_SIGMA_GATE="${T4G_ICH_SIGMA_GATE:-}"
export T4G_ICH_M_GATE="${T4G_ICH_M_GATE:-0}"

[[ -f "${CKPT_TRANSFORMER}/diffusion_pytorch_model.bin" ]] || { echo "缺 D 臂 EMA 档: ${CKPT_TRANSFORMER}"; exit 1; }

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export REPO_DIR
cd "${REPO_DIR}"

echo "== ICH-D 池生成: seeds=${SEEDS} ckpt=${CKPT_TRANSFORMER} -> ${OUT_ROOT} =="
echo "   gen_script=${GEN_SCRIPT} sigma_gate='${T4G_ICH_SIGMA_GATE}' m_gate=${T4G_ICH_M_GATE}"
nvidia-smi || true

"${TRAIN_PYTHON}" eveworld/method/scripts/bestofn_dispatch.py \
  --seeds ${SEEDS} \
  --data-path "${DATA_PATH}" --out-root "${OUT_ROOT}" \
  --transformer "${CKPT_TRANSFORMER}" \
  --text-encoder "${PRETRAIN}/text_encoder" \
  --vae "${PRETRAIN}/vae" \
  --lam "${LAM}" \
  --python "${TRAIN_PYTHON}" \
  --gen-script "${GEN_SCRIPT}" \
  --eag-weight 0 \
  --num-frames 93 --steps 30 --height 480 --width 768 --fps 16 \
  --limit "${LIMIT:-0}" --skip-existing
RC=$?
echo "KJOB_DONE rc=${RC}"
exit "${RC}"
