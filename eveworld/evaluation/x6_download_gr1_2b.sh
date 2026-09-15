#!/usr/bin/env bash
set -euo pipefail

# 下载 open-gigaai/GigaWorld-0-Video-GR1-2b（DreamGen 评测用, 39/38 号线）。
# 可断点续传, 中断后重跑即可。text_encoder(t5-11b) 与 VAE 本地已有, 不重复下载。
# 用法: bash x6_download_gr1_2b.sh
# 下载完成后告诉 Claude, 由其组装模型目录并提交 92x8 推理 kjob。

ROOT_DIR="${ROOT_DIR:-/data/datasets/gagi}"
DEST="${DEST:-${ROOT_DIR}/gigaworld0_gr1_2b}"
REPO_ID="open-gigaai/GigaWorld-0-Video-GR1-2b"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-${ROOT_DIR}/.hf_xet_cache}"

source /home/jovyan/miniconda/etc/profile.d/conda.sh
conda activate giga_models

python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="${REPO_ID}", local_dir="${DEST}", max_workers=8)
print("下载完成:", p)
PY

echo "== 布局探测 =="
find "${DEST}" -maxdepth 2 -name "config.json" -o -maxdepth 2 -name "*.safetensors" | head -20
echo "完成。请回到会话告知 Claude, 路径: ${DEST}"
