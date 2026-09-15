#!/usr/bin/env bash
# 下载 Cosmos-Predict2.5-2B (NVIDIA, 世界模型 2B, Video2World 支持单图→视频)。
# GATED: 需先在网页同意 NVIDIA Open Model License 并 hf auth login（或传 HF_TOKEN）。
# 机器人 physical-AI 定位，与 pick-place 叙事最契合。可与其它 dl_*.sh 并行执行。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-nvidia/Cosmos-Predict2.5-2B}"
DST="${DST:-${XMODEL_ROOT}/cosmos_predict25_2b}"

activate_env
check_login_if_gated 1

echo "==== [Cosmos-Predict2.5-2B] 开始下载 (gated) ===="
# Cosmos 结构非标准 diffusers，先整仓下（排除明显训练态），跑通推理再谈裁剪。
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*optim*"

# Cosmos repo 内容形态不确定，校验放宽为目录非空即可。
echo "[dl] 目录内容概览："
ls -la "${DST}" 2>/dev/null | head -30
du -sh "${DST}" 2>/dev/null || true
if [[ -n "$(ls -A "${DST}" 2>/dev/null)" ]]; then
  echo "[dl] Cosmos-Predict2.5-2B 目录非空，检查上面文件清单确认权重完整。"
else
  echo "[dl] 目录为空，可能未登录导致 401，请检查日志。" >&2
  exit 1
fi
