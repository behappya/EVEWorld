#!/usr/bin/env bash
# EVE · 下载 GR1 域内 IDM 权重 (DreamGen/GR00T 官方, repo: seonghyeonye/IDM_gr1)
#
# 用途: TEA 评测的锚定层 (IDM 反推生成视频动作 -> jerk/OOD 可执行性)。
# 本地 /data/.../idm_gr1_probe/ 已有该 repo 的 config.json + experiment_cfg/ + metadata.json,
# 只缺权重(*.safetensors)。本脚本把完整 repo 补齐到同一目录, 使 IDM.from_pretrained(DEST) 可直接加载。
#
# 特性: 幂等 + 断点续传 + 下完自检权重存在。可反复执行。
#
# 用法:
#   bash eveworld/data_curation/scripts/download_idm_gr1.sh              # 下到默认目录
#   DEST=/your/path bash eveworld/data_curation/scripts/download_idm_gr1.sh
#   DRY_RUN=1 bash eveworld/data_curation/scripts/download_idm_gr1.sh    # 只打印计划
set -euo pipefail

REPO_ID="${REPO_ID:-seonghyeonye/IDM_gr1}"
DEST="${DEST:-/data/datasets/gagi/idm_gr1_probe}"
# 若你的网络需要镜像, 取消下一行注释(或在调用前 export):
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 优先用 EVEWorld 环境自带的 hf CLI (其 Python 有 huggingface_hub);
# 避开 ~/.local/bin/hf —— 它 shebang 指向系统 /usr/bin/python3, 无 huggingface_hub。
HF_BIN=""
for cand in \
  "${HF_BIN_OVERRIDE:-}" \
  "/home/jovyan/miniconda/envs/EVEWorld/bin/hf" \
  "/home/jovyan/miniconda/envs/EVEWorld/bin/huggingface-cli" \
  "hf" "huggingface-cli"; do
  [[ -z "$cand" ]] && continue
  if command -v "$cand" >/dev/null 2>&1; then HF_BIN="$cand"; break; fi
done
# 兜底: 直接用 EVEWorld 的 python -m huggingface_hub 下载
GM_PY="/home/jovyan/miniconda/envs/EVEWorld/bin/python"
if [[ -z "$HF_BIN" ]]; then
  if [[ -x "$GM_PY" ]] && "$GM_PY" -c "import huggingface_hub" 2>/dev/null; then
    HF_BIN="$GM_PY -m huggingface_hub.commands.huggingface_cli"
  else
    echo "[ERR] 未找到可用 hf CLI, 且 EVEWorld 无 huggingface_hub。" >&2
    echo "      修复: ~/miniconda/envs/EVEWorld/bin/pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
  fi
fi

echo "=========================================="
echo " EVE · download IDM_gr1"
echo "  repo   : $REPO_ID"
echo "  dest   : $DEST"
echo "  hf bin : $HF_BIN"
echo "  mirror : ${HF_ENDPOINT:-<default hf.co>}"
echo "=========================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] 将执行:"
  echo "  $HF_BIN download $REPO_ID --repo-type model --local-dir $DEST"
  exit 0
fi

mkdir -p "$DEST"

# hf download 默认断点续传、跳过已存在且校验一致的文件, 幂等安全。
# 不加 --include, 拉全 repo(config/权重/experiment_cfg 全齐), 与本地已有文件合并。
echo "[*] 开始下载 (断点续传, 已存在文件会跳过)..."
"$HF_BIN" download "$REPO_ID" --repo-type model --local-dir "$DEST"

echo ""
echo "[*] 自检: 权重文件是否到位"
# HF PreTrainedModel 权重为 *.safetensors 或 pytorch_model*.bin
shopt -s nullglob
weights=( "$DEST"/*.safetensors "$DEST"/pytorch_model*.bin "$DEST"/model*.safetensors )
if [[ ${#weights[@]} -eq 0 ]]; then
  echo "[ERR] 未发现权重文件 (*.safetensors / pytorch_model*.bin)。" >&2
  echo "      当前 $DEST 内容:" >&2
  ls -la "$DEST" >&2
  echo "      可能原因: 网络/鉴权失败, 或该 repo 权重在子目录。请贴出上面 ls 结果。" >&2
  exit 2
fi

echo "[OK] 权重文件:"
for w in "${weights[@]}"; do
  sz=$(du -h "$w" | cut -f1)
  echo "     $w  ($sz)"
done

echo ""
echo "[*] 关键文件清单:"
for f in config.json experiment_cfg/conf.yaml experiment_cfg/metadata.json; do
  if [[ -e "$DEST/$f" ]]; then echo "     [有] $f"; else echo "     [缺] $f  <- 注意"; fi
done

echo ""
echo "[DONE] IDM_gr1 已就绪。加载方式:"
echo "   from gr00t.model.idm import IDM"
echo "   idm = IDM.from_pretrained('$DEST')"
echo "   # 或官方管线: python IDM_dump/dump_idm_actions.py --checkpoint '$DEST' ..."
