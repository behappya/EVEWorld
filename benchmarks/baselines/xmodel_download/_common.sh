#!/usr/bin/env bash
# 跨模型下载公共库。被各 dl_*.sh 通过 `source` 引入。
# 目标：为 EVE 跨模型 Model-Laziness 对比下载原生 I2V 视频生成模型权重。
# 全部 CPU 操作，可在本机（无 GPU）跑；hf download 断点续传，可重复执行。
# 大文件落在 /data，不进 git 仓库。

set -euo pipefail

# ---- 共享配置 ----
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
# hf_transfer 在部分环境不稳，默认关闭；想加速可 export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# 禁用 Xet 传输协议：hf_xet 未装但 hub 1.23 仍尝试走 Xet, 对 Cosmos 等 repo 会崩
# (RuntimeError: Unable to parse string as hex hash value)。退回普通 HTTP 下载。
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_world1}"   # 该 env 内 hf==1.23.0 可用
XMODEL_ROOT="${XMODEL_ROOT:-/data/datasets/gagi/xmodels}"

activate_env() {
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  if ! command -v hf >/dev/null 2>&1; then
    echo "[FATAL] 找不到 hf CLI（env=${CONDA_ENV}）。" >&2
    exit 1
  fi
}

# 检查 gated 模型的登录态；未登录只警告不退出（Wan 免登录）。
check_login_if_gated() {
  local gated="$1"
  if [[ "${gated}" == "1" ]]; then
    if ! hf auth whoami >/dev/null 2>&1; then
      cat >&2 <<EOF
[WARN] 该模型是 gated repo，需要先：
   1) 浏览器打开 model 页点 "Agree/Access repository"
   2) 终端里: hf auth login  （粘贴 https://huggingface.co/settings/tokens 的 token）
   或给本脚本传 HF_TOKEN=hf_xxx 环境变量。
   未登录会在下载时 401。仍尝试继续……
EOF
    fi
  fi
}

# 统一的下载封装。
# 用法: dl_repo <repo_id> <dst_dir> [extra hf download args...]
# 注意：--include/--exclude 每个 flag 只接受一个 glob，必须重复 flag，
#       否则后续 glob 落入位置参数触发 "Ignoring --include/--exclude" 并下载整个 repo。
dl_repo() {
  local repo_id="$1"; shift
  local dst="$1"; shift
  mkdir -p "${dst}"
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local log="${dst}/download_${ts}.log"
  echo "[dl] repo   = ${repo_id}"
  echo "[dl] dst    = ${dst}"
  echo "[dl] log    = ${log}"
  local token_args=()
  if [[ -n "${HF_TOKEN:-}" ]]; then token_args=(--token "${HF_TOKEN}"); fi
  hf download "${repo_id}" \
    --local-dir "${dst}" \
    "${token_args[@]}" \
    "$@" \
    2>&1 | tee "${log}"
}

# 下载后校验：确认关键子目录/文件存在。
# 用法: verify_paths <base_dir> <path1> <path2> ...
verify_paths() {
  local base="$1"; shift
  local ok=1
  for p in "$@"; do
    if [[ -e "${base}/${p}" ]]; then
      echo "  OK  ${base}/${p}"
    else
      echo "  缺  ${base}/${p}"; ok=0
    fi
  done
  du -sh "${base}" 2>/dev/null || true
  if [[ "${ok}" == "1" ]]; then
    echo "[dl] 校验通过：${base}"
  else
    echo "[dl] 有缺失，请检查日志后重跑本脚本（会断点续传）。" >&2
    return 1
  fi
}
