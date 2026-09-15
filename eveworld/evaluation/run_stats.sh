#!/usr/bin/env bash
# EVE · 统计聚合(纯 CPU)。把多模型/多seed的过程度量做 bootstrap CI + 配对检验 + 画图。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${HERE}/../../common/env.sh"
METRICS_DIR="${METRICS_DIR:-${EVE_OUT}/eval/metrics}"
OUT="${EVE_OUT}/eval/stats"; mkdir -p "${OUT}"
if [[ ! -d "${METRICS_DIR}" ]]; then
  eve_warn "无 ${METRICS_DIR};请先跑评测生成度量。"; exit 0
fi
${PYBIN} "${HERE}/aggregate_stats.py" --metrics-dir "${METRICS_DIR}" --out-dir "${OUT}"
eve_log "统计完成 -> ${OUT}"
