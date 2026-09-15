#!/usr/bin/env bash
# EVE P0 生死关卡编排(纯 CPU,本机可跑)。
# 前置:pip install -r eveworld/diagnosis/requirements.txt
#      并在 common/env.sh 配好 EVE_VIDEO_ROOT / EVE_OUT
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../common/env.sh"

P0="${EVE_P0}"
OUT="${EVE_OUT}/p0"; mkdir -p "${OUT}"

# 允许用环境变量指定各模型的视频目录;默认示例见下(请改成你的真实产物路径)
BASELINE_DIR="${BASELINE_DIR:-${EVE_VIDEO_ROOT}/baseline}"
SFT_DIR="${SFT_DIR:-${EVE_VIDEO_ROOT}/sft}"
META="${GR1_META:-${GR1_DATA_ROOT}/raw_hf/metadata.csv}"

eve_log "P0 步骤 1/4:事件抽取"
for tag in baseline sft; do
  d="BASELINE_DIR"; [[ "$tag" == "sft" ]] && d="SFT_DIR"
  vdir="${!d}"
  if [[ -d "$vdir" ]]; then
    eve_log "  抽取 $tag <- $vdir"
    ${PYBIN} "${P0}/scripts/event_extract.py" --video-dir "$vdir" --meta "$META" --out-dir "${OUT}/events_${tag}"
  else
    eve_warn "  跳过 $tag:目录不存在 $vdir(请设 ${d}=... 指向你的生成视频)"
  fi
done

eve_log "P0 步骤 2/4:过程度量"
for tag in baseline sft; do
  [[ -d "${OUT}/events_${tag}" ]] || continue
  ${PYBIN} "${P0}/metrics/process_metrics.py" --events-dir "${OUT}/events_${tag}" \
    --out "${OUT}/metrics_${tag}.json" --tag "$tag"
done

eve_log "P0 步骤 3/4:正交性(需你提供物理分表 PHYS_CSV,列名 PHYS_COL)"
PHYS_CSV="${PHYS_CSV:-}"; PHYS_COL="${PHYS_COL:-pbench_domain}"
if [[ -n "$PHYS_CSV" && -f "${OUT}/metrics_baseline.json" ]]; then
  ${PYBIN} "${P0}/metrics/orthogonality.py" --process "${OUT}/metrics_baseline.json" \
    --physics "$PHYS_CSV" --phys-col "$PHYS_COL" --out-prefix "${OUT}/ortho_baseline"
else
  eve_warn "  跳过正交性:请设 PHYS_CSV=你的逐视频物理分表(video,${PHYS_COL})"
fi

eve_log "P0 步骤 4/4:提示人工标注(生死关卡)"
cat <<TIP
  下一步(人工,必做):
   1) 采样 150-300 条: 两名标注者各跑
      ${PYBIN} ${P0}/annotation/annotate.py --video-dir <SAMPLED_DIR> --out ${OUT}/lab_A.jsonl --resume
      ${PYBIN} ${P0}/annotation/annotate.py --video-dir <SAMPLED_DIR> --out ${OUT}/lab_B.jsonl --resume
   2) 一致性 κ:
      ${PYBIN} ${P0}/annotation/agreement.py kappa --a ${OUT}/lab_A.jsonl --b ${OUT}/lab_B.jsonl
   3) 校准自动度量:
      ${PYBIN} ${P0}/annotation/agreement.py calib --labels ${OUT}/lab_A.jsonl --metrics ${OUT}/metrics_baseline.json

  go/no-go 判据(三项全过才进方法阶段):
    频繁性: LAZINESS_RATE 足够高(见 metrics_*.json)
    可测性: calib 的 agreement≥0.7 或 spearman≥0.6
    正交性: ortho_baseline.json 的 pearson |r|<0.4 且 存在物理分高但作弊的样本
TIP
eve_log "P0 自动部分完成。产物: ${OUT}"
