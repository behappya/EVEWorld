#!/usr/bin/env bash
set -uo pipefail

# 跨模型 DreamGen 生成 · 补缺档位(串行, 单节点 8 卡数据并行, 一个 job 跑完再下一个)。
# 复用 launch_xmodel_dreamgen_infer_kjob.sh(与 run_all_xmodel_dreamgen.sh 同款范式)。
#
# 现状(2026-07-14, full 非 smoke):
#   model               3.8s  5.8s  7.8s  9.8s  15.8s
#   wan22_ti2v_5b        缺    92    缺    92    92
#   wan22_i2v_a14b       缺    92    缺    92    92
#   cogvideox15_5b_i2v   缺    92    缺    92    16(未跑完)
# 本脚本默认补: 3.8s + 7.8s (3 模型 x 2 档 = 6 job)。可选带上 CogVideoX 15.8s 补全。
#
# 时长档 -> 帧数 @16fps: 3.8s=61  5.8s=93  7.8s=125  9.8s=157  15.8s=253
#   (CogVideoX 约束 latent=(nf-1)/4+1 须偶: 61->16✓ 125->32✓ 253->64✓)
#
# 幂等: 若目标 run 已有 generation_summary.json 且视频数达标, 自动跳过。
#
# 用法:
#   bash run_xmodel_fill_missing.sh                    # 补 3.8s + 7.8s (6 job)
#   DURS="38" bash run_xmodel_fill_missing.sh          # 只补 3.8s
#   INCLUDE_COG158=1 bash run_xmodel_fill_missing.sh   # 额外补 CogVideoX 15.8s(重跑92)
#   MODELS="wan_ti2v cogvideox" bash run_xmodel_fill_missing.sh   # 只补指定模型
#   SMOKE=1 bash run_xmodel_fill_missing.sh            # 每档只 4 条验证
#   FORCE=1 bash run_xmodel_fill_missing.sh            # 忽略已存在, 强制重跑

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/launch_xmodel_dreamgen_infer_kjob.sh"
EVAL_ROOT="${XMODEL_EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/xmodel_eval}"
XMODELS_DIR="${XMODELS_DIR:-/data/datasets/gagi/xmodels}"

# family:权重目录名
DEFAULT_MODELS=("wan_ti2v:wan22_ti2v_5b" "wan:wan22_i2v_a14b" "cogvideox:cogvideox15_5b_i2v")
if [[ -n "${MODELS:-}" ]]; then
  SEL=()
  for m in "${DEFAULT_MODELS[@]}"; do
    fam="${m%%:*}"
    for want in ${MODELS}; do [[ "${fam}" == "${want}" ]] && SEL+=("${m}"); done
  done
  MODEL_LIST=("${SEL[@]}")
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

# 默认补缺档: 3.8s(38) + 7.8s(78)。可用 DURS 覆盖。
DURS="${DURS:-38 78}"
INCLUDE_COG158="${INCLUDE_COG158:-0}"

SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"
DATA_LIMIT_VAL=$([[ "${SMOKE}" == "1" ]] && echo 4 || echo 0)
EXPECT_N=$([[ "${SMOKE}" == "1" ]] && echo 4 || echo 92)
POLL_TIMEOUT="${POLL_TIMEOUT:-14400}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

frames_for() { case "$1" in 38) echo 61;; 58) echo 93;; 78) echo 125;; 98) echo 157;; 158) echo 253;; *) echo 93;; esac; }
dur_label()  { case "$1" in 38) echo 3p8s;; 58) echo 5p8s;; 78) echo 7p8s;; 98) echo 9p8s;; 158) echo 15p8s;; esac; }

# 待跑清单: "family:wdir:dur"
JOBS=()
for entry in "${MODEL_LIST[@]}"; do
  for dur in ${DURS}; do JOBS+=("${entry}:${dur}"); done
done
# 可选: CogVideoX 15.8s 补全(当前 16/92)
if [[ "${INCLUDE_COG158}" == "1" ]]; then
  JOBS+=("cogvideox:cogvideox15_5b_i2v:158")
fi

echo "============================================================"
echo "跨模型 DreamGen 补缺生成 (串行, 8卡/节点)"
echo "  模型:   ${MODEL_LIST[*]}"
echo "  时长档: ${DURS}$([[ "${INCLUDE_COG158}" == "1" ]] && echo " + cogvideox:158")"
echo "  SMOKE=${SMOKE} FORCE=${FORCE} 期望条数=${EXPECT_N}"
echo "  产物根: ${EVAL_ROOT}"
echo "  待提交 job 数: ${#JOBS[@]}"
echo "============================================================"

total=0; done_ok=0; skipped=0; failed_list=()
for job in "${JOBS[@]}"; do
  IFS=':' read -r fam wdir dur <<< "${job}"
  mpath="${XMODELS_DIR}/${wdir}"
  lbl="$(dur_label "${dur}")"
  nf="$(frames_for "${dur}")"
  tag=$([[ "${SMOKE}" == "1" ]] && echo "_smoke" || echo "")
  run_name="${wdir}_${lbl}${tag}"
  save_dir="${EVAL_ROOT}/${run_name}"
  summary="${save_dir}/generation_summary.json"
  total=$((total+1))

  if [[ ! -d "${mpath}" ]]; then
    echo "[skip] ${run_name}: 权重目录不存在 ${mpath}"
    failed_list+=("${run_name}(no-weights)"); continue
  fi

  # 幂等: 已完成则跳过
  if [[ "${FORCE}" != "1" && -f "${summary}" ]]; then
    have=$(ls "${save_dir}"/*.mp4 2>/dev/null | wc -l)
    if [[ "${have}" -ge "${EXPECT_N}" ]]; then
      echo "[have] ${run_name}: 已存在 ${have} 条, 跳过 (FORCE=1 可强制重跑)"
      skipped=$((skipped+1)); continue
    fi
  fi

  echo
  echo "------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] (${total}/${#JOBS[@]}) 提交 ${fam} ${lbl} frames=${nf} -> ${run_name}"
  echo "------------------------------------------------------------"
  rm -f "${summary}"

  MODEL_FAMILY="${fam}" MODEL_PATH="${mpath}" \
  RUN_NAME="${run_name}" DATA_LIMIT="${DATA_LIMIT_VAL}" \
  NUM_FRAMES="${nf}" \
  bash "${LAUNCH}" || echo "[warn] 提交返回非0, 仍尝试等待产物"

  echo "[wait] 等待 ${summary} (每 ${POLL_INTERVAL}s 查一次, 超时 ${POLL_TIMEOUT}s)..."
  waited=0; ok=0
  while [[ ${waited} -lt ${POLL_TIMEOUT} ]]; do
    [[ -f "${summary}" ]] && { ok=1; break; }
    sleep "${POLL_INTERVAL}"; waited=$((waited+POLL_INTERVAL))
    # 心跳: 每次查完都报一行, 让前台能看到"在动"(已生成条数 + 已等时长)
    have=$(ls "${save_dir}"/*.mp4 2>/dev/null | wc -l)
    logmt=""
    [[ -f "${save_dir}/run.log" ]] && logmt=" | log更新 $(stat -c '%y' "${save_dir}/run.log" 2>/dev/null | cut -d. -f1 | cut -d' ' -f2)"
    echo "  [$(date +%H:%M:%S)] ${run_name}: ${have}/${EXPECT_N} 条, 已等 $((waited/60))m${logmt}"
  done

  if [[ ${ok} -eq 1 ]]; then
    got=$(grep -oE '"ok"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
    tot=$(grep -oE '"total"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
    echo "[done] ${run_name}: ok=${got:-?}/${tot:-?}  ($(date +%H:%M:%S))"
    done_ok=$((done_ok+1))
  else
    echo "[TIMEOUT] ${run_name} 超时未见 summary。检查 ${save_dir}/run.log"
    failed_list+=("${run_name}")
  fi
done

echo
echo "============================================================"
echo "结束: 完成 ${done_ok} / 提交 $((total-skipped)) (跳过已存在 ${skipped})"
[[ ${#failed_list[@]} -gt 0 ]] && printf "  未完成: %s\n" "${failed_list[@]}"
echo "产物在: ${EVAL_ROOT}/<模型>_<时长>/"
echo "  完成后可对这些目录跑 eveworld/evaluation/tea/qwen_laziness.py 与 eveworld/evaluation/tea/ncm.py (generated-only, 不用 crop)"
echo "============================================================"
