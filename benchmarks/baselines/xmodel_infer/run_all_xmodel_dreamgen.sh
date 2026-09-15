#!/usr/bin/env bash
set -uo pipefail

# 一键顺序跑：已下载的对比模型 × DreamGen × {5.8s, 9.8s, 15.8s}。
# 每个 (模型,时长) 提交一个 GPU kjob，等它写出 generation_summary.json（=跑完）后再提交下一个。
# 只需执行这一个脚本。Cosmos 未下载，默认不含。
#
# 用法:
#   bash run_all_xmodel_dreamgen.sh              # 全 92 条 × 3 档 × 3 模型 (9 个 job 顺序跑)
#   SMOKE=1 bash run_all_xmodel_dreamgen.sh      # 每个只跑 4 条 (快速验证 9 个组合能否出片)
#   DURS="58" bash run_all_xmodel_dreamgen.sh    # 只跑 5.8s 档
#   MODELS="wan_ti2v cogvideox" bash run_all_xmodel_dreamgen.sh   # 只跑指定模型
#
# 时长档->帧数: 各模型不同(CogVideoX 需 16k+1)。见下 frames_for()。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/launch_xmodel_dreamgen_infer_kjob.sh"
EVAL_ROOT="${XMODEL_EVAL_ROOT:-/data/datasets/gagi/gr1_dreamgen_eval/xmodel_eval}"

XMODELS_DIR="${XMODELS_DIR:-/data/datasets/gagi/xmodels}"

# 模型清单: "family:权重目录名"。Cosmos 未下, 不列(下好后加 "cosmos:cosmos_predict25_2b")。
DEFAULT_MODELS=("wan_ti2v:wan22_ti2v_5b" "wan:wan22_i2v_a14b" "cogvideox:cogvideox15_5b_i2v")

# 允许用 MODELS="wan_ti2v cogvideox" 只选 family 子集
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

# 时长档 (可用 DURS="58 98" 覆盖)
DURS="${DURS:-58 98 158}"

SMOKE="${SMOKE:-0}"
DATA_LIMIT_VAL=$([[ "${SMOKE}" == "1" ]] && echo 4 || echo 0)
POLL_TIMEOUT="${POLL_TIMEOUT:-14400}"   # 单个 job 最长等待秒数(默认4h)
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# 各 family 时长档帧数: 统一 93/157/253。
# 三者对 CogVideoX 也合法: latent_frames=(nf-1)//4+1 须为偶数(patch_size_t=2),
# 93/157/253 -> 24/40/64 偶✓; 而 97/161/257(16k+1) latent 为奇数会报错, 不能用。
frames_for() {
  local fam="$1" dur="$2"
  case "${dur}" in 58) echo 93;; 98) echo 157;; 158) echo 253;; esac
}
dur_label() { case "$1" in 58) echo 5p8s;; 98) echo 9p8s;; 158) echo 15p8s;; esac; }

echo "============================================================"
echo "一键顺序跑 xmodel DreamGen"
echo "  模型: ${MODEL_LIST[*]}"
echo "  时长档: ${DURS}   SMOKE=${SMOKE} (DATA_LIMIT=${DATA_LIMIT_VAL})"
echo "  产物根: ${EVAL_ROOT}"
echo "============================================================"

total=0; done_ok=0; failed_list=()
for entry in "${MODEL_LIST[@]}"; do
  fam="${entry%%:*}"; wdir="${entry##*:}"
  mpath="${XMODELS_DIR}/${wdir}"
  if [[ ! -d "${mpath}" ]]; then
    echo "[skip] ${fam}: 权重目录不存在 ${mpath}"
    continue
  fi
  for dur in ${DURS}; do
    nf="$(frames_for "${fam}" "${dur}")"
    lbl="$(dur_label "${dur}")"
    tag=$([[ "${SMOKE}" == "1" ]] && echo "_smoke" || echo "")
    run_name="${wdir}_${lbl}${tag}"
    save_dir="${EVAL_ROOT}/${run_name}"
    summary="${save_dir}/generation_summary.json"
    total=$((total+1))

    echo
    echo "------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] (${total}) 提交 ${fam} ${lbl} frames=${nf} -> ${run_name}"
    echo "------------------------------------------------------------"
    # 删旧 summary, 避免复用旧文件误判完成
    rm -f "${summary}"

    MODEL_FAMILY="${fam}" MODEL_PATH="${mpath}" \
    RUN_NAME="${run_name}" DATA_LIMIT="${DATA_LIMIT_VAL}" \
    NUM_FRAMES="${nf}" \
    bash "${LAUNCH}" || { echo "[warn] 提交返回非0, 仍尝试等待产物"; }

    # 轮询等待完成(summary 出现)
    echo "[wait] 等待 ${summary} 出现 (每 ${POLL_INTERVAL}s 查一次, 超时 ${POLL_TIMEOUT}s)..."
    waited=0; ok=0
    while [[ ${waited} -lt ${POLL_TIMEOUT} ]]; do
      if [[ -f "${summary}" ]]; then ok=1; break; fi
      sleep "${POLL_INTERVAL}"; waited=$((waited+POLL_INTERVAL))
    done

    if [[ ${ok} -eq 1 ]]; then
      got=$(grep -oE '"ok"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
      tot=$(grep -oE '"total"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
      echo "[done] ${run_name}: ok=${got}/${tot}  ($(date +%H:%M:%S))"
      done_ok=$((done_ok+1))
    else
      echo "[TIMEOUT] ${run_name} 超时未见 summary, 跳到下一个。检查 ${save_dir}/run.log"
      failed_list+=("${run_name}")
    fi
  done
done

echo
echo "============================================================"
echo "全部结束: ${done_ok}/${total} 完成"
[[ ${#failed_list[@]} -gt 0 ]] && printf "  未完成/超时: %s\n" "${failed_list[@]}"
echo "产物在: ${EVAL_ROOT}/<模型>_<时长>${tag}/"
echo "============================================================"
