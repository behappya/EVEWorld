#!/usr/bin/env bash
set -euo pipefail
# EVE · EAG 采样生成(集群 kjob, 单卡)。
# 默认一键配对提交: baseline(w=0) + EAG(w=0.03), 同 seed, 便于"EAG 是否降偷懒"对比。
#
# 用法:
#   bash eveworld/method/scripts/launch_eag_generate_kjob.sh              # 配对 baseline+EAG
#   EAG_WEIGHT=0.05 PAIR=0 bash ...                                  # 只跑单个 w
#   LIMIT=8 bash ...                                                 # 冒烟(前8条)
#   DRY_RUN=1 bash ...                                               # 只打印

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = giga-world-0
export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_eve_eag_generate.sh}"

GAGI="${GAGI_ROOT:-/data/datasets/gagi}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
EAG_ROOT="${EAG_ROOT:-${GAGI}/eve_outputs/eag_eval}"
SEED="${SEED:-6666}"; NUM_FRAMES="${NUM_FRAMES:-93}"; LIMIT="${LIMIT:-0}"
PAIR="${PAIR:-1}"                       # 1=配对提交 baseline+EAG
EAG_WEIGHT="${EAG_WEIGHT:-0.03}"

submit_one() {
  local w="$1" tag="$2"
  local save="${EAG_ROOT}/${tag}_seed${SEED}_f${NUM_FRAMES}"
  echo "----- 提交 ${tag} (eag_weight=${w}) -> ${save} -----"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] submit: EAG_WEIGHT=${w} SAVE_DIR=${save} LAM=${LAM} DATA_PATH=${DATA_PATH} NUM_FRAMES=${NUM_FRAMES} SEED=${SEED} LIMIT=${LIMIT}"
    return 0
  fi
  # 后台提交: submit 脚本结尾 exec kjobctl 会替换/阻塞进程, 若前台串行则第一个会卡住
  # 导致配对的第二个永远提交不了(实测 bug)。用 () & 隔离子 shell 后台提交, 各自独立。
  ( "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
      "REPO_DIR=${REPO_DIR}" "GAGI_ROOT=${GAGI}" \
      "DATA_PATH=${DATA_PATH}" "LAM=${LAM}" "SAVE_DIR=${save}" \
      "EAG_WEIGHT=${w}" "NUM_FRAMES=${NUM_FRAMES}" "SEED=${SEED}" "LIMIT=${LIMIT}" ) &
  sleep 8   # 错开两次 kjobctl 提交, 避免竞争
}

if [[ ! -f "${LAM}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "[ERR] LAD checkpoint 不存在: ${LAM} (先跑 launch_lam_pretrain_kjob.sh)" >&2
  exit 1
fi

echo "[EVE] EAG 生成. lam=${LAM} seed=${SEED} frames=${NUM_FRAMES} limit=${LIMIT} pair=${PAIR}"
if [[ "${PAIR}" == "1" ]]; then
  submit_one 0 baseline
  submit_one "${EAG_WEIGHT}" "eag_w${EAG_WEIGHT}"
else
  submit_one "${EAG_WEIGHT}" "eag_w${EAG_WEIGHT}"
fi
wait   # 等所有后台提交完成再退出(否则 launcher 提前退出会杀掉未提交完的子 shell)
echo "[EVE] 全部提交完成。生成完对 <save>/generated_only 跑: eveworld/evaluation/tea/ncm.py + eveworld/evaluation/tea/qwen_laziness.py"
