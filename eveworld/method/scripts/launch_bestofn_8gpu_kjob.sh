#!/usr/bin/env bash
# EVE · best-of-N 生成 —— 提交【单节点 8 卡】1 个 kjob(取代原碎片化的 8 个单卡 job)。
# 8 个 seed 并行占满 1 个节点的 8 张卡, 各卡串行跑 92 条; 默认补齐模式(SKIP_EXISTING=1)。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = giga-world-0

export REPO_DIR
export RL_DIR="${RL_DIR:-/home/jovyan/new_rl/rl}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_bestofn_8gpu.sh}"

# 透传给 payload 的覆盖项(submit_gigaworld0_kjob.sh 会 maybe_export 这些名字)
export SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
export NUM_FRAMES="${NUM_FRAMES:-93}"
export LIMIT="${LIMIT:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export EAG_WEIGHT="${EAG_WEIGHT:-0}"
export SEED="${SEED:-6666}"   # 占位, payload 用 SEEDS 分发

echo "[EVE] 提交 best-of-N 单节点8卡 job"
echo "  seeds=[${SEEDS}] frames=${NUM_FRAMES} limit=${LIMIT} skip_existing=${SKIP_EXISTING} eag_w=${EAG_WEIGHT}"
echo "  前置: 先自行 kill 掉旧的 8 个单卡 job(否则重复占卡)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] would submit ${JOB_SCRIPT} with SEEDS='${SEEDS}' LIMIT=${LIMIT} SKIP_EXISTING=${SKIP_EXISTING}"
  exit 0
fi

# submit_gigaworld0_kjob.sh 需要 SEEDS/SKIP_EXISTING/EAG_WEIGHT/OUT_ROOT 等在其 maybe_export 白名单;
# 不在白名单的用 EXTRA_ARGS 直接以 KEY=VALUE 追加透传给 payload。
EXTRA_ARGS=("SEEDS=${SEEDS}" "SKIP_EXISTING=${SKIP_EXISTING}" "EAG_WEIGHT=${EAG_WEIGHT}"
            "NUM_FRAMES=${NUM_FRAMES}" "LIMIT=${LIMIT}")
exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" "${EXTRA_ARGS[@]}"
