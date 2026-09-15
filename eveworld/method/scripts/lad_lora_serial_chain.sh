#!/usr/bin/env bash
# EVE · LAD-LoRA 消融【单节点串行】编排器。
# 按队列依次提交 W_LAD 消融, 每个 job 跑完(检测 kjob pod 消失)才发下一个 -> 任意时刻只占 1 节点。
# 用户要求: 不并行多节点, 一个节点串起来跑。
# 后台运行: nohup bash eveworld/method/scripts/lad_lora_serial_chain.sh > /tmp/lad_chain.log 2>&1 &
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

WEIGHTS="${WEIGHTS:-0.1 0.0 0.2}"     # 队列顺序: 主设置 -> control -> 强正则
SEED="${SEED:-6666}"
POLL="${POLL:-120}"                    # 轮询间隔秒
LAUNCH="eveworld/method/scripts/launch_lad_lora_train_kjob.sh"

# 只轮询【本链条提交的那个具体 job】, 不影响 best-of-N 等其它节点上的作业。
# job 名从提交输出 "job.batch/slurm-profile-slurm-XXXXX created" 捕获。
job_alive() {  # $1 = job 名(slurm-profile-slurm-XXXXX)
  timeout 40 kubectl get pods 2>/dev/null | grep -E "$1" | grep -qiE "Running|Pending|ContainerCreating" && return 0 || return 1
}

for w in ${WEIGHTS}; do
  echo "======================================================"
  echo "[chain] $(date '+%F %T') 提交 W_LAD=${w} SEED=${SEED}"
  echo "======================================================"
  RUN_NAME="$(date +%Y%m%d_%H%M%S)_eve_lad_lora_w${w}_seed${SEED}"
  OUT="$(bash "${LAUNCH}" "W_LAD=${w}" "SEED=${SEED}" "RUN_NAME=${RUN_NAME}" 2>&1)"
  echo "${OUT}" | tail -3
  JOB="$(echo "${OUT}" | grep -oE "slurm-profile-slurm-[a-z0-9]+" | head -1)"
  if [[ -z "${JOB}" ]]; then
    echo "[chain] !! 未捕获到 job 名, 提交可能失败, 中止链条。" >&2
    exit 1
  fi
  echo "[chain] 已提交 ${RUN_NAME} -> job ${JOB}, 等待启动..."
  sleep 60   # 给 kjob pod 拉起时间

  # 等这个具体 job 结束(pod 消失)
  while job_alive "${JOB}"; do
    sleep "${POLL}"
  done
  RUN_LOG="/data/datasets/gagi/giga_world_0_outputs/eve/${RUN_NAME}/run.log"
  echo "[chain] $(date '+%T') W_LAD=${w} (job ${JOB}) 结束。run.log 尾部:"
  tail -4 "${RUN_LOG}" 2>/dev/null || echo "  (无 run.log)"
  # 校验完成标记
  if grep -q "Step\[150/150\]" "${RUN_LOG}" 2>/dev/null; then
    echo "[chain] ✓ W_LAD=${w} 训练完成(Step[150/150])"
  else
    echo "[chain] ⚠ W_LAD=${w} 未见完成标记, 可能中途失败, 继续下一个(请人工查 ${RUN_LOG})"
  fi
done

echo "[chain] $(date '+%F %T') 全部 ${WEIGHTS} 串行完成。"
echo "[chain] 下一步: 对各 run 的 checkpoint 跑 generate_eag.py --lora <ckpt> --eag-weight 0 生成 -> Qwen 独立裁判评偷懒 + 画质。"
