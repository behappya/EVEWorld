#!/usr/bin/env bash
# GW-0 域内六档在同一判官实例(@12)重判 eval175,消除跨 endpoint 不可比问题。
# 对照组核心:gr1_2b(vanilla SFT) vs t4g_wmapA_pre_seed42_s250(我们的方法)。
set -uo pipefail

source /home/jovyan/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-EVEWorld}"

GAGI=/data/datasets/gagi
REPO=giga-world-0
OUT_ROOT=${GAGI}/eve_v2_outputs/eval175_eval_indomain_rejudge
EVAL_OUT=${GAGI}/gr1_dreamgen_eval/eval_outputs
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
CONCURRENCY="${CONCURRENCY:-50}"
MODELS="pretrain,gr1_2b,round0,t4g_wmapA_pre_seed42_s250,t4g_wmapA_pre_noaug_s50,t4g_wmaponly_s150"

cd ${REPO}/eveworld/evaluation
echo "===== 1. prepare(生成已存在, 仅审计+出 manifest) ====="
python eval175_prepare.py \
  --output-root "${OUT_ROOT}" \
  --models "${MODELS}" \
  --no-stage
echo "prepare rc=$? (报警不阻塞, manifest 为准)"

echo "===== 2. 六模型并行判分 (每模型 ${CONCURRENCY} 并发) ====="
cd ${REPO}
pids=()
for m in ${MODELS//,/ }; do
  python benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.py \
    --manifest "${OUT_ROOT}/manifests/${m}.jsonl" \
    --output-root "${EVAL_OUT}" \
    --run-name "eval175_${m}_qwen36_12" \
    --qwen-base "${QWEN_BASE}" \
    --qwen-model Qwen/Qwen3.6-35B-A3B \
    --metrics qwen_if,pa_i \
    --concurrency "${CONCURRENCY}" --max-inflight "${CONCURRENCY}" \
    > "${EVAL_OUT}/eval175_${m}_qwen36_12_judge.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "${p}" || fail=$((fail+1)); done

echo "===== 3. 汇总 (failed=${fail}) ====="
for m in ${MODELS//,/ }; do
  s="${EVAL_OUT}/eval175_${m}_qwen36_12_summary.json"
  if [ -f "$s" ]; then
    python3 -c "
import json; d=json.load(open('$s'))
r=[f'{k}={v[\"positive\"]}/{v[\"count\"]}({v[\"score\"]:.3f})err{v[\"error_count\"]}' for k,v in d.items() if isinstance(v,dict)]
print('  ${m}:', ' '.join(r))"
  else
    echo "  ${m}: 无 summary"
  fi
done
