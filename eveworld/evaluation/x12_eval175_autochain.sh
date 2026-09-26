#!/usr/bin/env bash
# X12: eval175 生成完成轮询 -> manifest -> qwen3.6@55 判分 (三臂: A-s50/A-s100/pretrain)
set -uo pipefail
GAGI=/data/datasets/gagi
GEN=$GAGI/eve_v2_outputs/eval175_gen
ARMS="t4g_wmapA_s50 t4g_wmapA_s100 pretrain"
declare -A TGT=( [gr1_env]=29 [gr1_object]=50 [gr1_behavior]=47 )

echo "[chain] 轮询生成完成 (每3分钟)..."
while true; do
  done_all=1; stat=""
  for arm in $ARMS; do
    for sp in gr1_env gr1_object gr1_behavior; do
      n=$(ls "$GEN/$arm/$sp/generated_only/"*.mp4 2>/dev/null | wc -l)
      stat+="$arm/$sp:$n "
      [[ "$n" -lt "${TGT[$sp]}" ]] && done_all=0
    done
  done
  echo "[chain] $(date +%H:%M) $stat"
  [[ "$done_all" == 1 ]] && break
  sleep 180
done
echo "[chain] 全部生成完成, 建 manifest..."

python3 - <<'EOF'
import json, os
GAGI='/data/datasets/gagi'
GEN=f'{GAGI}/eve_v2_outputs/eval175_gen'
INP=f'{GAGI}/gr1_dreamgen_eval/giga_input'
OUT=f'{GAGI}/gr1_dreamgen_eval/eval_manifests'
for arm in ['t4g_wmapA_s50','t4g_wmapA_s100','pretrain']:
    rows=[]
    for sp in ['gr1_env','gr1_object','gr1_behavior']:
        prompts=[it['prompt'] for it in json.load(open(f'{INP}/eval175_{sp}.json'))]
        vdir=f'{GEN}/{arm}/{sp}/generated_only'
        vids={int(f.split('_',1)[0]): f for f in os.listdir(vdir) if f.endswith('.mp4')}
        for i,p in enumerate(prompts):
            if i in vids:
                rows.append({'key':f'{sp}/{i}','video_path':f'{vdir}/{vids[i]}','prompt':p})
    fp=f'{OUT}/eval175_{arm}.jsonl'
    with open(fp,'w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'{arm}: {len(rows)} -> {fp}')
EOF

source /home/jovyan/miniconda/etc/profile.d/conda.sh; conda activate "${CONDA_ENV:-EVEWorld}"
cd giga-world-0
for arm in $ARMS; do
  echo "[chain] 判分 $arm ..."
  python benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.py \
    --manifest $GAGI/gr1_dreamgen_eval/eval_manifests/eval175_${arm}.jsonl \
    --qwen-base http://127.0.0.1:8000/v1 --metrics qwen_if,pa_i \
    --concurrency 400 --max-inflight 400 --model-timeout 1200 \
    --run-name eval175_${arm}_qwen36_55 2>&1 | tail -3
done
echo "[chain] 全部判分完成"
