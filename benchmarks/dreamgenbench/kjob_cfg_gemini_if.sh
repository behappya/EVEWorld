#!/usr/bin/env bash
# Gemini-IF evaluation for the 24-cell CFG sweep.
# The job requests one GPU only to use the standard kjob queue; inference is API/CPU bound.
#SBATCH --job-name=cfg_gemini_if
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -euo pipefail

for arg in "$@"; do [[ "$arg" == *=* ]] || { echo "bad arg: $arg" >&2; exit 2; }; export "$arg"; done
[[ "$(hostname)" == coder-workspace-* && "${ALLOW_LOCAL_RUN:-0}" != 1 ]] && exit 2

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
MANIFEST="${MANIFEST:-/data/datasets/gagi/eve_v2_outputs/cfg_grid_seed004/gemini_if_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/gagi/eve_v2_outputs/gemini_eval/cfg_grid_seed004_gemini36_repeat01}"
MODEL="${MODEL:-gemini-3.6-flash}"
CONCURRENCY="${CONCURRENCY:-100}"

source /home/jovyan/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-EVEWorld}"
cd "$REPO_DIR"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/eveworld/evaluation:${EVEWORLD_ROOT}/benchmarks/dreamgenbench:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

[[ -s "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }
exec python eveworld/evaluation/eval_gemini_dreamgen_qwen_protocol.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --metrics qwen_if \
  --concurrency "$CONCURRENCY" \
  --frame-count 49 \
  --jpeg-quality 85 \
  --temperature 0 \
  --thinking-level low \
  --no-include-thoughts \
  --model-timeout 1200 \
  --model-max-tokens 32000 \
  --resume
