#!/usr/bin/env bash
set -euo pipefail

# Prepare generated-only PBench Robot videos and run the VBench quality metrics.
# This loads VBench metric models locally and may download checkpoints into
# VBENCH_CACHE_DIR. Run it on a machine/job with enough GPU/RAM for VBench.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality}"
METADATA_JSONL="${METADATA_JSONL:-/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.metadata.jsonl}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_serving_full_20260612_145541}"
DOMAIN_SUMMARY="${DOMAIN_SUMMARY:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_qwen_vqa_eval/20260613_131107_qwen36vl/qwen_vqa_summary.json}"
RESOLUTION_NAME="${RESOLUTION_NAME:-pbench_robot}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/data/datasets/gagi/vbench_cache}"

LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
CPU="${CPU:-0}"
LOCAL_CKPT="${LOCAL_CKPT:-1}"
DIMENSIONS="${DIMENSIONS:-i2v_subject i2v_background aesthetic_quality imaging_quality background_consistency motion_smoothness subject_consistency overall_consistency}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
REQUIRE_CHECKPOINTS="${REQUIRE_CHECKPOINTS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${REPO_DIR}/scripts/vbench_compat:${PYTHONPATH:-}"
export VBENCH_CACHE_DIR
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/datasets/gagi/.cache}"
export TORCH_HOME="${TORCH_HOME:-/data/datasets/gagi/torch_cache}"
export HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}" "${VBENCH_CACHE_DIR}"
RUN_LOG="${RUN_LOG:-${OUTPUT_ROOT}/quality_eval_$(date +%Y%m%d_%H%M%S).log}"

echo "============================================" | tee -a "${RUN_LOG}"
echo "PBench Robot VBench quality launcher" | tee -a "${RUN_LOG}"
echo "Repo:               ${REPO_DIR}" | tee -a "${RUN_LOG}"
echo "Output root:        ${OUTPUT_ROOT}" | tee -a "${RUN_LOG}"
echo "Metadata:           ${METADATA_JSONL}" | tee -a "${RUN_LOG}"
echo "Source videos:      ${SOURCE_VIDEO_DIR}" | tee -a "${RUN_LOG}"
echo "Domain summary:     ${DOMAIN_SUMMARY}" | tee -a "${RUN_LOG}"
echo "VBench cache:       ${VBENCH_CACHE_DIR}" | tee -a "${RUN_LOG}"
echo "Torch cache:        ${TORCH_HOME}" | tee -a "${RUN_LOG}"
echo "HF home:            ${HF_HOME}" | tee -a "${RUN_LOG}"
echo "Resolution name:    ${RESOLUTION_NAME}" | tee -a "${RUN_LOG}"
echo "Limit:              ${LIMIT}" | tee -a "${RUN_LOG}"
echo "Local checkpoints:  ${LOCAL_CKPT}" | tee -a "${RUN_LOG}"
echo "Require ckpts:      ${REQUIRE_CHECKPOINTS}" | tee -a "${RUN_LOG}"
echo "Skip existing:      ${SKIP_EXISTING}" | tee -a "${RUN_LOG}"
echo "Dimensions:         ${DIMENSIONS}" | tee -a "${RUN_LOG}"
echo "Precheck only:      ${PRECHECK_ONLY}" | tee -a "${RUN_LOG}"
echo "Run log:            ${RUN_LOG}" | tee -a "${RUN_LOG}"
echo "============================================" | tee -a "${RUN_LOG}"

echo "==> Checking required Python imports" | tee -a "${RUN_LOG}"
python - <<'PY' 2>&1 | tee -a "${RUN_LOG}"
missing = []
for name in ["vbench", "clip", "pyiqa", "decord", "dreamsim", "open_clip", "skvideo", "torchmetrics"]:
    try:
        __import__(name)
        print(f"{name}: OK")
    except Exception as exc:
        missing.append((name, exc))
        print(f"{name}: MISSING {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit(1)
PY

PREPARE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  PREPARE_ARGS+=(--overwrite)
fi

echo "==> Preparing generated-only videos and VBench metadata" | tee -a "${RUN_LOG}"
python benchmarks/pbench/prepare_pbench_vbench_inputs.py \
  --metadata-jsonl "${METADATA_JSONL}" \
  --source-video-dir "${SOURCE_VIDEO_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --resolution-name "${RESOLUTION_NAME}" \
  --limit "${LIMIT}" \
  "${PREPARE_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "==> PRECHECK_ONLY=1, skipping VBench metric computation and checkpoint downloads" | tee -a "${RUN_LOG}"
  exit 0
fi

if [[ "${CPU}" != "1" ]]; then
  echo "==> Checking CUDA availability for local VBench metrics" | tee -a "${RUN_LOG}"
  if ! python - <<'PY' 2>&1 | tee -a "${RUN_LOG}"
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(1)
PY
  then
    cat <<'EOF' | tee -a "${RUN_LOG}"
CUDA is not available on this host.
Do not run VBench quality locally on the workspace host. Submit it to a GPU kjob with:
  ./benchmarks/pbench/launch_pbench_robot_vbench_quality_kjob.sh

For a slow CPU-only debug run, set CPU=1 explicitly.
EOF
    exit 3
  fi
fi

if [[ "${REQUIRE_CHECKPOINTS}" == "1" ]]; then
  echo "==> Checking local checkpoints before VBench metric computation" | tee -a "${RUN_LOG}"
  missing=0

  require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
      echo "missing: ${path}" | tee -a "${RUN_LOG}"
      missing=1
    fi
  }

  if [[ " ${DIMENSIONS} " == *" i2v_subject "* || " ${DIMENSIONS} " == *" subject_consistency "* ]]; then
    require_path "${VBENCH_CACHE_DIR}/dino_model/facebookresearch_dino_main/hubconf.py"
    require_path "${VBENCH_CACHE_DIR}/dino_model/dino_vitbase16_pretrain.pth"
  fi
  if [[ " ${DIMENSIONS} " == *" i2v_background "* ]]; then
    require_path "${OUTPUT_ROOT}/vbench_work/models/facebookresearch_dino_main/hubconf.py"
    require_path "${OUTPUT_ROOT}/vbench_work/models/checkpoints/dino_vitbase16_pretrain.pth"
    require_path "${OUTPUT_ROOT}/vbench_work/models/dino_vitb16_pretrain.pth"
    require_path "${OUTPUT_ROOT}/vbench_work/models/open_clip_vitb16_pretrain.pth.tar"
    require_path "${OUTPUT_ROOT}/vbench_work/models/clip_vitb16_pretrain.pth.tar"
    require_path "${OUTPUT_ROOT}/vbench_work/models/ensemble_lora/adapter_config.json"
  fi
  if [[ " ${DIMENSIONS} " == *" background_consistency "* ]]; then
    require_path "${VBENCH_CACHE_DIR}/clip_model/ViT-B-32.pt"
  fi
  if [[ " ${DIMENSIONS} " == *" aesthetic_quality "* ]]; then
    require_path "${VBENCH_CACHE_DIR}/clip_model/ViT-L-14.pt"
    require_path "${VBENCH_CACHE_DIR}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
  fi
  if [[ " ${DIMENSIONS} " == *" imaging_quality "* ]]; then
    require_path "${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
  fi
  if [[ " ${DIMENSIONS} " == *" motion_smoothness "* ]]; then
    require_path "${VBENCH_CACHE_DIR}/amt_model/amt-s.pth"
  fi
  if [[ " ${DIMENSIONS} " == *" overall_consistency "* ]]; then
    require_path "${VBENCH_CACHE_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth"
    require_path "${VBENCH_CACHE_DIR}/ViCLIP/bpe_simple_vocab_16e6.txt.gz"
  fi

  if [[ "${missing}" != "0" ]]; then
    echo "Checkpoint check failed. Run ./benchmarks/pbench/download_vbench_quality_checkpoints.sh, then rerun this script." | tee -a "${RUN_LOG}"
    exit 2
  fi
fi

EVAL_ARGS=()
if [[ "${CPU}" == "1" ]]; then
  EVAL_ARGS+=(--cpu)
fi
if [[ "${LOCAL_CKPT}" == "1" ]]; then
  EVAL_ARGS+=(--local)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EVAL_ARGS+=(--skip-existing)
else
  EVAL_ARGS+=(--no-skip-existing)
fi

echo "==> Running VBench quality metrics" | tee -a "${RUN_LOG}"
# shellcheck disable=SC2086
python benchmarks/pbench/eval_pbench_robot_vbench_quality.py \
  --output-root "${OUTPUT_ROOT}" \
  --domain-summary "${DOMAIN_SUMMARY}" \
  --resolution-name "${RESOLUTION_NAME}" \
  --dimensions ${DIMENSIONS} \
  "${EVAL_ARGS[@]}" 2>&1 | tee -a "${RUN_LOG}"
