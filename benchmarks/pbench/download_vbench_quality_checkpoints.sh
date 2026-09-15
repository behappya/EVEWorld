#!/usr/bin/env bash
set -euo pipefail

# Download checkpoints needed by benchmarks/pbench/eval_pbench_robot_vbench_quality.sh.
# The script is resumable: rerun it after interruption.

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/datasets/gagi/giga_world_0_outputs/pbench_robot_vbench_quality}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/data/datasets/gagi/vbench_cache}"
DREAMSIM_CACHE_DIR="${DREAMSIM_CACHE_DIR:-${OUTPUT_ROOT}/vbench_work/models}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_models}"
HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
HF_XET_CACHE="${HF_XET_CACHE:-/data/datasets/gagi/.hf_xet_cache}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HOME HF_XET_CACHE HF_HUB_ENABLE_HF_TRANSFER

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

download_to() {
  local url="$1"
  local path="$2"
  mkdir -p "$(dirname "${path}")"
  if [[ -f "${path}.complete" && -s "${path}" ]]; then
    echo "exists: ${path}"
    return
  fi
  echo "download: ${url}"
  echo "     to: ${path}"
  wget -c -nv -O "${path}" "${url}"
  touch "${path}.complete"
}

download_hf_to() {
  local repo_id="$1"
  local filename="$2"
  local path="$3"
  mkdir -p "$(dirname "${path}")" "${HF_HOME}" "${HF_XET_CACHE}"
  if [[ -f "${path}.complete" && -s "${path}" ]]; then
    echo "exists: ${path}"
    return
  fi
  echo "hf download: ${repo_id}/${filename}"
  echo "        to: ${path}"
  python -m huggingface_hub.cli.hf download \
    "${repo_id}" \
    "${filename}" \
    --repo-type model \
    --local-dir "$(dirname "${path}")" \
    --cache-dir "${HF_HOME}/hub" \
    --max-workers 1
  if [[ ! -s "${path}" ]]; then
    echo "HF download finished but target file is missing or empty: ${path}" >&2
    exit 1
  fi
  touch "${path}.complete"
}

require_cmd wget
require_cmd unzip
require_cmd git
require_cmd python

mkdir -p "${VBENCH_CACHE_DIR}" "${DREAMSIM_CACHE_DIR}"

echo "============================================"
echo "PBench Robot VBench checkpoint downloader"
echo "VBench cache:   ${VBENCH_CACHE_DIR}"
echo "DreamSim cache: ${DREAMSIM_CACHE_DIR}"
echo "HF_HOME:        ${HF_HOME}"
echo "HF_XET_CACHE:   ${HF_XET_CACHE}"
echo "============================================"

echo "==> DINO repository and checkpoint"
DINO_REPO="${VBENCH_CACHE_DIR}/dino_model/facebookresearch_dino_main"
if [[ -d "${DINO_REPO}/.git" ]]; then
  echo "exists: ${DINO_REPO}"
else
  mkdir -p "$(dirname "${DINO_REPO}")"
  git clone --depth 1 https://github.com/facebookresearch/dino "${DINO_REPO}"
fi
download_to \
  "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" \
  "${VBENCH_CACHE_DIR}/dino_model/dino_vitbase16_pretrain.pth"

echo "==> DreamSim torch.hub DINO cache"
DREAMSIM_DINO_REPO="${DREAMSIM_CACHE_DIR}/facebookresearch_dino_main"
if [[ -d "${DREAMSIM_DINO_REPO}/.git" || -f "${DREAMSIM_DINO_REPO}/hubconf.py" ]]; then
  echo "exists: ${DREAMSIM_DINO_REPO}"
else
  mkdir -p "${DREAMSIM_CACHE_DIR}"
  git clone --depth 1 https://github.com/facebookresearch/dino "${DREAMSIM_DINO_REPO}"
fi
mkdir -p "${DREAMSIM_CACHE_DIR}/checkpoints"
ln -sfn "${VBENCH_CACHE_DIR}/dino_model/dino_vitbase16_pretrain.pth" \
  "${DREAMSIM_CACHE_DIR}/checkpoints/dino_vitbase16_pretrain.pth"

echo "==> CLIP checkpoints used by VBench local mode"
download_to \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
  "${VBENCH_CACHE_DIR}/clip_model/ViT-B-32.pt"
download_to \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
  "${VBENCH_CACHE_DIR}/clip_model/ViT-L-14.pt"

echo "==> Aesthetic predictor"
download_to \
  "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true" \
  "${VBENCH_CACHE_DIR}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"

echo "==> MUSIQ image quality checkpoint"
download_to \
  "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
  "${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"

echo "==> AMT motion smoothness checkpoint"
download_hf_to \
  "lalala125/AMT" \
  "amt-s.pth" \
  "${VBENCH_CACHE_DIR}/amt_model/amt-s.pth"

echo "==> ViCLIP overall consistency checkpoint and tokenizer"
download_hf_to \
  "OpenGVLab/VBench_Used_Models" \
  "ViClip-InternVid-10M-FLT.pth" \
  "${VBENCH_CACHE_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth"
download_to \
  "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz" \
  "${VBENCH_CACHE_DIR}/ViCLIP/bpe_simple_vocab_16e6.txt.gz"

echo "==> DreamSim ensemble checkpoint"
DREAMSIM_ZIP="${DREAMSIM_CACHE_DIR}/pretrained.zip"
download_to \
  "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip" \
  "${DREAMSIM_ZIP}"
unzip -n "${DREAMSIM_ZIP}" -d "${DREAMSIM_CACHE_DIR}"

echo "============================================"
echo "Downloaded checkpoint tree:"
du -sh "${VBENCH_CACHE_DIR}" "${DREAMSIM_CACHE_DIR}" 2>/dev/null || true
echo "Done."
echo "============================================"
