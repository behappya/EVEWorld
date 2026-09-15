#!/usr/bin/env bash
set -euo pipefail

# Resumable downloader for the GigaWorld-0 video_gr1 checkpoint.
# Re-run this script if the download is interrupted.
#
# By default this only downloads the GR1 transformer, then reuses the
# text_encoder and VAE from the existing video_pretrain directory via symlinks.
# If those shared components are missing, it downloads them from Hugging Face.

ROOT_DIR="${ROOT_DIR:-/data/datasets/gagi}"
BASE_DIR="${BASE_DIR:-${ROOT_DIR}/giga_world_0_video_gr1}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${ROOT_DIR}/giga_world_0_video_pretrain}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_WORKERS="${MAX_WORKERS:-8}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
REUSE_PRETRAIN_COMPONENTS="${REUSE_PRETRAIN_COMPONENTS:-1}"

GIGAWORLD_REPO="open-gigaai/GigaWorld-0-Video-GR1-2b"
TEXT_ENCODER_REPO="google-t5/t5-11b"
VAE_REPO="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-${ROOT_DIR}/.hf_xet_cache}"

ensure_deps() {
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import huggingface_hub
import hf_xet
PY
  then
    return
  fi

  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "Missing huggingface_hub and/or hf_xet. Installing into current Python environment..."
    "${PYTHON_BIN}" -m pip install -U huggingface_hub hf_xet
  else
    echo "Missing dependency. Install it with:" >&2
    echo "  ${PYTHON_BIN} -m pip install -U huggingface_hub hf_xet" >&2
    exit 1
  fi
}

hf_download() {
  "${PYTHON_BIN}" -m huggingface_hub.cli.hf download "$@"
}

download_component() {
  local label="$1"
  shift

  echo
  echo "==> Downloading ${label}"
  hf_download "$@" --max-workers "${MAX_WORKERS}"
}

link_or_download_shared_dir() {
  local label="$1"
  local target_name="$2"
  shift 2

  local target_path="${BASE_DIR}/${target_name}"
  local source_path="${PRETRAIN_DIR}/${target_name}"

  if [[ "${REUSE_PRETRAIN_COMPONENTS}" == "1" && -d "${source_path}" ]]; then
    if [[ -L "${target_path}" ]]; then
      local current_target
      current_target="$(readlink "${target_path}")"
      if [[ "${current_target}" == "${source_path}" ]]; then
        echo "exists: ${target_path} -> ${source_path}"
        return
      fi
      echo "Error: ${target_path} is already a symlink to ${current_target}" >&2
      exit 1
    fi

    if [[ -e "${target_path}" ]]; then
      echo "exists: ${target_path}"
      return
    fi

    echo
    echo "==> Reusing ${label} from video_pretrain"
    ln -s "${source_path}" "${target_path}"
    echo "created: ${target_path} -> ${source_path}"
    return
  fi

  download_component "${label}" "$@"
}

main() {
  mkdir -p "${BASE_DIR}" "${HF_HOME}" "${HF_XET_CACHE}"
  ensure_deps

  echo "Root dir: ${ROOT_DIR}"
  echo "Target dir: ${BASE_DIR}"
  echo "Pretrain dir: ${PRETRAIN_DIR}"
  echo "HF_HOME: ${HF_HOME}"
  echo "HF_XET_CACHE: ${HF_XET_CACHE}"
  echo "Reuse pretrain text_encoder/VAE: ${REUSE_PRETRAIN_COMPONENTS}"
  echo
  echo "If this script is interrupted, run the same command again."

  download_component "GigaWorld-0 Video GR1 transformer" \
    "${GIGAWORLD_REPO}" \
    --include "transformer/*" \
    --local-dir "${BASE_DIR}"

  link_or_download_shared_dir "T5-11B text encoder" "text_encoder" \
    "${TEXT_ENCODER_REPO}" \
    --local-dir "${BASE_DIR}/text_encoder"

  link_or_download_shared_dir "Wan VAE" "vae" \
    "${VAE_REPO}" \
    --include "vae/*" \
    --local-dir "${BASE_DIR}"

  echo
  echo "Download commands finished. Expected inference paths:"
  echo "  --transformer-model-path ${BASE_DIR}/transformer"
  echo "  --text-encoder-model-path ${BASE_DIR}/text_encoder"
  echo "  --vae-model-path ${BASE_DIR}/vae"
  echo
  echo "Current size:"
  du -sh "${BASE_DIR}" 2>/dev/null || true
}

main "$@"
