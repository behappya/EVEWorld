#!/usr/bin/env bash
set -euo pipefail

# Download DreamGenBench evaluation code only. LFS smudge is disabled so this
# does not automatically download large media/model files.

ROOT_DIR="${ROOT_DIR:-/home/jovyan/gagibench}"
TARGET_DIR="${TARGET_DIR:-${ROOT_DIR}/GR00T-Dreams}"
REPO_URL="${REPO_URL:-https://github.com/NVIDIA/GR00T-Dreams.git}"

mkdir -p "${ROOT_DIR}"

echo "DreamGenBench code downloader"
echo "Target dir: ${TARGET_DIR}"
echo "Repo URL:   ${REPO_URL}"
echo "LFS smudge: disabled"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "Repo exists; fetching latest refs."
  git -C "${TARGET_DIR}" fetch --depth 1 origin main
  git -C "${TARGET_DIR}" checkout main
  git -C "${TARGET_DIR}" reset --hard origin/main
else
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
fi

echo
echo "DreamGenBench code ready: ${TARGET_DIR}"
echo "Official entry points:"
echo "  python -m dreamgenbench.eval_sr_qwen_whole"
echo "  python -m dreamgenbench.eval_sr_gpt4o_whole"
echo "  python -m dreamgenbench.eval_qwen_pa"

