#!/usr/bin/env bash
set -euo pipefail

# Resumable downloader for the public GR1 robot dataset used for the
# GigaWorld-0 DreamGen/GR1 fine-tuning setting.
#
# This script downloads only the dataset, then prepares a lightweight
# GigaWorld raw_data directory:
#   raw_data/1.mp4 -> raw_hf/gr1/1.mp4
#   raw_data/1.txt
#
# Re-run the same command if it is interrupted.

ROOT_DIR="${ROOT_DIR:-/data/datasets/gagi}"
BASE_DIR="${BASE_DIR:-${ROOT_DIR}/gr1_finetune_data}"
RAW_HF_DIR="${RAW_HF_DIR:-${BASE_DIR}/raw_hf}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${BASE_DIR}/raw_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_WORKERS="${MAX_WORKERS:-8}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
COPY_VIDEOS="${COPY_VIDEOS:-0}"
STRICT_MISSING="${STRICT_MISSING:-0}"

export RAW_HF_DIR RAW_DATA_DIR COPY_VIDEOS STRICT_MISSING

DATASET_REPO="nvidia/PhysicalAI-Robotics-GR00T-GR1"

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

prepare_raw_data() {
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import shutil
from pathlib import Path

raw_hf_dir = Path(os.environ["RAW_HF_DIR"])
raw_data_dir = Path(os.environ["RAW_DATA_DIR"])
copy_videos = os.environ.get("COPY_VIDEOS", "0") == "1"
strict_missing = os.environ.get("STRICT_MISSING", "0") == "1"

metadata_path = raw_hf_dir / "metadata.csv"
video_dir = raw_hf_dir / "gr1"
if not metadata_path.exists():
    raise SystemExit(f"metadata.csv not found: {metadata_path}")
if not video_dir.exists():
    raise SystemExit(f"video directory not found: {video_dir}")

raw_data_dir.mkdir(parents=True, exist_ok=True)
manifest_path = raw_data_dir / "manifest.jsonl"
missing_path = raw_data_dir / "missing_videos.txt"

with metadata_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit(f"No rows parsed from {metadata_path}")

seen = set()
missing = []
written = 0
with manifest_path.open("w", encoding="utf-8") as manifest:
    for row in rows:
        file_name = (row.get("file_name") or "").strip()
        prompt = (row.get("text") or "").strip()
        if not file_name or not prompt:
            raise SystemExit(f"Bad metadata row: {row!r}")
        if file_name in seen:
            raise SystemExit(f"Duplicate file_name in metadata: {file_name}")
        seen.add(file_name)

        src_video = video_dir / file_name
        if not src_video.exists():
            missing.append(str(src_video))
            continue

        stem = Path(file_name).stem
        dst_video = raw_data_dir / f"{stem}.mp4"
        dst_text = raw_data_dir / f"{stem}.txt"

        if dst_video.exists() or dst_video.is_symlink():
            if dst_video.is_symlink() and dst_video.resolve() == src_video.resolve():
                pass
            elif copy_videos and dst_video.is_file() and dst_video.stat().st_size == src_video.stat().st_size:
                pass
            else:
                raise SystemExit(f"Refusing to overwrite existing video: {dst_video}")
        else:
            if copy_videos:
                shutil.copy2(src_video, dst_video)
            else:
                dst_video.symlink_to(src_video)

        tmp_text = dst_text.with_suffix(".txt.tmp")
        tmp_text.write_text(prompt + "\n", encoding="utf-8")
        tmp_text.replace(dst_text)

        manifest.write(json.dumps(
            {
                "file_name": file_name,
                "video": str(dst_video),
                "source_video": str(src_video),
                "text": prompt,
            },
            ensure_ascii=True,
        ) + "\n")
        written += 1

if missing:
    missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
    message = (
        f"metadata rows: {len(rows)}\n"
        f"prepared pairs: {written}\n"
        f"missing videos: {len(missing)}\n"
        f"missing list: {missing_path}\n"
        "The current Hugging Face GR1 dataset card says it contains 92 videos; "
        "metadata.csv contains 100 rows, so these missing rows are skipped by default."
    )
    if strict_missing:
        raise SystemExit(message)
    print("WARNING: " + message.replace("\n", "\nWARNING: "))
elif missing_path.exists():
    missing_path.unlink()

print(f"prepared raw_data pairs: {written}")
print(f"raw_data_dir: {raw_data_dir}")
print(f"manifest: {manifest_path}")
PY
}

main() {
  mkdir -p "${BASE_DIR}" "${RAW_HF_DIR}" "${RAW_DATA_DIR}" "${HF_HOME}" "${HF_XET_CACHE}"
  ensure_deps

  echo "GR1 fine-tuning dataset downloader"
  echo "Dataset repo: ${DATASET_REPO}"
  echo "Base dir:     ${BASE_DIR}"
  echo "Raw HF dir:   ${RAW_HF_DIR}"
  echo "Raw data dir: ${RAW_DATA_DIR}"
  echo "HF_HOME:      ${HF_HOME}"
  echo "HF_XET_CACHE: ${HF_XET_CACHE}"
  echo "Copy videos:  ${COPY_VIDEOS}"
  echo "Strict missing videos: ${STRICT_MISSING}"
  echo
  echo "If this script is interrupted, run the same command again."

  echo
  echo "==> Downloading GR1 dataset files"
  hf_download "${DATASET_REPO}" \
    --repo-type dataset \
    --local-dir "${RAW_HF_DIR}" \
    --include "README.md" ".gitattributes" "metadata.csv" "gr1/*.mp4" \
    --max-workers "${MAX_WORKERS}"

  echo
  echo "==> Preparing GigaWorld raw_data directory"
  prepare_raw_data

  echo
  echo "Done. Next step is to pack the dataset on a GPU node:"
  echo "  python scripts/pack_data.py \\"
  echo "    --video-dir ${RAW_DATA_DIR} \\"
  echo "    --save-dir ${BASE_DIR}/packed_data \\"
  echo "    --text-encoder-model-path /data/datasets/gagi/giga_world_0_video_pretrain/text_encoder"
  echo
  echo "Current size:"
  du -sh "${BASE_DIR}" 2>/dev/null || true
}

main "$@"
