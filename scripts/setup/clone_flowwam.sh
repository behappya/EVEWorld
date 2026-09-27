#!/usr/bin/env bash
#
# Clone the FlowWAM backbone checkout used by the RoboTwin cross-backbone
# experiments. The Wan2.2-TI2V-5B base weights and the released FlowWAM
# Stage-1 checkpoint are fetched separately by scripts/setup/download_models.sh.
#
# Usage:
#   bash scripts/setup/clone_flowwam.sh [DEST] [--dest DIR] [--ref REV]
#
# Defaults: DEST=third_party/FlowWAM, upstream main branch. Generation and the
# metric stack import from the checkout, so export FLOWWAM_ROOT when it lives
# somewhere other than third_party/FlowWAM.
#
set -euo pipefail

REPO_URL="https://github.com/YixiangChen515/FlowWAM"
DEST="third_party/FlowWAM"
REF=""

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dest)
            DEST="$2"
            shift 2
            ;;
        --dest=*)
            DEST="${1#*=}"
            shift
            ;;
        --ref)
            REF="$2"
            shift 2
            ;;
        --ref=*)
            REF="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "clone_flowwam.sh: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            DEST="$1"
            shift
            ;;
    esac
done

flowwam_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${flowwam_script_dir}/../.." && pwd)"
cd "${repo_root}"

if [ -d "${DEST}/.git" ]; then
    echo "checkout already present: ${DEST} (remove it and re-run to re-clone)"
else
    mkdir -p "$(dirname "${DEST}")"
    if [ -n "${REF}" ]; then
        echo "git clone --depth 1 --branch ${REF} ${REPO_URL} ${DEST}"
        git clone --depth 1 --branch "${REF}" "${REPO_URL}" "${DEST}"
    else
        echo "git clone --depth 1 ${REPO_URL} ${DEST}"
        git clone --depth 1 "${REPO_URL}" "${DEST}"
    fi
fi

echo
echo "checkout: ${repo_root}/${DEST}"
echo "export FLOWWAM_ROOT=${repo_root}/${DEST}"
echo "Generation and the RoboTwin metrics add \${FLOWWAM_ROOT} and"
echo "\${FLOWWAM_ROOT}/inference to PYTHONPATH; the training and inference"
echo "scripts here do that for you when the variable is set."
