#!/usr/bin/env bash
#
# Clone the GigaWorld-0 backbone that the DreamGenBench, WorldArena 1.0,
# EWMBench and PBench recipes fine-tune, and install the two upstream framework
# packages the training entry points import.
#
# Usage:
#   bash scripts/setup/clone_gigaworld.sh [DEST] [--dest DIR] [--ref REV] [--no-deps]
#
# Defaults: DEST=third_party/giga-world-0, upstream main branch, dependencies
# installed with ${PYTHON:-python} -m pip. Set PYTHON to pick an interpreter
# other than the first ``python`` on PATH.
#
set -euo pipefail

REPO_URL="https://github.com/open-gigaai/giga-world-0"
DEST="third_party/giga-world-0"
REF=""
INSTALL_DEPS=1

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
        --no-deps)
            INSTALL_DEPS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "clone_gigaworld.sh: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            DEST="$1"
            shift
            ;;
    esac
done

node_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${node_script_dir}/../.." && pwd)"
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

python_bin="${PYTHON:-python}"

if [ "${INSTALL_DEPS}" -eq 1 ]; then
    echo "installing the upstream framework packages (use --no-deps to skip)"
    "${python_bin}" -m pip install "git+https://github.com/open-gigaai/giga-train.git"
    "${python_bin}" -m pip install "git+https://github.com/open-gigaai/giga-datasets.git"
else
    echo "skipping the upstream framework packages; install them later with:"
    echo "  ${python_bin} -m pip install git+https://github.com/open-gigaai/giga-train.git"
    echo "  ${python_bin} -m pip install git+https://github.com/open-gigaai/giga-datasets.git"
fi

echo
echo "backbone checkout: ${repo_root}/${DEST}"
echo "The video transformer weights come from scripts/setup/download_models.sh;"
echo "point GW0_MODEL_DIR at that directory, or leave it unset to use the"
echo "default under \${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}."
