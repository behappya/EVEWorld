#!/usr/bin/env bash
#
# Fetch the backbones, the released checkpoints and the third-party weights the
# EVEWorld recipes need. Everything lands under
# ${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}; docs/checkpoints.md documents the
# layout, the flags and the verification steps, and the table below is the
# executable copy of it.
#
# Usage:
#   bash scripts/setup/download_models.sh [--checkpoint-root DIR] [--only NAME]
#                                         [--dry-run]
#
# Each entry is  name|repo_id|include_globs|subdirectory  and can be fetched on
# its own with --only NAME. Downloads resume and skip files that are already
# present, so re-running is cheap. Nothing here needs a GPU.
#
set -euo pipefail

CHECKPOINT_ROOT="${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}"
HF_CLI="${HF_CLI:-huggingface-cli}"
DRY_RUN=0
ONLY=()

MODELS=(
    "giga-world-0-video-pretrain-2b|open-gigaai/GigaWorld-0-Video-Pretrain-2b|giga-world-0-video-pretrain-2b"
    "giga-world-0-video-gr1-2b|open-gigaai/GigaWorld-0-Video-GR1-2b|giga-world-0-video-gr1-2b"
    "wan2.2-ti2v-5b|Wan-AI/Wan2.2-TI2V-5B-Diffusers|wan2.2-ti2v-5b"
    "flowwam-stage1|YixiangChen/FlowWAM|flowwam"
    "flowwam-robotwin|YixiangChen/FlowWAM|flowwam"
    "grounding-dino-swin-t-ogc|ShilongLiu/GroundingDINO|groundingdino"
    "sam2.1-hiera-large|facebook/sam2.1-hiera-large|sam2"
    "sam2.1-hiera-tiny|facebook/sam2.1-hiera-tiny|sam2"
    "qwen2.5-vl-7b-instruct|Qwen/Qwen2.5-VL-7B-Instruct|qwen2.5-vl-7b-instruct"
)

# Files to pull from each repository. An empty list downloads the whole repo.
includes_for() {
    case "$1" in
        giga-world-0-video-pretrain-2b|giga-world-0-video-gr1-2b)
            echo "transformer/config.json transformer/diffusion_pytorch_model.safetensors"
            ;;
        flowwam-stage1)
            echo "flowwam_worldarena_stage1.safetensors"
            ;;
        flowwam-robotwin)
            echo "flowwam_robotwin.safetensors flowwam_robotwin_action_norm_stats.npz"
            ;;
        grounding-dino-swin-t-ogc)
            echo "groundingdino_swint_ogc.pth GroundingDINO_SwinT_OGC.cfg.py"
            ;;
        sam2.1-hiera-large)
            echo "sam2.1_hiera_large.pt sam2.1_hiera_l.yaml"
            ;;
        sam2.1-hiera-tiny)
            echo "sam2.1_hiera_tiny.pt sam2.1_hiera_t.yaml"
            ;;
        *)
            echo ""
            ;;
    esac
}

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint-root)
            CHECKPOINT_ROOT="$2"
            shift 2
            ;;
        --checkpoint-root=*)
            CHECKPOINT_ROOT="${1#*=}"
            shift
            ;;
        --only)
            ONLY+=("$2")
            shift 2
            ;;
        --only=*)
            ONLY+=("${1#*=}")
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "download_models.sh: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

models_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${models_script_dir}/../.." && pwd)"
cd "${repo_root}"

if [ "${DRY_RUN}" -eq 0 ]; then
    if ! command -v "${HF_CLI}" >/dev/null 2>&1; then
        if command -v hf >/dev/null 2>&1; then
            HF_CLI="hf"
        else
            echo "${HF_CLI} not found on PATH." >&2
            echo "Install it with ${PYTHON:-python} -m pip install 'huggingface_hub[cli]'." >&2
            exit 1
        fi
    fi
    if command -v hf_transfer >/dev/null 2>&1; then
        echo "using hf_transfer for the transfers"
    fi
fi

selected() {
    local name="$1"
    local want
    if [ "${#ONLY[@]}" -eq 0 ]; then
        return 0
    fi
    for want in "${ONLY[@]}"; do
        if [ "${want}" = "${name}" ]; then
            return 0
        fi
    done
    return 1
}

echo "checkpoint root: ${repo_root}/${CHECKPOINT_ROOT}"
echo

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name repo dest <<< "${entry}"
    if ! selected "${name}"; then
        continue
    fi

    target="${CHECKPOINT_ROOT}/${dest}"
    cmd=("${HF_CLI}" download "${repo}" --local-dir "${target}")
    include="$(includes_for "${name}")"
    if [ -n "${include}" ]; then
        read -r -a globs <<< "${include}"
        for glob in "${globs[@]}"; do
            cmd+=(--include "${glob}")
        done
    fi

    echo "== ${name} (${repo})"
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '   cd %s &&' "${repo_root}"
        printf ' %q' "${cmd[@]}"
        printf '\n\n'
        continue
    fi

    mkdir -p "${target}"
    "${cmd[@]}"
    echo
done

echo "Done. Point the matching variables at the files that were just written:"
echo "  GW0_MODEL_DIR=${repo_root}/${CHECKPOINT_ROOT}/giga-world-0-video-pretrain-2b"
echo "  SAM2_CHECKPOINT=${repo_root}/${CHECKPOINT_ROOT}/sam2/sam2.1_hiera_large.pt"
echo "  SAM2_CHECKPOINT (tiny variant)=${repo_root}/${CHECKPOINT_ROOT}/sam2/sam2.1_hiera_tiny.pt"
echo "  GROUNDING_DINO_WEIGHTS=${repo_root}/${CHECKPOINT_ROOT}/groundingdino/groundingdino_swint_ogc.pth"
echo "  GROUNDING_DINO_CONFIG=${repo_root}/${CHECKPOINT_ROOT}/groundingdino/GroundingDINO_SwinT_OGC.cfg.py"
echo "  QWEN_IF_MODEL=${repo_root}/${CHECKPOINT_ROOT}/qwen2.5-vl-7b-instruct"
echo "Copy .env.example to .env and fill those in; scripts/setup/check_environment.py"
echo "reports which of them it can find."
