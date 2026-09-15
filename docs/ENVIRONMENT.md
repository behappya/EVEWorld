# Environment Setup

This document describes how to build the software environment for EVEWorld.

## Reference environment

All experiments were run with:

- Python 3.11 (conda environment), PyTorch 2.11.0 + CUDA 12.8
- Training/generation: nodes with 8x NVIDIA H20Z GPUs (jobs are submitted with the `kjob_*` / `launch_*_kjob.sh` wrappers; they are SLURM-style cluster scripts — adapt paths and scheduler flags to your cluster)
- Evaluation and data tooling: any CPU/GPU machine

## Quick setup

```bash
conda env create -f environment.yml
conda activate eveworld

# Vendored, pinned framework snapshot (used for all experiments):
pip install -e ./giga-models

# GigaWorld-0 training/data frameworks:
pip install git+https://github.com/open-gigaai/giga-train.git
pip install git+https://github.com/open-gigaai/giga-datasets.git

# CUDA-matched PyTorch build (cu128) if your default pip resolves a different one:
#   see giga-world-0/scripts/install_torch_cuda128.sh
# Remaining training deps:
#   bash giga-world-0/scripts/install_gigaworld_training_deps.sh
```

The exact versions pinned in `requirements.txt` mirror the reference environment. A conda skeleton is provided in `environment.yml`.

## External tools

| Tool | Used for | Where documented |
|---|---|---|
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (+ weights) | target detection for IGR annotations and MLR | set the model path in the detection scripts (`GDINO_PATH`) |
| [WorldArena 1.0 evaluator](https://github.com/WorldArena-Official/WorldArena) | WorldArena 1.0 metrics | `benchmarks/worldarena/README.md` |
| [DreamGenBench](https://github.com/NVIDIA/DreamGen) | DreamGenBench prompts/judging | `benchmarks/dreamgenbench/README.md` |
| [EWMBench](https://github.com/AgibotTech/EWMBench) | EWMBench metrics | `benchmarks/ewmbench/README.md` |
| [FlowWAM](https://github.com/) backbone | cross-backbone experiment | `benchmarks/robotwin_flowwam/README.md` |
| FlowWAM Stage-1 checkpoint + Wan2.2-TI2V-5B | cross-backbone experiment | `benchmarks/robotwin_flowwam/README.md` |

## API credentials for judging

Instruction-following judging calls LLM/VLM endpoints. Provide credentials via
environment variables — never hard-code keys:

- `OPENAI_API_KEY` — OpenAI-compatible endpoints (Qwen-IF judging, vLLM serving)
- `DIFROST_API_TOKEN`, `DIFROST_GENAI_BASE_URL`, `DIFROST_HOST`, `DIFROST_MODEL` — the Gemini/GPT consensus judge gateway (any OpenAI/GenAI-compatible gateway works; set `DIFROST_GENAI_BASE_URL` accordingly)

## Notes

- `benchmarks/dreamgenbench/setup_dreamgenbench_eval_venv.sh` builds the separate DreamGenBench judging venv (Qwen2.5-VL based), and `benchmarks/pbench/setup_videophy_env.sh` builds the VideoPhy-2 environment. These are intentionally separate from the main conda env.
- Many scripts read `EVEWORLD_ROOT` (repository root), `GAGI_ROOT` (data root), and related variables. See `eveworld/common/env.sh` for the shared environment contract.
