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
conda activate EVEWorld
# `environment.yml` names the env `EVEWorld`; to use a different name, create it
# with `conda env create -f environment.yml -n <your-name>` and select it when you
# launch cluster scripts with `CONDA_ENV=<your-name>`.

# Vendored, pinned framework snapshot (used for all experiments):
pip install -e ./giga-models

# GigaWorld-0 training/data frameworks:
pip install git+https://github.com/open-gigaai/giga-train.git
pip install git+https://github.com/open-gigaai/giga-datasets.git

# CUDA-matched PyTorch (cu128) + NATTEN, if the pip build above resolved a different
# CUDA build (check `python -c "import torch; print(torch.version.cuda)"`):
CONDA_ENV=EVEWorld bash giga-world-0/scripts/install_torch_cuda128.sh
# Remaining training deps:
CONDA_ENV=EVEWorld bash giga-world-0/scripts/install_gigaworld_training_deps.sh
```

The exact versions pinned in `requirements.txt` mirror the reference environment. A conda skeleton is provided in `environment.yml`.

## External tools

| Tool | Used for | Where documented |
|---|---|---|
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (+ weights) | target detection for IGR annotations and MLR | set the model path in the detection scripts (`GDINO_PATH`) |
| [SAM2](https://github.com/facebookresearch/sam2) (+ checkpoint, optional) | gripper-occlusion evidence for the MLR occlusion rules | `--sam2-checkpoint` / `SAM2_CHECKPOINT`; see `benchmarks/worldarena/README.md` |
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

- Cluster/job scripts select their conda env through `CONDA_ENV`; the default is `EVEWorld`, the environment this repository's `environment.yml` creates. Every script keeps that default overridable, e.g. `CONDA_ENV=my_env bash eveworld/pipeline/t4g_final_align_kjob.sh`.
- `benchmarks/dreamgenbench/setup_dreamgenbench_eval_venv.sh` builds the separate DreamGenBench judging venv (Qwen2.5-VL based), and `benchmarks/pbench/setup_videophy_env.sh` builds the VideoPhy-2 environment. These are intentionally separate from the main conda env.
- Many scripts read `EVEWORLD_ROOT` (repository root), `GAGI_ROOT` (data root), and related variables. See `eveworld/common/env.sh` for the shared environment contract.
- The WorldArena MLR runner needs no SAM2 install when it runs with `--occlusion-rule none`. The occlusion rules of the paper's Algorithm 1 additionally need SAM2 (`pip install git+https://github.com/facebookresearch/sam2`) plus a checkpoint passed as `--sam2-checkpoint` or `SAM2_CHECKPOINT`; `--occlusion-rule paper_overlap` is the paper configuration, and `mlr_protocol_profiles.yaml` collects all the runnable presets.

## Environments this repository uses

`EVEWorld` is the only environment the repository creates itself
(`conda env create -f environment.yml`); every other name below is an
external or per-benchmark environment that a runner defaults to and that you
can point elsewhere with the override variable.

| Environment | Role | Provided by | Override |
|---|---|---|---|
| `EVEWorld` | main env: the `giga_models` / `giga_train` packages, torch/diffusers, generation and most judging, MLR scorers | `environment.yml` | `CONDA_ENV` (`MLR_CONDA_ENV` for the MLR runners) |
| `giga_world1` | cross-model I2V baseline generation (diffusers ≥ 0.39, four pipelines) and GroundingDINO detection | you build it | `CONDA_ENV` / `GEN_CONDA_ENV` |
| `WorldArena` | official WorldArena 1.0 evaluator | `benchmarks/worldarena/prepare_evaluator.sh` | `EVAL_CONDA_ENV` |
| `EWMBench`, `vila` | official EWMBench metrics and the VILA judge | their upstream repos | `CONDA_ENV` |
| `giga_world_train_venv` | isolated training venv for GigaWorld-0 training | `giga-world-0/scripts/setup_gigaworld_train_venv.sh` | `TRAIN_VENV` |
| `dreamgenbench_eval_venv`, `videophy` | per-benchmark judging venvs | the benchmark's own setup script (`benchmarks/dreamgenbench/setup_dreamgenbench_eval_venv.sh`, `benchmarks/pbench/setup_videophy_env.sh`) | script-local |

The vendored `giga-models/` package keeps its upstream README, which shows its
own `conda create -n giga_models` recipe; in this repository use
`environment.yml` / `EVEWorld` instead.
