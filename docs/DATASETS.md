# Datasets and Checkpoints

This page lists every dataset, benchmark, and model checkpoint used by the
EVEWorld experiments, and where the preparation code lives.

## Overview

| Asset | Role in the paper | Preparation code |
|---|---|---|
| DreamGen GR1 fine-tuning split (92 videos) | Main post-training set | `giga-world-0/scripts/download_gr1_finetune_dataset_hf.sh`, `benchmarks/dreamgenbench/` |
| DreamGenBench (126 tasks) | Main evaluation benchmark | `benchmarks/dreamgenbench/download_dreamgenbench_*.sh` |
| GigaWorld-0 Video-Pretrain-2B | Backbone initialization | `giga-world-0/scripts/download_video_pretrain_hf.sh` |
| WorldArena 1.0 (1,000 prompts) | Zero-shot cross-domain evaluation | `benchmarks/worldarena/` |
| AgiBot (777 clips) | Cross-distribution training | `eveworld/agibot/` |
| EWMBench | Cross-distribution evaluation | `benchmarks/ewmbench/` |
| RoboTwin (50 tasks, held-out episodes 45-49) | Cross-backbone evaluation | `benchmarks/robotwin_flowwam/` |
| FlowWAM Stage-1 + Wan2.2-TI2V-5B | Cross-backbone base model | `benchmarks/robotwin_flowwam/README.md` |
| GroundingDINO (weights) | Target detection (IGR annotation + MLR) | set `GDINO_PATH` in detection scripts |
| PBench Robot (913 QA) | Output-length robustness | `benchmarks/pbench/` |
| CogVideoX / Wan2.2 / Cosmos weights | Cross-model baselines | `benchmarks/baselines/xmodel_download/` |

## Layout convention

Large artifacts (datasets, checkpoints, generated videos) are **not** part of
this repository. Scripts resolve them under a data root, by default:

```
${GAGI_ROOT}/
├── gr1_finetune_data/            # packed GR1 fine-tuning clips
├── gr1_dreamgen_eval/            # DreamGenBench inputs and eval outputs
├── giga_world_0_video_pretrain/  # backbone checkpoint
├── giga_world_0_outputs/         # training runs and generated videos
├── flowwam/                      # FlowWAM data, checkpoints, held-out outputs
└── worldarena1/                  # WorldArena 1.0 manifests, videos, evaluation
```

Set `EVEWORLD_ROOT` to this repository's root and `GAGI_ROOT` to your data
root; see `eveworld/common/env.sh` for the full shared contract.

## Licenses

All datasets, benchmarks, and pretrained models remain subject to their
original licenses (DreamGen/DreamGenBench: NVIDIA; WorldArena; AgiBot:
AgiBotTech; EWMBench; RoboTwin; FlowWAM; Wan/CogVideoX/Cosmos model licenses;
GroundingDINO: IDEA). This repository only contains code and evaluation
protocols.
