<h1 align="center">EVEWorld</h1>
<p align="center"><strong>Physical Evolution Supervision for Embodied World Models</strong></p>

<p align="center">
  <a href="https://behappya.github.io/EVEWorld/">Project Page</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#reproducing-the-paper">Reproduction</a> ·
  <a href="#citation">BibTeX</a>
</p>

<p align="center">
  <a href="https://behappya.github.io/EVEWorld/"><img src="https://img.shields.io/badge/Project%20Page-EVEWorld-4c4cf0" alt="Project page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License: Apache-2.0"></a>
  <a href="docs/installation.md"><img src="https://img.shields.io/badge/Python-3.11-3776ab" alt="Python 3.11"></a>
  <a href="https://github.com/behappya/EVEWorld/actions/workflows/tests.yml"><img src="https://github.com/behappya/EVEWorld/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
</p>

<p align="center">
  <img src="assets/comparison.png" width="95%" alt="Comparison of Standard SFT, IGR, and EVEWorld on physically consistent target evolution">
</p>

<p align="center"><em>EVEWorld combines restoration supervision for target-instance consistency with temporal alignment for cross-frame consistency.</em></p>

## Overview

Visual plausibility does not guarantee physically consistent target evolution. In embodied video world models, the manipulated target may be spuriously duplicated or disappear during a rollout, while count-preserving deformation or identity drift can still occur across adjacent frames.

**EVEWorld** introduces two complementary forms of evolution supervision: **Instance-Guided Restoration (IGR)** restores clean demonstrations from count-perturbed inputs to promote target-instance consistency, while **Temporal Instance Alignment (TIA)** aligns target representations across adjacent frames to promote cross-frame consistency. We further introduce **Model Laziness Rate (MLR)**, a rollout-level, occlusion-aware and persistence-aware diagnostic for persistent target-instance violations.

## Key results

| Setting | Standard SFT | EVEWorld | Main observation |
|---|---:|---:|---|
| DreamGenBench · MLR ↓ | 11.11% | **1.59%** | −85.7% relative |
| DreamGenBench · Qwen-IF ↑ | 73.81 | **80.16** | improved instruction following |
| DreamGenBench · Gemini-IF ↑ | 53.57 | **60.85** | improved instruction following |
| WorldArena 1.0 · Overall ↑ | 53.95 | **56.76** | zero-shot domain transfer |
| WorldArena 1.0 · MLR ↓ | 27.39% | **13.38%** | fewer persistent count violations |

The main DreamGen comparison uses matched post-training settings. Cross-backbone FlowWAM/RoboTwin results are reported separately because component contributions are backbone-dependent.

## What is released

- [x] IGR training pipeline
- [x] TIA alignment module
- [x] MLR evaluation protocol
- [x] DreamGen / WorldArena / EWMBench / RoboTwin evaluation entry points
- [x] paper-table reproduction scripts
- [x] aggregate paper tables under `results/paper/tables/` and the item-level row schema in `results/item_level/README.md`
- [x] environment and third-party-backbone setup scripts
- [x] qualitative project-page assets
- [ ] model weights — not released yet; the reproduction scripts train them from the released backbones
- [ ] per-item evaluation outputs — the harnesses under `scripts/evaluate/` regenerate them

## Quick Start

### Main GigaWorld-based setting

```bash
conda env create -f envs/gigaworld.yaml
conda activate gigaworld
pip install -e ".[train,eval]"
python scripts/setup/check_environment.py
```

`python scripts/setup/check_environment.py` exits non-zero only when a core item is missing; optional items are reported and otherwise ignored unless `--strict` is given.

### Evaluation only

```bash
conda env create -f envs/evaluation.yaml
conda activate eveworld-eval
pip install -e ".[eval]"
```

### FlowWAM / RoboTwin

```bash
conda env create -f envs/flowwam.yaml
conda activate flowwam
pip install -e ".[train,eval]"
```

The first end-to-end reproduction is `bash scripts/reproduce/table1_dreamgen.sh`; the full artifact map is in [Reproducing the paper](#reproducing-the-paper).

### Reference compute

The main paper setting uses 8× NVIDIA H20Z GPUs with an effective batch size of 64. Reproduction requirements differ across evaluation-only, GigaWorld, and FlowWAM workflows; see `docs/installation.md` and `docs/reproduction.md` for per-task requirements.

## Backbone setup

EVEWorld integrates with GigaWorld-0 for the main DreamGen setting and FlowWAM for the cross-backbone RoboTwin experiments. Upstream source trees and weights are not redistributed in this repository.

```bash
bash scripts/setup/clone_gigaworld.sh   # -> third_party/giga-world-0
bash scripts/setup/clone_flowwam.sh     # -> third_party/FlowWAM
```

GigaWorld-0 comes from <https://github.com/open-gigaai/giga-world-0>. Both scripts accept a destination argument and a `--ref` override, and both checkout directories are gitignored. `clone_gigaworld.sh` also installs the `giga-train` and `giga-datasets` packages, which are not on PyPI. FlowWAM training starts from the released Stage-1 checkpoint on top of the Wan2.2-TI2V-5B base; both are fetched by `bash scripts/setup/download_models.sh`. See [`docs/third_party.md`](docs/third_party.md) and [`third_party/README.md`](third_party/README.md) for the weight table and licenses.

## Method

**Instance-Guided Restoration (IGR).** IGR constructs count-edited demonstrations by inserting an additional target instance and trains the model to restore the clean latent representation. Restoration-related regions receive 3× reconstruction weight, with the spatial weight map normalized to unit mean.

**Temporal Instance Alignment (TIA).** TIA operates at a selected transformer layer, performs local cross-frame target matching, and transports matched previous-frame features into the current representation through a bounded residual. A correspondence objective favors the true next-frame target location.

The two objectives are optimized jointly. At inference, EVEWorld requires only the initial image and instruction; external target localization is not required.

<p align="center">
  <img src="assets/method_overview.png" width="85%" alt="EVEWorld method overview: Instance-Guided Restoration and Temporal Instance Alignment">
</p>

See [`docs/training.md`](docs/training.md) for training schedules and implementation details.

## Reproducing the paper

Each paper artifact has a script under `scripts/reproduce/`:

| Paper artifact | Command | Output |
|---|---|---|
| Table 1 · DreamGenBench | `bash scripts/reproduce/table1_dreamgen.sh` | `results/item_level/dreamgen/` |
| Table 3 · WorldArena 1.0 | `bash scripts/reproduce/table3_worldarena.sh` | `results/item_level/worldarena/` |
| Table 4 · EWMBench (AgiBot) | `bash scripts/reproduce/table4_ewmbench.sh` | `results/item_level/ewmbench/` |
| Table 5 · RoboTwin (FlowWAM) | `bash scripts/reproduce/table5_robotwin.sh` | `results/item_level/robotwin/table5.json` |
| Table 6 · Component ablation | `bash scripts/reproduce/table6_ablation.sh` | `results/item_level/dreamgen/` |
| Figure 4 · CFG grid | `bash scripts/reproduce/figure4_cfg.sh` | grid cells under `${GRID_ROOT}`; summaries under `results/item_level/dreamgen/` |

See [`docs/reproduction.md`](docs/reproduction.md) for inputs, compute requirements, and expected values.

## Results

### DreamGenBench — main GigaWorld-based setting

DreamGenBench with the GigaWorld-0 backbone. The general video models are evaluated zero-shot; Standard SFT and EVEWorld are post-trained on the 92 DreamGen clips with the same initialization and the same 250-step budget.

| Method | MLR (%) ↓ | Qwen-IF (%) ↑ | Gemini-IF (%) ↑ |
|---|---|---|---|
| CogVideoX1.5-5B-I2V | 28.57 | 38.89 | 5.56 |
| Wan2.2-TI2V-5B | 17.46 | 38.89 | 10.32 |
| Wan2.2-I2V-A14B | 11.11 | 64.29 | 15.87 |
| Cosmos-Predict2-2B | 14.29 | 62.70 | 24.60 |
| GigaWorld-0 | 12.70 | 79.37 | 60.19 |
| Standard SFT | 11.11 | 73.81 | 53.57 |
| **EVEWorld** | **1.59** | **80.16** | **60.85** |

MLR is computed on the shared eligible set `U_63` for every model. EVEWorld reduces MLR from 11.11% to 1.59% against Standard SFT, an 85.7% relative reduction, while instruction following improves on both judges.

### Component ablation

All four variants share the initialization, the 92-clip training set and the 250-step optimization budget, so the differences isolate IGR and TIA. Gemini-IF is broken down over the three generalization splits: unseen environments, unseen objects and unseen behaviors.

| Variant | IGR | TIA | Env ↑ | Object ↑ | Behavior ↑ | Overall ↑ | MLR (%) ↓ |
|---|---|---|---|---|---|---|---|
| Standard SFT | | | 51.72 | 42.00 | 67.02 | 53.57 | 11.11 |
| IGR only | ✓ | | 49.43 | 36.67 | **69.50** | 51.85 | 4.76 |
| TIA only | | ✓ | 55.17 | 42.67 | 68.79 | 55.29 | 7.94 |
| **EVEWorld** | ✓ | ✓ | **72.41** | **50.33** | 64.89 | **60.85** | **1.59** |

### Cross-backbone evaluation — FlowWAM / RoboTwin

Component contributions are backbone-dependent: TIA-only is strongest on several FlowWAM metrics, while the full model remains better than matched Standard SFT on every reported metric.

| Variant | IGR | TIA | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Flow-EPE ↓ | MLR (%) ↓ |
|---|---|---|---|---|---|---|---|
| FlowWAM | | | 12.218 | 0.748 | 0.383 | 3.033 | 52.05 |
| Standard SFT | | | 11.687 | 0.735 | 0.414 | 2.828 | 47.89 |
| + IGR | ✓ | | 12.544 | 0.770 | 0.380 | 2.487 | 40.03 |
| + TIA | | ✓ | **13.445** | **0.776** | **0.333** | **1.850** | **22.54** |
| **EVEWorld** | ✓ | ✓ | 12.765 | 0.769 | 0.365 | 2.207 | 35.21 |

The FlowWAM arm is a rank-32 LoRA fine-tune over the 2,250 RoboTwin training episodes, evaluated on 250 held-out episodes at 40 denoising steps and CFG 5.0 with the robot-only flow condition.

### Generalization results

WorldArena 1.0 transfer: 1,000 prompts evaluated zero-shot from the step-250 DreamGenBench checkpoint of each arm, so the table measures transfer to a different prompt distribution and a different generation setting.

| Model | Overall ↑ | MLR (%) ↓ |
|---|---|---|
| Standard SFT | 53.95 | 27.39 |
| **EVEWorld** | **56.76** | **13.38** |

EWMBench after AgiBot post-training: both arms start from the GigaWorld-0 video-pretrain checkpoint, use the same AgiBot clips and the same optimization budget, and are scored by the official suite on 777 AgiBot clips generated three times each.

| Model | Motion ↑ | Semantics ↑ | DYN ↑ | HSD ↑ | nDTW ↑ | Scene ↑ | Overall ↑ |
|---|---|---|---|---|---|---|---|
| GigaWorld-0 (pretrained) | 50.30 | 2.2497 | 7.19 | 23.29 | 19.83 | **89.58** | 3.6486 |
| Standard SFT | 61.51 | 2.2477 | 14.88 | **24.35** | **22.28** | 84.38 | 3.7066 |
| **EVEWorld** | **63.65** | **2.2524** | **17.55** | 24.12 | 21.97 | 86.37 | **3.7525** |

Protocols, metrics, and judge details are in [`docs/evaluation.md`](docs/evaluation.md).

## Data and checkpoints

`data/` holds the evaluation splits, per-clip metadata, and the judge prompts; the datasets themselves stay outside the repository. `checkpoints/` holds the download instructions, and `bash scripts/setup/download_models.sh` fetches the listed weights into `${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}`, skipping files that already exist. See [`docs/data_preparation.md`](docs/data_preparation.md) and [`docs/checkpoints.md`](docs/checkpoints.md).

## Repository structure

```text
EVEWorld/
├── assets/        figures, qualitative panels, and project-page media
├── checkpoints/   download instructions for the released weights
├── configs/       training, evaluation, and ablation run configurations
├── data/          splits, per-clip metadata, and judge prompts (datasets stay outside the repo)
├── docs/          installation, data, training, inference, evaluation, reproduction, checkpoints, third party
├── envs/          conda environment files (gigaworld, flowwam, eveworld-eval)
├── experiments/   alternative supervision designs and the analyses behind the paper figures
├── results/       aggregate paper tables and the item-level row schema
├── scripts/       setup, prepare, train, inference, evaluate, and reproduce entry points
├── src/eveworld/  the library: IGR, TIA, data tooling, integrations, and evaluation
├── tests/         unit tests for the IGR, TIA, and MLR components
├── third_party/   clone instructions for the two backbones (checkouts are gitignored)
└── index.html  css/  js/   static project page
```

## Documentation

| Document | Covers |
|---|---|
| [`docs/installation.md`](docs/installation.md) | environments, extras, hardware, and environment variables |
| [`docs/data_preparation.md`](docs/data_preparation.md) | datasets, splits, and the `scripts/prepare/` converters |
| [`docs/training.md`](docs/training.md) | training recipes, schedules, and commands |
| [`docs/inference.md`](docs/inference.md) | inference entry points and generation settings |
| [`docs/evaluation.md`](docs/evaluation.md) | MLR, judges, and the benchmark harnesses |
| [`docs/reproduction.md`](docs/reproduction.md) | inputs, compute, and expected values per artifact |
| [`docs/checkpoints.md`](docs/checkpoints.md) | downloader flags, landing paths, and checkpoint layout |
| [`docs/third_party.md`](docs/third_party.md) | backbones, weights, and licenses |

## Citation

The author list remains anonymous; the release links the project page and this repository.

```bibtex
@software{eveworld2026,
  title  = "{EVEWorld}: Physical Evolution Supervision for Embodied World Models",
  author = "{Anonymous Author(s)}",
  year   = {2026},
  url    = "{https://behappya.github.io/EVEWorld/}",
  note   = "Code and project page released for anonymous review."
}
```

The machine-readable entry is [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [Apache-2.0 License](LICENSE). Third-party components — GigaWorld-0 / GigaModels, FlowWAM, GroundingDINO, SAM2, RAFT, LPIPS, WorldArena, DreamGenBench, EWMBench, and PBench — keep their own licenses; see [`docs/third_party.md`](docs/third_party.md) and [`third_party/README.md`](third_party/README.md).

## Acknowledgements

EVEWorld builds on the GigaWorld-0 and FlowWAM backbones, and relies on GroundingDINO and SAM2 for target localization and tracking. See [`docs/third_party.md`](docs/third_party.md) for details and licenses.

## Contributing

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). The unit tests run with `pytest -q`, and the formatting and lint hooks with `pre-commit run --all-files`; the `tests` workflow runs both on every push.
