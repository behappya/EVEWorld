# Reproduction

`scripts/reproduce/` holds one wrapper per paper artefact. Each wrapper is a
thin driver: it resolves the repository root, exports the paths the stages
need, and chains data preparation, training, generation, judging and metrics
with `set -euo pipefail`, so the run stops at the first failing stage. The
underlying commands are the ones documented in
[`training.md`](training.md), [`inference.md`](inference.md) and
[`evaluation.md`](evaluation.md); the wrappers only fix their arguments.

```bash
bash scripts/reproduce/table1_dreamgen.sh
```

## Prerequisites

The wrappers assume the backbones and the released checkpoints are
already in place. Fetch them once with
[`scripts/setup/download_models.sh`](../scripts/setup/download_models.sh),
which downloads into `${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}` and can be
re-run cheaply because it skips files that already exist:

```bash
bash scripts/setup/clone_gigaworld.sh
bash scripts/setup/clone_flowwam.sh
bash scripts/setup/download_models.sh
```

The three conda environments of [`installation.md`](installation.md) are
needed as well: the GigaWorld-0 arms run in `gigaworld`, the FlowWAM arm in
`flowwam`, and the judging and metric stages in `eveworld-eval`. Every wrapper
reads its settings from `configs/`, so a variant can be reproduced by pointing
the wrapper's config variable at a different file under `configs/paper/`.

## Artefact map

| Paper artefact | Wrapper | Data | Headline result |
|---|---|---|---|
| Table 1 `tab:dreamgen_overall` | `scripts/reproduce/table1_dreamgen.sh` | DreamGenBench, 126 prompts. | EVEWorld 1.59 MLR / 80.16 Qwen-IF / 60.85 Gemini-IF. |
| Table 3 `tab:worldarena` | `scripts/reproduce/table3_worldarena.sh` | WorldArena 1.0, 1,000 prompts. | EVEWorld 56.76 Overall / 13.38 MLR. |
| Table 4 `tab:agibot_transfer` | `scripts/reproduce/table4_ewmbench.sh` | AgiBot, 777 clips. | EVEWorld 63.65 Motion / 3.7525 Overall. |
| Table 5 `tab:flowwam_transfer` | `scripts/reproduce/table5_robotwin.sh` | RoboTwin, 250 held-out episodes. | EVEWorld 12.765 PSNR / 2.207 Flow-EPE / 35.21 MLR. |
| Table 6 `tab:component_ablation` | `scripts/reproduce/table6_ablation.sh` | DreamGenBench, four arms. | EVEWorld 60.85 Gemini-IF / 1.59 MLR. |
| Figure 4 `fig:cfg` | `scripts/reproduce/figure4_cfg.sh` | DreamGenBench, 24 grid cells. | Gemini-IF rises with CFG while MLR does not fall monotonically. |

Every wrapper writes item-level records and an aggregate under
`results/item_level/<benchmark>/`, named after the arm it belongs to, so a
table cell can be traced back to its clip. The directories are `dreamgen`,
`worldarena`, `ewmbench` and `robotwin`, one per benchmark the wrappers
cover.

## Environment variables you will need

The variables below are the ones the stages read;
[`.env.example`](../.env.example) lists them with example values, and
[`installation.md`](installation.md) describes the setup that writes them.

| Variable | Used for |
|---|---|
| `EVEWORLD_CHECKPOINT_ROOT` | Root of the released and backbone checkpoints; defaults to `checkpoints`. |
| `HF_HOME` | Hugging Face cache, including the offline backbone download. |
| `GIGA_MODELS_DIR` | GigaModels source tree used by the GigaWorld-0 arms. |
| `GIGA_MODELS_CACHE`, `GIGA_MODELS_REPO_CACHE` | Kernel and repository caches of GigaModels. |
| `FLOWWAM_ROOT` | FlowWAM checkout under `third_party/FlowWAM`. |
| `GROUNDING_DINO_CONFIG`, `GROUNDING_DINO_WEIGHTS` | Detector of the MLR protocol. |
| `SAM2_CHECKPOINT` | SAM2 tracker weights used for the masks and the robot mask. |
| `DREAMGEN_DATA_ROOT`, `AGIBOT_DATA_ROOT`, `ROBOTWIN_DATA_ROOT`, `WORLDARENA_DATA_ROOT` | Source media of the four evaluation sets. |
| `QWEN_BASE_URL`, `QWEN_IF_MODEL` | Endpoint and local model directory of the Qwen judge. |
| `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` | API keys of the two judges; the first of each pair that is set wins. |
| `GEMINI_BASE_URL`, `DIFROST_GENAI_BASE_URL`, `DIFROST_API_TOKEN`, `DIFROST_HOST`, `DIFROST_THINKING_LEVEL` | Endpoint and gateway settings of the Gemini judge. |
| `CUDA_VISIBLE_DEVICES` | GPU selection, including per-shard selection. |

The judging stages need working credentials; without them a wrapper still
generates the clips and computes the deterministic metrics, and reports the
judge stage as failed.

## Table 1: DreamGenBench

**Input.** The DreamGenBench evaluation split, 126 prompts, with the shared
eligible set `U_63` for MLR. The wrapper trains the two GigaWorld-0 arms that
this repository owns and scores the released backbone and the four external
baselines from their downloaded outputs when they are present.

**Compute.** Two fine-tunes at `480 x 768`, 93 frames and effective batch 64
on 8 GPUs: the standard SFT baseline for 200 steps and the EVEWorld arm for
250 steps. Generation is 126 clips per model at 30 denoising steps and CFG
7.0, followed by the two judging passes at 49 frames per clip and the MLR
pass over the eligible set.

**Command.**

```bash
bash scripts/reproduce/table1_dreamgen.sh
```

**Expected numbers.**

| Model | MLR (%) | Qwen-IF (%) | Gemini-IF (%) |
|---|---|---|---|
| CogVideoX1.5-5B-I2V | 28.57 | 38.89 | 5.56 |
| Wan2.2-TI2V-5B | 17.46 | 38.89 | 10.32 |
| Wan2.2-I2V-A14B | 11.11 | 64.29 | 15.87 |
| Cosmos-Predict2-2B | 14.29 | 62.70 | 24.60 |
| GigaWorld-0 (pretrained) | 12.70 | 79.37 | 60.19 |
| Standard SFT | 11.11 | 73.81 | 53.57 |
| **EVEWorld** | **1.59** | **80.16** | **60.85** |

The MLR column is the seeded value of the shared `U_63` set: 11.11% is 7 of
63 and 1.59% is 1 of 63, so a single clip changes the EVEWorld column by 1.59
points and the SFT column by the same absolute amount.

## Table 3: WorldArena 1.0

**Input.** The 1,000-prompt WorldArena manifest with the 157 detector-eligible
prompts, and the step-250 checkpoint of the DreamGenBench arm.

**Compute.** Zero-shot generation of 1,000 clips at `480 x 768` and 93 frames,
30 denoising steps and CFG 7.0, then the eight local core metrics plus Overall
and the MLR pass on the shared eligible set. Generation dominates the run and
shards over `--shard-index` and `--num-shards`.

**Command.**

```bash
bash scripts/reproduce/table3_worldarena.sh
```

**Expected numbers.** Overall is the equal-weight mean EWMScore-local-8:

| Model | Overall | MLR (%) |
|---|---|---|
| Standard SFT | 53.95 | 27.39 |
| **EVEWorld** | **56.76** | **13.38** |

Both models score 27.39% MLR with the released protocol but violate on
different clips, so the paired comparison uses the same 157-prompt eligible
set rather than the per-model subsets.

## Table 4: AgiBot transfer (EWMBench)

**Input.** The 777 AgiBot clips behind `AGIBOT_DATA_ROOT` and the AgiBot
fine-tuning split; the arm trains from the video-pretrain checkpoint for 50
steps at `480 x 640`.

**Compute.** The 50-step fine-tune plus 777 clips generated three times each
at generation seeds 42, 43 and 44 — 2,331 clips — followed by the official
EWMBench scoring over the `eval_layout` tree and an MLR pass.

**Command.**

```bash
bash scripts/reproduce/table4_ewmbench.sh
```

**Expected numbers.**

| Model | Motion | Semantics | DYN | HSD | nDTW | Scene | Overall |
|---|---|---|---|---|---|---|---|
| GigaWorld-0 (pretrained) | 50.30 | 2.2497 | 7.19 | 23.29 | 19.83 | **89.58** | 3.6486 |
| Standard SFT | 61.51 | 2.2477 | 14.88 | **24.35** | **22.28** | 84.38 | 3.7066 |
| **EVEWorld** | **63.65** | **2.2524** | **17.55** | 24.12 | 21.97 | 86.37 | **3.7525** |

The config and the scorer label the fourth component `hsr`; the paper's tables
call it HSD. The Semantics and Scene columns sit within a few thousandths of
each other across the two AgiBot-trained arms, so read them together with the
Motion, DYN and Overall columns.

## Table 5: RoboTwin transfer

**Input.** The 250 held-out RoboTwin episodes (50 tasks at episodes 45-49) and
the FlowWAM backbone. Post-training uses the 2,250 training episodes of
episodes 0-44, and the disjoint episode-45 dev slice screens the checkpoint
before the held-out run.

**Compute.** A rank-32 LoRA fine-tune for 1,128 steps (4 epochs) at 8 frames
per batch on 8 GPUs, then 250 clips per variant at 40 denoising steps and CFG
5.0 with the robot-only flow condition. PSNR, SSIM, LPIPS and Flow-EPE are
computed per clip, Flow-EPE through RAFT on both rollouts.

**Command.**

```bash
bash scripts/reproduce/table5_robotwin.sh
```

**Expected numbers.**

| Variant | PSNR (dB) | SSIM | LPIPS | Flow-EPE | MLR (%) |
|---|---|---|---|---|---|
| FlowWAM | 12.218 | 0.748 | 0.383 | 3.033 | 52.05 |
| Standard SFT | 11.687 | 0.735 | 0.414 | 2.828 | 47.89 |
| + IGR | 12.544 | 0.770 | 0.380 | 2.487 | 40.03 |
| + TIA | **13.445** | **0.776** | **0.333** | **1.850** | **22.54** |
| **EVEWorld** | 12.765 | 0.769 | 0.365 | 2.207 | 35.21 |

The wrapper writes the aggregate to `results/item_level/robotwin/table5.json`,
the same file the evaluator produces by hand.

## Table 6: Component ablation

**Input.** The DreamGenBench split and the same `U_63` eligible set as Table 1.
The four arms are standard SFT, IGR only, TIA only, and the joint EVEWorld
model, which is the arm Table 1 reports as well.

**Compute.** Three additional 250-step fine-tunes at the Table 1 recipe (the
joint arm is shared), 126 clips generated per arm, and Gemini-IF plus MLR over
every arm. The per-arm configs live in `configs/paper/` beside the main one.

**Command.**

```bash
bash scripts/reproduce/table6_ablation.sh
```

**Expected numbers.**

| Variant | IGR | TIA | Gemini-IF Env | Gemini-IF Object | Gemini-IF Behavior | Gemini-IF Overall | MLR (%) |
|---|---|---|---|---|---|---|---|
| Standard SFT | | | 51.72 | 42.00 | 67.02 | 53.57 | 11.11 |
| IGR only | yes | | 49.43 | 36.67 | **69.50** | 51.85 | 4.76 |
| TIA only | | yes | 55.17 | 42.67 | 68.79 | 55.29 | 7.94 |
| **EVEWorld** | yes | yes | **72.41** | **50.33** | 64.89 | **60.85** | **1.59** |

MLR falls with either component alone and falls further with both. Gemini-IF
moves the other way around: IGR alone costs overall instruction following
against standard SFT, and only the joint objective recovers it while keeping
the MLR gain.

## Figure 4: Classifier-free guidance

**Input.** The DreamGenBench split, the training run of the EVEWorld arm, and
the Gemini judge. The grid combines six training steps with four guidance
weights at a fixed generation seed of 4 and 30 inference steps.

**Compute.** 24 cells times 126 clips is 3,024 generations at `480 x 768`, 93
frames and 30 denoising steps, plus one Gemini judging pass per cell. The
output tree is written as
`<out_root>/step_%03d/cfg_<tag>/generated_only/<request_id>.mp4`, which is the
layout the aggregate reads; the tag replaces the decimal point of the guidance
weight with `p`.

**Command.**

```bash
bash scripts/reproduce/figure4_cfg.sh
```

**Expected behaviour.** Gemini-IF increases with the guidance weight across
the grid while MLR does not fall monotonically in the same direction: raising
CFG trades one failure mode for another. That is the reason the paper
supervises evolution explicitly instead of tuning guidance, and the
released numbers in the figure are monotone in Gemini-IF and non-monotone in
MLR for that reason. The grid config is
[`configs/ablations/cfg/grid.yaml`](../configs/ablations/cfg/grid.yaml).

## Additional studies

The remaining analyses are driven from `configs/ablations/` and summarised
under `experiments/analysis/`; they reuse the wrappers above for generation.

| Study | Configs | Reads |
|---|---|---|
| IGR corruption and weight-map design | `configs/ablations/igr/` | `p_dup.yaml`, `wmap_binary3x.yaml`, `wmap_legacy_multilevel.yaml`, `wmap_pathloc2x.yaml`, `wmap_pathloc3x.yaml`, `wmap_interaction2x.yaml`. |
| TIA layer, gate and window | `configs/ablations/tia/` | `layer_sweep.yaml`, `gate.yaml`, `warmup.yaml`, `window.yaml`, `temperature.yaml`, `gamma.yaml`. |
| Checkpoint and seed stability | `configs/ablations/checkpoint/` | `steps.yaml`, `seed_stability.yaml`. |
| MLR protocol sensitivity | `configs/ablations/` and the MLR evaluator | `tau_occ in {0.10, 0.15, 0.20, 0.25}` times `k in {1, 2, 3}`. |
| Horizon scaling | [`configs/eval/pbench.yaml`](../configs/eval/pbench.yaml) | Seven length tiers from 61 to 477 frames. |

`experiments/analysis/mlr_sensitivity/` regenerates the threshold table,
`experiments/analysis/tia_layer_probe/` the per-block end-point errors behind
the layer choice, and `experiments/analysis/seed_stability/` the seed-repeat
spread; the long-horizon study lives in `experiments/analysis/long_horizon/`.

## Protocol discipline

Detector thresholds, checkpoints, sampled timestamps and guidance weights were
frozen on development data before the reported runs. Keep the published
settings when comparing against a released number: an MLR value produced with
different thresholds, a different eligible set or a different number of
sampled timestamps is a different measurement. Every MLR and EWMBench run
records the settings that produced it alongside the item-level output, so the
provenance of a number is recoverable from `results/item_level/<benchmark>/`.

Generation is stochastic, so a re-run reproduces the protocol and the
qualitative ordering rather than the exact decimals; the reported single-clip
differences such as 1.59% versus 2.38% MLR are inside the Wilson interval
printed with every summary.
