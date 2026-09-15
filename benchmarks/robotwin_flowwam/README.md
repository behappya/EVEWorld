# RoboTwin Held-Out Evaluation with the FlowWAM Backbone

Cross-backbone transfer experiment: EVEWorld's IGR and TIA components are retrained on the
external **FlowWAM** backbone (Wan2.2-TI2V-5B based) and evaluated on held-out RoboTwin
episodes with PSNR / SSIM / LPIPS / Flow-EPE fidelity metrics plus MLR.

## Paper mapping

- **Table 4 (`tab:flowwam_transfer`)**, Section 4.3 "Cross-Backbone Generalization under
  Held-Out Actions": the five-row held-out table (FlowWAM Stage-1, Standard SFT, + IGR,
  + TIA, EVEWorld) — 250 videos per row from episodes 45–49 of each of 50 RoboTwin tasks,
  robot-only head-camera flow condition, seed 42, CFG 5.0, 40 denoising steps,
  full-trajectory 24 FPS generation at 480×640, TIA inference-time injection disabled.
- **Appendix `app:training_setup`**: both arms initialize from the released FlowWAM Stage-1
  checkpoint and train a rank-32 LoRA on 2,250 episodes (tasks × episodes 0–44) for four
  epochs (1,128 steps) with AdamW (lr 1e-4, weight decay 0.01); FlowWAM attaches TIA at
  block 12 with λ_TIA = 0.1 and no noise gating.
- **Appendix `app:flowwam_benchmarks` (`tab:flowwam_action_fidelity`)**: checkpoint
  screening on the fixed episode-45 slice (50 development videos) before the main
  250-episode comparison.
- **Appendix `app:tia_layer_selection` (`tab:tia_layer_probe`)**: the offline layer-wise
  correspondence probe that selects block 12 for FlowWAM (`kjob_flowwam_tia_probe.sh`).

The method-side port (training loop, IGR paste, TIA adapter, generation, manifest builder,
probes) lives in [`eveworld/flowwam_port/`](../../eveworld/flowwam_port/); this directory
holds the benchmark-side orchestration and metric code.

## Contents

| File | What it does |
|---|---|
| `kjob_eve_flowwam_train.sh` | Trains one or more arms (`ARMS="control eve"`, serially) via `eveworld/flowwam_port/eve_flowwam_train.py` under `accelerate` (bf16, 8 GPUs). |
| `kjob_flowwam_arm_generate.sh` | Generates videos from trained arm checkpoints over a manifest (`CKPT_NAME`, `FLOW_COND`, `CFG_SCALE`, `GEN_STEPS`, `FULL_TRAJ`, `TIA_INJECT`, seed). |
| `kjob_flowwam_heldout_serial.sh` | End-to-end held-out run: builds the 50-task × episodes 45–49 manifest (`eveworld/flowwam_port/build_flowwam_heldout_manifest.py`) and generates the `control` and `eve` arms with the paper protocol (robot_only flow, CFG 5.0, 40 steps, seed 42, direct full trajectory). |
| `flowwam_fidelity_metrics.py` | Frame-aligned PSNR (all frames) and SSIM (sampled) against ground-truth videos; expects exactly 250 manifest rows. |
| `kjob_flowwam_lpips_epe.sh` → `flowwam_lpips_flow_epe.py` | LPIPS (AlexNet) and RAFT optical-flow EPE, sharded over 8 GPUs (`--worker` mode per shard). |
| `kjob_flowwam_five_row_node.sh` | Two-node campaign (submit with `ROLE=A` and `ROLE=B`) that trains/generates/evaluates the five Table-4 variants with smoke tests, media audits, and per-stage status markers; MLR runs through [`benchmarks/worldarena/mlr_flowwam_variants.py`](../worldarena/). |
| `flowwam_five_row_aggregate.py` | Aggregates audited PSNR/SSIM, LPIPS/Flow-EPE, and MLR summaries into the frozen five-row table (`aggregate/table5.json` / `table5.csv`); fails on any coverage violation. |
| `build_flowwam_checkpoint_screen_manifest.py` | Splits the 250-row manifest into a 50-video development set (episode 45) and a 200-video confirmation set. |
| `kjob_flowwam_checkpoint_screen.sh` | Screens candidate checkpoints (ROLE=A/B split) on the dev slice: generate 50 videos, score PSNR/SSIM. |
| `aggregate_flowwam_checkpoint_screen.py` | Ranks screened checkpoints by PSNR/SSIM (`dev50_ranking.{json,csv}`). |
| `kjob_flowwam_archive_candidate_screen.sh` | Re-scores one archived checkpoint (`TAG=`, `CHECKPOINT=`) on the same dev slice. |
| `kjob_mlr_cfg_flowwam.sh` | Runs the frozen GroundingDINO MLR protocol (24 sampled timestamps) over the CFG-grid and FlowWAM held-out outputs via [`benchmarks/worldarena/mlr_cfg_flowwam_dispatch.py`](../worldarena/). |
| `kjob_flowwam_gdino_anno.sh` | GroundingDINO target annotation of the training corpus (IGR supervision input) via `eveworld/flowwam_port/gdino_dispatch.py`. |
| `kjob_flowwam_tia_probe.sh` | 8-GPU layer-wise TIA correspondence probe (`eveworld/flowwam_port/tia_probe_dispatch.py`). |
| `kjob_dist_smoke.sh` | `torchrun` distributed smoke test for the `flowwam` environment. |
| `flowwam_campaign_checks.py` | Coverage/integrity gates used by the five-row campaign (manifest, protocol, training args, PSNR/LPIPS/MLR summaries, eligible-set hash). |
| `audit_flowwam_variant.py` | Validates one variant's video coverage and media integrity (640×480) against the manifest. |
| `audit_cfg_flowwam_outputs.py` | PyAV-based audit of the CFG-grid cells and the FlowWAM held-out videos (resolution, fps, frame counts). |

## Usage

All `kjob_*` scripts are SLURM-style cluster wrappers (8×GPU nodes) whose absolute paths —
`FLOWWAM_ROOT`, conda installation, data and output roots — are site-specific defaults that
**must be adapted to your cluster** before submission. Every wrapper accepts `KEY=VALUE`
arguments, which it exports before running.

Environment contract: `EVEWORLD_ROOT` points to this repository root and `GAGI_ROOT` to the
shared data root (see [`eveworld/common/env.sh`](../../eveworld/common/env.sh) and
[`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md)). The `flowwam` conda env runs
training/generation; the `giga_models` env runs the metric stack.

Typical sequence:

```bash
# 1. Train the matched control and the EVEWorld arm (FlowWAM port, LoRA r=32).
sbatch benchmarks/robotwin_flowwam/kjob_eve_flowwam_train.sh \
  ARMS="control eve" RUN_TAG=arm

# 2. Build the held-out manifest and generate both arms (250 videos each).
sbatch benchmarks/robotwin_flowwam/kjob_flowwam_heldout_serial.sh

# 3. Fidelity metrics against the aligned ground truth.
python benchmarks/robotwin_flowwam/flowwam_fidelity_metrics.py \
  --manifest "$OUT_ROOT/manifest.json" --flow-root "$OUT_ROOT" \
  --output "$OUT_ROOT/fidelity_psnr_ssim.json"
sbatch benchmarks/robotwin_flowwam/kjob_flowwam_lpips_epe.sh

# 4. MLR on the FlowWAM outputs (shares the WorldArena 1.0 detector protocol).
sbatch benchmarks/robotwin_flowwam/kjob_mlr_cfg_flowwam.sh
```

For the full five-row ablation, submit `kjob_flowwam_five_row_node.sh` twice
(`ROLE=A`, `ROLE=B`) and then run `flowwam_five_row_aggregate.py`. For checkpoint selection,
run `kjob_flowwam_checkpoint_screen.sh` (both roles) followed by
`aggregate_flowwam_checkpoint_screen.py`; the dev/test split keeps selection disjoint from
the final 250-episode evaluation.

## Notes

- **External backbone.** FlowWAM itself is *not* vendored into this repository. Set
  `FLOWWAM_ROOT` to a FlowWAM checkout (training and inference entry points are imported
  from it), and obtain the released FlowWAM Stage-1 checkpoint and the Wan2.2-TI2V-5B base
  weights. Generation and metric code add `${FLOWWAM_ROOT}` and `${FLOWWAM_ROOT}/inference`
  to `PYTHONPATH`.
- **Data prerequisites.** Extracted RoboTwin episodes at 640×480 / 24 FPS (50 tasks;
  episodes 0–44 for training, 45–49 held out), plus GroundingDINO annotations for the IGR
  training data (`kjob_flowwam_gdino_anno.sh`).
- **Dependencies.** `opencv-python`, `scikit-image`, `numpy` (PSNR/SSIM); `lpips`, `torch`,
  `av`, and FlowWAM's RAFT wrapper (`raft_flow_extractor`) for the perceptual/flow metrics;
  GroundingDINO for MLR. Distributed runs pin `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`.
- **Determinism gates.** Metric scripts fail closed: 250 unique manifest rows, zero
  truncated videos, and a hashed MLR eligible set are enforced before numbers are reported.
