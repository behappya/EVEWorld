# EVEWorld × FlowWAM Port

Port of the EVEWorld components — Instance-Guided Restoration (IGR) and Temporal Instance Alignment (TIA) — onto the **FlowWAM** backbone (Wan2.2-TI2V-5B dual-stream, optical-flow-conditioned), trained on the 50-task RoboTwin dataset and evaluated on held-out RoboTwin actions.

## Paper Mapping

This package implements the cross-backbone transfer experiment of the paper:

- **§4.1 Instance-Guided Restoration** — duplicate-paste corruption (`igr_paste.py`, Eq. `eq:paste`), static spatial weight maps (`weightmap_640.py`), and the restore-the-clean-video objective with up-weighted corrupted region (`eve_flowwam_train.py`, Eq. `eq:pipeline`).
- **§4.2 Temporal Instance Alignment** — offline layer probe (`tia_probe.py`, Eq. `eq:epe`), zero-initialized cross-frame feature transport and the contrastive correspondence loss (`tia_adapter.py`, Eq. `eq:cic`).
- **§5.3 Generalization Across Domains and Training Data**, Table `tab:flowwam_transfer` — the five training arms of `eve_flowwam_train.py` (`control`, `sft`, `igr`, `tia`, `eve`) produce the five table rows; `arm_generate.py` produces the evaluated videos (PSNR/SSIM/LPIPS/Flow-EPE + MLR).
- **Appendix, TIA Layer Selection** (`app:tia_layer_selection`, Table `tab:tia_layer_probe`) — the probe selects block ℓ\* = 12 on FlowWAM.
- **Appendix, Implementation Details** (`app:training_setup`) — FlowWAM configuration: rank-32 LoRA, 29-frame clips (24 FPS, 480×640), AdamW lr 1e-4, TIA rank 64, 7×7 window, τ = 0.07, γ = 0.1, λ_TIA = 0.1 at block 12.
- **Appendix, FlowWAM across Checkpoints** (Table `tab:flowwam_action_fidelity`) — checkpoint-screen generation also goes through `arm_generate.py`.

The GigaWorld-0 originals of this code live in [`eveworld/pipeline/`](../pipeline/) (IGR paste, GDINO annotation) and [`eveworld/tia_transport/cic_transport_transformer.py`](../tia_transport/cic_transport_transformer.py) (TIA math, mirrored here line-for-line minus the sequence-parallel guards).

## Contents

| File | What it does |
|---|---|
| `eve_flowwam_train.py` | Controlled-variant training entry point. `--arm {control,sft,igr,tia,eve}` switches IGR paste/weighting and TIA on/off; denoising target is always the **clean** latent (restoration); rank-32 LoRA on the DiT plus `flow_stream` and (for `tia`/`eve`) the TIA adapter; per-GPU batch 1, manual gradient all-reduce. |
| `eve_flowwam_dataset.py` | Window dataset. Each sample is a `t_lat_win=8` latent window (29 pixel frames): clean frames, on-the-fly IGR-corrupted frames, weight window, RAFT flow-codec video (frame 0 white, FlowWAM recipe), prompt, and TIA target cells. Deterministic per-task held-out split (last `--heldout-per-task` episodes per task are excluded from training). |
| `tia_adapter.py` | `TIAAdapter` (rank-64 low-rank projection, adjacent-frame 7×7 local matching, attention transport, zero-init `output_proj` + tanh residual), `TIAInjection` (monkey-patches `_dual_stream_block_fn` to fire after block ℓ\* and capture post-transport features), `tia_infonce_loss` (Eq. `eq:cic`). |
| `tia_probe.py` | Offline per-block correspondence probe (Eq. `eq:epe`): chained target tracking on captured RGB tokens vs. GDINO ground truth, aggregated per (block, noise level); supports `--shard i/n`. |
| `tia_probe_dispatch.py` | Fans `tia_probe.py` out over `N_GPU` GPUs and aggregates per-block EPE / moving-pair hit rate into `probe_summary.json`, marking the best block (ℓ\*). |
| `igr_paste.py` | Window-level IGR paste (Eq. `eq:paste`): crop the target patch from the window's first annotated frame, sample a paste plan inside the safe zone (static background, non-arm cells), alpha-blend with feathered edges, bump the weight window to `W_PASTE=6.0`. Falls back to the clean sample when no plan is feasible. |
| `weightmap_640.py` | Precomputes the static part of the spatial weight map (31×30×40 latent geometry): `W_BG=0.5` baseline, `W_OBJ=4.0` on the dilated target trajectory, nearest-frame fill for detection holes. |
| `gdino_annotate.py` | Per-latent-frame GroundingDINO localization of the target (`target_name`) and the robot arm → one annotation JSON per episode; shared input for weight maps, paste plans, TIA cells, and the probe. Supports `--shard i/n` and resume. |
| `gdino_dispatch.py` | Fans `gdino_annotate.py` out over `N_GPU` GPUs (one shard per GPU). |
| `arm_generate.py` | Manifest-driven generation from a trained arm checkpoint: base → Stage-1 → arm delta (LoRA merge + `flow_stream`/`tia_adapter` load); first-frame + instruction conditioning; `--flow-cond robot_only` teacher-forces the clean robot-only flow; `--full-traj on|direct` covers full-length trajectories via overlapping chunks with crossfaded seams. `--shard i/n` for multi-process campaigns. |
| `dual_stream_cond.py` | Import-time codegen of the upstream dual-stream `model_fn` with the flow-stream timestep pinned to 0 (clean-flow teacher forcing, same mechanism as the first-frame prefix). Asserts if the upstream source drifts. |
| `build_manifest.py` | Scans the extracted 640×480 RoboTwin dataset (`<task>/aloha-agilex_clean_50/`) into the training manifest: video, robot-only video, HDF5, instruction, target asset/name, arm. |
| `build_flowwam_heldout_manifest.py` | Builds the 50-task × episodes 45–49 = 250-row held-out fidelity manifest (with first-frame PNGs) used for the Table `tab:flowwam_transfer` evaluation. |
| `postprocess_pc.py` | Post-processing probe for the photometric-consistency metric (`repeat2`, `ema`); analysis tooling, not part of training or reported results. |
| `dist_smoke.py` | Minimal multi-rank NCCL `all_reduce` smoke test for validating the distributed environment before training. |

## Usage

The typical pipeline order (per-GPU shard counts assume an 8-GPU node):

```bash
# 0. Environment: a conda env with FlowWAM's deps + GroundingDINO; FLOWWAM_ROOT
#    points to a local FlowWAM checkout (provides training/, inference/,
#    diffsynth, RAFT/flow-codec utilities).
export EVEWORLD_ROOT=/path/to/EVEWorld
export FLOWWAM_ROOT=/path/to/FlowWAM

# 1. Training manifest from the extracted RoboTwin 640 dataset
python eveworld/flowwam_port/build_manifest.py \
  --root <robotwin_640_extracted> --output <workdir>/manifest_640.json

# 2. GDINO annotations (8-way shard), then static weight maps
N_GPU=8 python eveworld/flowwam_port/gdino_dispatch.py \
  --manifest <workdir>/manifest_640.json --out-dir <workdir>/anno_640
python eveworld/flowwam_port/weightmap_640.py   # edit ANNO_DIR/OUT_DIR or symlink

# 3. TIA layer probe -> probe_summary.json (selects l*, 12 for FlowWAM)
N_GPU=8 python eveworld/flowwam_port/tia_probe_dispatch.py --per-task 3

# 4. Train one arm (repeat per arm: control | sft | igr | tia | eve)
accelerate launch --num_processes 8 --mixed_precision bf16 \
  eveworld/flowwam_port/eve_flowwam_train.py \
  --arm eve --output-path <run_dir> --models-root <wan2.2_ti2v_5b_root> \
  --resume-checkpoint <flowwam_stage1.safetensors> --l-star 12

# 5. Held-out manifest + generation (Table flowwam_transfer protocol:
#    robot-only flow condition, CFG 5.0, 40 steps, seed 42, full trajectory)
python eveworld/flowwam_port/build_flowwam_heldout_manifest.py \
  --data-root <robotwin_640_extracted> --out <eval_dir>/manifest.json
python eveworld/flowwam_port/arm_generate.py \
  --arm-ckpt <run_dir>/final.safetensors --stage1 <flowwam_stage1.safetensors> \
  --manifest <eval_dir>/manifest.json --out <eval_dir>/videos \
  --flow-cond robot_only --cfg-scale 5.0 --steps 40 --full-traj direct --shard 0/8
```

Fidelity metrics (PSNR/SSIM/LPIPS/Flow-EPE) and MLR on the generated videos are computed by the evaluation pipeline in [`benchmarks/robotwin_flowwam/`](../../benchmarks/robotwin_flowwam/).

**Cluster-job caveat.** The `kjob_*` / `launch_*` scripts under [`benchmarks/robotwin_flowwam/`](../../benchmarks/robotwin_flowwam/) are SLURM-style wrappers around the entry points above; their absolute paths (dataset roots, conda paths, checkpoint locations) are cluster-specific and must be adapted before reuse. They also pin `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`; verify collectives with `dist_smoke.py` first.

## Notes

- **Backbone dependency.** This package is *not* self-contained: it imports the FlowWAM codebase (`flow_action_train`, `diffsynth` dual-stream pipeline, `pipeline_loader`, RAFT flow extractor / reversible flow codec) from `FLOWWAM_ROOT`. The GigaWorld-0 vendored snapshots in this repo are not used here.
- **Model prerequisites.** Wan2.2-TI2V-5B weights (`--models-root`) and the released FlowWAM Stage-1 checkpoint (`--resume-checkpoint` for training, `--stage1` for generation).
- **Working directory.** The tokenizer is resolved through a relative `models/` path, so training/generation must run from a cwd containing a `models -> <flowwam models>` symlink (the cluster wrapper `cd`s into `$FLOWWAM_ROOT/training` for this reason).
- **GDINO import path.** `gdino_annotate.py` inserts `../pipeline` on `sys.path` to import `t4g_gdino.GDinoLocator` from [`eveworld/pipeline/t4g_gdino.py`](../pipeline/t4g_gdino.py); run it from `eveworld/flowwam_port/` (or with the repository root on `PYTHONPATH`).
- **Data prerequisites.** Extracted RoboTwin 640×480 episodes with `video/`, `robot_only/video/head_camera/`, `data/*.hdf5`, `instructions/*.json`, and `scene_info.json` per task; GroundingDINO weights for annotation; GPU RAFT for the flow codec (lazy-initialized per dataset worker).
- **Dispatch helper.** Cluster wrappers fan generation out via `arm_generate_dispatch.py`; single-process runs can call `arm_generate.py --shard i/n` directly (it is idempotent per output video).
- Training is video-window based with per-GPU batch 1; gradients are all-reduced manually instead of DDP because the mixed CPU/GPU module placement (offloaded T5/VAE) breaks DDP verification on the reference cluster.
