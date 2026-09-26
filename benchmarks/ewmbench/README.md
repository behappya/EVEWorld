# EWMBench Evaluation (AgiBot Transfer)

Generation, official evaluation, and training-data preparation for the EWMBench transfer experiment, plus a supplementary WorldModelBench (WMB) robotics campaign.

## Paper Mapping

- **Experiments → "Generalization Across Domains and Training Data"** (Table `tab:agibot_transfer`): Standard SFT and EVEWorld are trained on the same 777-clip AgiBot training set under the same budget and evaluated on EWMBench. Full per-component scores are in **Appendix → "Complete EWMBench Results"** (Table `tab:agibot_transfer_full`).
- **Appendix → "Implementation Details"**: AgiBot training runs 50 steps at 480×640 with 93-frame clips at 16 FPS; generation uses the raw step-50 checkpoint, three benchmark seeds (42/43/44), 30 denoising steps, and CFG 7.0 — matching the defaults in `kjob_ewmbench_8gpu_serial.sh`.
- The `kjob_agibot_*` scripts prepare the AgiBot training data (detection, IGR copy-paste assets, packing) consumed by the training recipe in [`eveworld/agibot/`](../../eveworld/agibot/).
- The `kjob_wmb_*` / `wmb_crop_right.py` scripts are a supplementary WorldModelBench robotics-subset campaign; they are **not** part of the paper's reported tables.

## Contents

| File | What it does |
|---|---|
| `kjob_ewmbench_8gpu_serial.sh` | EWMBench generation: for each model × seeds {42,43,44} × 21 episodes on one 8-GPU node via [`giga-world-0/scripts/inference.py`](../../giga-world-0/scripts/inference.py). Writes side-by-side mp4 (left input \| right generation, 1296×480). Skips model/seed pairs that already have `generation_summary.json`. |
| `ewmbench_layout_convert.py` | Converts side-by-side mp4s into the official EWMBench evaluation layout: crops the right (generated) half to 640×480 and dumps `frame_%05d.jpg` under `eval_layout/<model>_dataset/<task>/<episode>/<1\|2\|3>/video/` (seed42→1, seed43→2, seed44→3). |
| `kjob_ewmbench_eval_serial.sh` | Official EWMBench chain against an external EWMBench checkout (`EWM_DIR`): `processing/video_resize.py`, `processing/detection_tracking.py` (YOLO trajectories; ground truth detected once and cached), then `evaluate.py` over `scene_consistency trajectory_consistency semantics diversity`. |
| `kjob_agibot_detect.sh` | GroundingDINO entity + dual-arm gripper detection over the 777 AgiBot training clips (8-GPU shards via [`eveworld/agibot/agi_detect_dispatch.py`](../../eveworld/agibot/agi_detect_dispatch.py)). |
| `kjob_agibot_aug_prep.sh` | Builds IGR copy-paste augmentation assets (`.npz`: patch/box/zones, with the dual-arm gripper corridor excluded from paste zones) via [`eveworld/agibot/agi_aug_prep_dispatch.py`](../../eveworld/agibot/agi_aug_prep_dispatch.py). |
| `kjob_agibot_pack.sh` | Packs the final AgiBot training videos (with text-encoder embeddings) via [`giga-world-0/scripts/pack_data.py`](../../giga-world-0/scripts/pack_data.py). |
| `kjob_wmb_detect.sh` | GroundingDINO detection for the WMB adaptation trainset (500 clips) via [`eveworld/agibot/w4_dispatch.py`](../../eveworld/agibot/w4_dispatch.py). |
| `kjob_wmb_apre_train.sh` | WMB-adaptation EVEWorld-recipe training via [`giga-world-0/scripts/train.py`](../../giga-world-0/scripts/train.py) on a prebuilt runtime config ([`eveworld/pipeline/t4g_joint_trainer.py`](../../eveworld/pipeline/t4g_joint_trainer.py)); tops up missing aug assets first via [`eveworld/agibot/w6_dispatch.py`](../../eveworld/agibot/w6_dispatch.py). Requires `T4G_W_LAT=40`/`T4G_WPIX=640` (exported in the script) for the 640-wide grid. |
| `kjob_wmb_robotics_8gpu_serial.sh` | Generation on the WorldModelBench robotics subset (50 prompts); outputs one mp4 per first-frame name, as the WMB judge expects. |
| `wmb_crop_right.py` | Crops WMB side-by-side mp4s to generated-only 640×480 @ 16 FPS videos under `eval_videos/<model>/`. |
| `kjob_wmb_judge_serial.sh` | Scores the 50 robotics videos per model with the VILA judge using the external WorldModelBench checkout (`WMB_DIR`) `evaluation.py`. |

## Usage

All `kjob_*.sh` files are **SLURM-style cluster wrappers** (`#SBATCH` headers; 8-GPU nodes except the single-GPU packing job) whose defaults point at cluster-specific absolute paths (data roots, conda installations, external benchmark checkouts). Adapt them to your environment before submitting; every script also accepts `KEY=VALUE` overrides as positional arguments, e.g.:

```bash
export EVEWORLD_ROOT=/path/to/EVEWorld   # repo root; used for PYTHONPATH and script locations

# 1. AgiBot training-data preparation (detection -> IGR assets -> packing)
sbatch kjob_agibot_detect.sh
sbatch kjob_agibot_aug_prep.sh
sbatch kjob_agibot_pack.sh

# 2. Train Standard SFT / EVEWorld on AgiBot and assemble probe checkpoints
#    (launcher and probe packaging live in eveworld/agibot/, see Notes)

# 3. Generate EWMBench videos (3 seeds x 21 episodes per model)
sbatch kjob_ewmbench_8gpu_serial.sh \
  MODELS="pretrain ewm_vanilla_s50 ewm_apre_wmaponly_s50"

# 4. Convert to the official frame-sequence layout
python ewmbench_layout_convert.py \
  --models pretrain ewm_vanilla_s50 ewm_apre_wmaponly_s50

# 5. Run the official EWMBench evaluation
sbatch kjob_ewmbench_eval_serial.sh \
  MODELS="pretrain ewm_vanilla_s50 ewm_apre_wmaponly_s50"

# 6. (Supplementary) WMB robotics campaign
sbatch kjob_wmb_detect.sh
sbatch kjob_wmb_apre_train.sh
sbatch kjob_wmb_robotics_8gpu_serial.sh MODELS="pretrain wmb_vanilla_s100 wmb_apre_s600"
python wmb_crop_right.py --models pretrain wmb_vanilla_s100 wmb_apre_s600
sbatch kjob_wmb_judge_serial.sh MODELS="pretrain wmb_vanilla_s100 wmb_apre_s600"
```

Model names are resolved by the `model_dir_for` case table inside the generation scripts: `pretrain` is the GigaWorld-0 base model, `ewm_vanilla_*` / `ewm_apre_*` are AgiBot Standard SFT / EVEWorld probes, and `wmb_*` are WMB-adaptation probes. Probe directories are assembled from raw checkpoints with [`eveworld/agibot/agi_make_probe.py`](../../eveworld/agibot/agi_make_probe.py) (AgiBot line) and [`eveworld/agibot/w7_make_probe.py`](../../eveworld/agibot/w7_make_probe.py) (WMB line).

## Notes

- **Environment variables.** `EVEWORLD_ROOT` (repository root) must be set. `GAGI` is the data/output root used throughout these scripts; `OUTPUT_ROOT`, `DATA_PATH`, `SAVE_ROOT`, `LAYOUT_ROOT`, `VIDEO_ROOT` override individual stages. See `eveworld/common/env.sh` and [`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md) for the shared environment contract.
- **External checkouts.** The official [EWMBench](https://github.com/AgibotTech/EWMBench) and WorldModelBench repositories are not vendored; clone them separately and point `EWM_DIR` / `WMB_DIR` at them. The EWMBench `semantics` dimension captions videos through a locally served OpenAI-compatible VLM endpoint (`CAPTION_API_MODEL`, referenced from the eval config).
- **Conda environments.** Generation/packing use the main environment (`CONDA_ENV`, default `EVEWorld`); GroundingDINO detection and aug-asset prep use `giga_world1`; the official EWMBench eval and the VILA judge use their own environments (`EWMBench`, `vila`). All are overridable via `CONDA_ENV`.
- **Data prerequisites.** EWMBench prompt/first-frame json (`ewmbench_it2v.json`), the 777-clip AgiBot training set (disjoint from EWMBench episodes), WMB `trainset_v1` plus the 50 robotics first frames, and the VILA judge checkpoint.
- **Idempotency.** Generation skips model/seed pairs with an existing `generation_summary.json`; the layout converter skips episodes already at the expected 93 frames; EWMBench ground-truth trajectories are detected once (sentinel file under `SAVE_ROOT`) and reused across models. Pass `OVERWRITE=1` to the eval script to recompute only the dimensions listed in `DIMENSIONS`.
- Training launchers, configs, and the trainer itself live in [`eveworld/agibot/`](../../eveworld/agibot/) (`agi_joint_launch.sh`) and [`eveworld/pipeline/`](../../eveworld/pipeline/); this directory contains only the benchmark-facing generation/evaluation/data-prep wrappers.
