# End-to-End Reproduction Guide

This guide walks through reproducing the paper's main results. Each stage has
a canonical entry point; cluster (`kjob_*`) wrappers are SLURM-style templates
— set `EVEWORLD_ROOT` (this repo) and `GAGI_ROOT` (data root) first and adapt
site-specific paths. See `docs/ENVIRONMENT.md` for the software stack and
`docs/DATASETS.md` for data acquisition.

## 0. Setup

```bash
conda env create -f environment.yml && conda activate EVEWorld
pip install -e ./giga-models
export EVEWORLD_ROOT=$(pwd)
export GAGI_ROOT=/path/to/your/data
```

Download the GigaWorld-0 Video-Pretrain-2B backbone and the GR1 fine-tuning
split (scripts under `giga-world-0/scripts/download_*.sh`), and prepare
GroundingDINO weights for the detection stages.

## 1. IGR supervision assets (offline)

Build the IGR training pairs: parse instructions, localize the target with
GroundingDINO, sample paste regions, and precompute the spatial weight maps.

- Detection and annotation: `eveworld/pipeline/t4g_detect.py`, `t4g_gdino.py`
- Copy-paste corruption and weight maps: `eveworld/pipeline/t4g_aug_prep.py`, `t4g_weightmap_precompute.py`
- Alternative weight-map designs (legacy multi-level, binary, path+loc, interaction; side-by-side configs, no preferred default): `eveworld/pipeline/t4g_weightmap_variants.py` + `t4g_weightmap_<variant>_config.py` / `_launch.sh`, driven by `t4g_weightmap_variant_matrix.sh`
- Details: `eveworld/pipeline/README.md`

## 2. TIA layer selection (offline probe)

Run the layer-wise correspondence probe (endpoint error per block) to select
the TIA attachment layer (blocks 22-23 on GigaWorld-0; block 12 on FlowWAM):

- `eveworld/pipeline/t4g_probe.py`, `t4g_measure.py`
- FlowWAM variant: `eveworld/flowwam_port/tia_probe.py`

## 3. Train EVEWorld (IGR + TIA)

The joint objective (IGR restoration + TIA correspondence) is implemented by
`eveworld/pipeline/t4g_joint_trainer.py`. The reported main run is the
"A-pre" recipe at training seed 42: `eveworld/pipeline/t4g_cfg_repro_seed42_config.py`
(deriving from `t4g_apre_noaug_config.py`, with `p_aug=0.5`), launched via
`benchmarks/dreamgenbench/kjob_cfg_train_then_sweep.sh` and evaluated at the
step-250 checkpoint:

```bash
bash benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
  BASE_CONFIG_MODULE=eveworld.pipeline.t4g_cfg_repro_seed42_config MAX_STEPS=250
```

Main-run recipe (paper, Appendix "Implementation Details"): full-parameter
training from GigaWorld-0 Video-Pretrain-2B, CAME-8bit, lr 2^-14.5
(~4.32e-5), effective batch 64, 250 steps, 480x768, 93-frame clips @16 FPS,
on 92 GR1 videos; TIA lambda warms 0->0.5 over the first 20 steps and applies
at sigma in [0.2, 0.5] (all visible in the config). Controls: Standard SFT
(uniform loss) and the component ablations (`t4g_ablation_*` configs).
`t4g_joint_cleanv2_config.py` is the uniform-3x weight-map sibling variant;
`eveworld/pipeline/t4g_final_*` is a later exploratory arm, not the reported
main configuration.

## 4. Generate evaluation videos

- DreamGenBench (126 prompts, seed 004, 30 denoising steps, CFG 7.0):
  `benchmarks/dreamgenbench/` generation chains (`kjob_gr1_dreamgen_generation.sh`, eval175 campaigns).
- WorldArena 1.0 / EWMBench / RoboTwin: the per-benchmark `kjob_*` generation scripts under `benchmarks/`.

## 5. Evaluate

- **MLR** (deterministic GroundingDINO protocol, plus the SAM2 occlusion check of Appendix `app:mlr_protocol`): `benchmarks/worldarena/mlr_eval.py` + `mlr_dispatch.py`, with the occlusion evidence in `mlr_occlusion.py` and the runnable presets in `benchmarks/worldarena/mlr_protocol_profiles.yaml`. The default profile `worldarena1_mlr_gdino_v2` reproduces the frozen WorldArena 1.0 rule; the appendix's occlusion-aware Algorithm 1 is `appendix_alg1_sam2_occlusion` (needs SAM2, `--occlusion-rule paper_overlap`). Per-benchmark drivers are linked from each benchmark README.
- **DreamGenBench IF**: Qwen-IF (`benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.py`), Gemini-IF (`eveworld/evaluation/eval_gemini_dreamgen_qwen_protocol.py`); aggregate with `eval175_summarize.py`.
- **WorldArena 1.0**: eight local metrics (`benchmarks/worldarena/local_metric_eval.py`) aggregated by `aggregate_core_scores.py`.
- **EWMBench**: official-layout conversion + scoring (`benchmarks/ewmbench/`).
- **RoboTwin/FlowWAM**: PSNR/SSIM/LPIPS/Flow-EPE + MLR (`benchmarks/robotwin_flowwam/`).

## 6. Ablations and analyses

- Component ablation (IGR-only / TIA-only / joint): `eveworld/pipeline/t4g_ablation_*` configs + `kjob_eve_ablation_*` chains.
- CFG sensitivity grid: `eveworld/evaluation/cfg_grid_*`, default grid `{1.0, 2.5, 5.0, 7.0}` (override with `--cfg-values` on `build_cfg_gemini_manifest.py` or `CFG_VALUES` in the `kjob_cfg_*` wrappers).
- Sequence-length scaling: `benchmarks/pbench/` and `benchmarks/dreamgenbench/run_*length*` scripts.
- Alternative designs (PhysicsLatent, EAG, LAD-LoRA, Causal Frontier, ICH-D): `eveworld/alternatives/README.md`.

## Notes

- Detector thresholds, checkpoints, and guidance settings were frozen on
  development data before final evaluation; keep the published protocol
  unchanged when comparing numbers. Each MLR entry in
  `benchmarks/worldarena/mlr_protocol_profiles.yaml` is such a frozen,
  self-contained set — record which profile produced a given number.
- Judge endpoints require API credentials via environment variables
  (`OPENAI_API_KEY`, `DIFROST_*`); see `docs/ENVIRONMENT.md`.
