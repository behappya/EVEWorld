# EVEWorld Pipeline (`eveworld/pipeline/`)

End-to-end evolution-supervision pipeline of the EVEWorld paper: GroundingDINO annotation of the
GR1 training videos, offline layer selection and evidence probes, IGR copy-paste corruption and
spatial weight maps, TIA correspondence training, the joint/final trainers, and the deterministic
MLR detection-and-counting instrument. Modules are prefixed `t4g_` (Track4Gen-style correspondence
supervision, arXiv 2412.06016) and run on top of the vendored [GigaWorld-0](../../giga-world-0/)
backbone and [GigaModels](../../giga-models/) training stack.

## Paper mapping

| Code group | Paper location |
|---|---|
| `t4g_aug_*` (corruption + restoration trainer) | Method, *Instance-Guided Restoration* (IGR); App. *IGR Sample Construction* |
| `t4g_weightmap_*` (regional weight maps) | IGR spatial up-weighting; App. figure *Regional weights for the GR1 configuration* |
| `t4g_corr_*` (contrastive correspondence loss/trainer) | Method, *Temporal Instance Alignment* (TIA) |
| `t4g_probe.py`, `t4g_measure.py` (layer-wise probe) | App. *TIA Layer Selection* (layer-probe table) |
| `t4g_joint_*` (IGR + TIA trainer/configs) | Main EVEWorld model (main run: `t4g_cfg_repro_seed42_config.py` via `kjob_cfg_train_then_sweep.sh`); App. *Training Objective* |
| `t4g_final_*` (final combined arm) | Later exploratory arm (not the reported main configuration) and its audit probes |
| `t4g_detect.py`, `t4g_gdino.py`, `t4g_exam*.py`, `ewm_dup.py` | MLR metric; App. *MLR Evaluation Protocol* and *Detector Calibration*; EWMBench transfer |
| `t4g_ablation_*_config.py`, `kjob_eve_ablation_*`, `watch_eve_ablation_*` | Table *Component ablation on DreamGenBench* |
| `t4g_ghost_probe.py`, `t4g_restore_probe.py`, `t4g_change_test.py`, `t4g_dup_test.py`, `t4g_empty_map.py` | Evidence chain behind the design; App. *Why Standard SFT becomes lazy and how IGR removes it* |
| `selfcase_*`, `t4g_selfcase_*` | Self-case mining; App. qualitative case figures |
| `t4g_ich_*` | App. *Comparison with failed alternatives* — ICH-D route (negative result) |
| `failure_routes_reval.py`, `kjob_failure_routes_mlr.sh` | Uniform re-evaluation manifests for the five failed alternatives (the other four live in [`eveworld/method/`](../method/)) |
| `t4g_cfg_repro_seed42_config.py` | App. *Sensitivity to inference-time guidance* (CFG sweep training config) |

## Contents

| Files | What they do |
|---|---|
| `t4g_gdino.py` | GroundingDINO zero-shot locator + instruction parser (mover / source / target names). |
| `t4g_detect.py`, `t4g_detect_dispatch.py`, `t4g_detect_kjob.sh` | Per-frame detection over the 92 GR1 videos → `t4g_anno/<vid>.json` (target cell per latent frame, destination cells, arrival time, distractors, `_packidx2vid.json`). |
| `t4g_gripper_detect.py` (+dispatch/kjob) | Gripper boxes → `gripper_anno/<vid>.json` (empty-gripper supervision cells). |
| `t4g_probe.py`, `launch_t4g_probe_kjob.sh`, `t4g_probe_kjob.sh`, `t4g_probe_cpu_smoke.py` | Layer-wise correspondence probe: hooked DiT block features vs. optical-flow trajectories (endpoint error) and static-group self-similarity; selects the TIA block. |
| `t4g_measure.py` (+dispatch/kjob) | GDINO-anchored EPE per layer/noise level and similarity-margin calibration. |
| `t4g_ghost_probe.py`, `t4g_restore_probe.py`, `t4g_empty_map.py`, `t4g_change_test.py`, `t4g_dup_test.py` (+dispatch/kjob, `t4g_ghost_smoke.py`) | Falsification probes: paste-retention ("ghost") measurements, erasure-restoration mirror test, safe-region mask budgets, and the change/similarity detection tests that killed weaker loss designs. |
| `t4g_aug_prep.py` (+dispatch/kjob) | Offline IGR assets → `aug_assets/<vid>.npz`: frame-0 target patch plus per-cell paste-safety zones. |
| `t4g_aug_paste.py` | Pure corruption logic: paste-plan sampling (never frame 0, only safe zones), scaling/blur, latent-region pasting. No training-stack deps. |
| `t4g_aug_trainer.py`, `t4g_aug_config.py`, `t4g_aug_launch.sh`, `t4g_aug_smoke.py` | IGR trainer: dual VAE encode (corrupted input / clean target), up-weighted pasted-region loss, sentinel metrics; CPU smoke for the corruption math. |
| `t4g_weightmap.py`, `t4g_weightmap_precompute.py`, `t4g_weightmap_armfix.py`, `t4g_weightmap_viz.py` | Contractual spatial weight maps (object trajectory / grasp / place / should-be-empty / background) from the annotation cache, cache precompute, arm-coverage fix variant, PNG calibration. |
| `t4g_weightmap_variants.py`, `t4g_weightmap_variants_precompute.py`, `t4g_weightmap_<variant>_config.py`, `t4g_weightmap_<variant>_launch.sh`, `t4g_weightmap_variant_matrix.sh` | The six IGR spatial-weight designs of the appendix design table as one registry plus per-variant configs/launchers/matrix entry — side-by-side options, no recommended default; see [`t4g_weightmap_variants.md`](t4g_weightmap_variants.md). |
| `t4g_corr_loss.py`, `t4g_corr_trainer.py`, `t4g_corr_config.py`, `t4g_corr_launch.sh`, `t4g_corr_smoke.py` | TIA losses (`L_id` adjacent-frame relay at the probed block; `L_change` self-similarity-change term) with forward hooks, λ warmup, gate logic, and four-checkpoint CPU smoke. |
| `t4g_joint_trainer.py`, `t4g_joint_config.py`, `t4g_joint_v2_config.py`, `t4g_joint_cleanv2_config.py`, `t4g_joint_paste3_config.py`, `t4g_joint_launch.sh`, `t4g_joint_wmap_smoke.py`, `t4g_wmaponly_config.py` | Joint IGR + TIA trainer (inherits the corr hooks, swaps in the restoration forward), recipe variants (v2 paste-shortcut blocking, clean-v2 data, weight-map-only ablation), a paste-weight twin (`t4g_joint_paste3_config.py`, `t4g_w_paste=3.0` next to the historical `4.0`), and weight-map wiring smoke. |
| `t4g_apre_noaug_{config,launch,preflight}`, `t4g_apre_noaug_b23_config.py`, `t4g_cleanv2_{apre_launch,preflight}`, `t4g_cfg_repro_seed42_config.py`, `t4g_cfg_repro_seed42_b23_config.py` | Legacy A-pre recipe reproductions (augmentation off / clean-v2 / seed-42 CFG-sweep base) with preflight validators, plus the paper-side `block23` twins of the first and last of them; the launch scripts pick the recipe with `BASE_CONFIG_MODULE=...`, and `T4G_EXPECTED_SAMPLES` gates whichever annotation set the run reads. |
| `t4g_final_trainer.py`, `t4g_final_config.py`, `t4g_final_launch.sh` | Final combined arm: pretrained base, clean-SFT anchor + `L_id` + online self-generated restoration cases (A2) judged by the frozen ICH-D scorer; `--selftest` CPU unit test. |
| `t4g_final_align.py`, `t4g_final_kprobe.py` (+kjob), `t4g_final_eval175_kjob.sh` | Final-arm audits: online-vs-offline restoration-target alignment, roll-length K sweep probe, and DreamGenBench generation for the checkpoint sweep. |
| `t4g_exam.py`, `t4g_exam_v2.py`, `t4g_exam_v3.py` (+dispatchers, `t4g_exam_kjob.sh`, `kjob_eval_t4g_exam*.sh`) | MLR counting instrument: initial-frame inventory `inv0`, DUP/VANISH persistence rules; v2 adds gripper-overlap rejection and higher thresholds, v3 builds `inv0` from the conditioning frame. |
| `ewm_dup.py`, `ewm_dup_dispatch.py` | EWMBench MLR variant (manual mover table, side-by-side crop, conditioning-frame inventory). |
| `t4g_gr92_gemini_manifest.py`, `run_eve_ablation_gemini_repeats.sh` | Gemini-IF manifests for the 92-prompt re-exam pools and judge-repeat runs (see [`eveworld/evaluation/`](../evaluation/)). |
| `t4g_ablation_{sft,cp_only,cwm_only,cic_only,igt}_config.py`, `t4g_strict_ablation_common.py`, `t4g_strict_ablation_preflight.py` | Publication ablation matrix (matched Standard SFT, CP-only, CWM-only, CIC-only, IGT) sharing fixed invariants, plus a preflight validator. |
| `kjob_eve_ablation_train_chain.sh`, `kjob_eve_ablation_generate_chain.sh`, `kjob_eve_cic_ema_checkpoint_sweep.sh`, `watch_eve_*.sh` | Chained ablation training/generation/eval and watcher scripts that submit follow-on stages when prerequisites finish. |
| `selfcase_mine_dispatch.py`, `selfcase_build.py`, `selfcase_gen_*.sh/py`, `selfcase_pool_fix_names.py` | Self-case pipeline: mine DUP rollouts from generation pools, build repaired case banks (`npz` base+patch), pool generation launchers, filename repair. |
| `selfcase_g1.py`, `selfcase_g1p.py`, `selfcase_cic_match.py` (+kjob) | Separability studies for duplication signatures (transient self-similarity, layer fusion, cross-frame "matching" features) with LOO logistic-regression probes. |
| `t4g_selfcase_trainer.py`, `t4g_selfcase_config.py`, `t4g_selfcase_launch.sh`, `t4g_ich_case_merge.py` | Self-case restoration training (model's own duplicates as cases; controlled counterpart of IGR) and multi-pool case-bank merge. |
| `t4g_ich_trainer.py`, `t4g_ich_{1L,3L}_*`, `t4g_ich_fusion_ablation.py`, `t4g_ich_d_fit.py`, `t4g_ich_d_trainer.py`, `t4g_ich_D_*`, `t4g_ichD_generate.py` (+kjob) | ICH-D deletion route: learned/frozen emergence detector + zero-init suppressor, fusion ablation, frozen-LR fitting, and an inference wrapper that restores `ich_d.*` weights so generation actually exercises the module. |
| `failure_routes_reval.py`, `kjob_failure_routes_mlr.sh` | Uniform negative-route manifests and MLR scoring for the failed-alternative comparison. |
| `t4g_viz.py`, `t4g_weightmap_viz.py`, `t4g_viz_kjob.sh` | Human-eye diagnostics: similarity heatmaps, argmax landings, weight-map overlays. |

## Usage

CPU smokes (no GPU, run from the repo root with the training venv python):

```bash
python eveworld/pipeline/t4g_corr_smoke.py        # TIA loss: gate/locality/tolerance/hooks
python eveworld/pipeline/t4g_aug_smoke.py         # IGR corruption math
python eveworld/pipeline/t4g_joint_wmap_smoke.py  # weight-map wiring in the joint trainer
python eveworld/pipeline/t4g_probe_cpu_smoke.py   # probe hooks on a tiny random model
```

Typical stage order (each step consumes the previous step's caches):

```bash
# 1. Annotate the 92 GR1 training videos (GPU, sharded)
python eveworld/pipeline/t4g_detect.py --shard-index 0 --num-shards 8 --out-dir <anno_dir>
# 2. Offline IGR assets and weight-map caches
python eveworld/pipeline/t4g_aug_prep.py --shard-index 0 --num-shards 8
python eveworld/pipeline/t4g_weightmap_precompute.py
# 3. Layer-selection probe for TIA (GPU)
bash eveworld/pipeline/launch_t4g_probe_kjob.sh            # DRY_RUN=1 prints the plan
# 4. Train (launchers print a plan; pass `submit` to actually submit)
bash eveworld/pipeline/t4g_corr_launch.sh submit           # TIA-only
bash eveworld/pipeline/t4g_aug_launch.sh submit            # IGR-only
bash eveworld/pipeline/t4g_joint_launch.sh submit          # IGR + TIA
SMOKE=1 bash eveworld/pipeline/t4g_final_launch.sh submit  # final arm, 4-step smoke
# 5. MLR counting over generated videos (usually via the dispatchers;
#    the list file carries one tab-separated arm/video/prompt row per line)
python eveworld/pipeline/t4g_exam.py --list-file LIST.tsv --out OUT.json
```

Environment variables used throughout: `EVEWORLD_ROOT` (this repo's root; kjob payloads locate
`giga-world-0/` relative to it), `GAGI_ROOT`/`GAGI` (dataset and output root, defaulting to an
absolute cluster path you must override), `REPO_DIR`, `CONDA_SH`/`CONDA_ENV`,
`TRAIN_VENV`/`TRAIN_PYTHON`, plus trainer knobs such as `T4G_W_PASTE`, `T4G_REGION_LEVELS`,
`T4G_A2_K`, `T4G_EXPECTED_SAMPLES`, and `T4G_W_LAT`/`T4G_WPIX` for non-GR1 grid sizes.

### Paper-side twins (side by side, neither is a default)

Where a historical config and the paper quote different numbers, both recipes ship as separate
modules and nothing existing was changed — pick one per run:

| Twin module | Differs from | Change | How to run |
|---|---|---|---|
| `t4g_apre_noaug_b23_config.py` | `t4g_apre_noaug_config.py` (`block22`) | TIA block `block23` | `BASE_CONFIG_MODULE=eveworld.pipeline.t4g_apre_noaug_b23_config bash eveworld/pipeline/t4g_apre_noaug_launch.sh submit` |
| `t4g_cfg_repro_seed42_b23_config.py` | `t4g_cfg_repro_seed42_config.py` (`block22`) | TIA block `block23` | `BASE_CONFIG_MODULE=eveworld.pipeline.t4g_cfg_repro_seed42_b23_config bash benchmarks/dreamgenbench/kjob_cfg_train_then_sweep.sh` |
| `t4g_joint_paste3_config.py` | `t4g_joint_config.py` (`t4g_w_paste=4.0`) | `t4g_w_paste=3.0` | `BASE_CONFIG_MODULE=eveworld.pipeline.t4g_joint_paste3_config bash eveworld/pipeline/t4g_joint_launch.sh submit` |
| `eveworld/tia_transport/cic_transport_b23_config.py` | `cic_transport_config.py` (`block22`) | `block23` on both the TIA and transport keys | `BASE_CONFIG_MODULE=eveworld.tia_transport.cic_transport_b23_config bash eveworld/pipeline/t4g_joint_launch.sh submit` (the paired campaign stays bit-exact and only registers `control`/`transport`) |

`t4g_joint_cleanv2_config.py` keeps `t4g_expected_samples=91` because its `t4g_anno_nohuman_v2`
cache really holds 91 clips (video 32 is excluded); the env var `T4G_EXPECTED_SAMPLES` overrides
both that value and the launcher's preflight gate, so the same code runs on a 92-clip cache
without editing the config.

> **Cluster jobs.** All `kjob_*.sh`, `launch_*.sh`, and `watch_*.sh` scripts are SLURM-style
> wrappers (`#SBATCH` headers, single-node 8-GPU payloads). Their absolute paths (data roots,
> conda/venv locations), job names, and account/namespace settings are cluster-specific and must
> be adapted before use. GPU payloads refuse to run on a GPU-less workspace host unless
> `ALLOW_LOCAL_RUN=1` is set; training launchers only print the plan unless given `submit`.
> Trailing `KEY=VALUE` arguments are exported inside the job and override config defaults.

## Notes

- Trainers subclass `GigaWorld0Trainer` from the vendored [`giga-world-0/`](../../giga-world-0/)
  snapshot and register transforms through [`giga-models/`](../../giga-models/); set `PYTHONPATH`
  to include the repo root, `giga-world-0/`, and `giga-models/` (the kjob payloads do this).
  Environment setup: [`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md).
- Data prerequisites: GR1 raw videos + packed training data, the `t4g_anno/` detection cache
  (step 1), `aug_assets/` (step 2), `weightmap_cache/` (step 2), a GroundingDINO checkpoint
  (`GDINO_PATH` in `t4g_gdino.py`), and — for self-case/ICH arms — a mined case bank.
- The detection caches produced here feed every trainer above; generation and instruction-following
  evaluation (Qwen-IF / Gemini-IF) live in [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/)
  and [`eveworld/evaluation/`](../evaluation/), with transfer pipelines in
  [`benchmarks/ewmbench/`](../../benchmarks/ewmbench/) and
  [`benchmarks/worldarena/`](../../benchmarks/worldarena/) (WorldArena 1.0).
- The zero-initialized cross-frame feature-transport implementation referenced by the TIA trainers
  lives in [`eveworld/tia_transport/`](../tia_transport/); the other four failed alternative
  designs (PhysicsLatent, EAG, LAD-LoRA, Causal Frontier) live in [`eveworld/method/`](../method/).
