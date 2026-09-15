# AgiBot → EWMBench Transfer Pipeline

Training-set construction, IGR/TIA adaptation, training recipes, and evaluation helpers for the AgiBot training-distribution transfer experiment, plus the auxiliary WorldModelBench-style (WMB) adaptation line built from Open X-Embodiment sources.

## Paper mapping

- **Section 5.3 "Generalization Across Domains and Training Data"** (`sec:generalization`), **Table `tab:agibot_transfer`** ("Generalization across training distributions"): the 777-clip AgiBot training set is built by `w9_extract_agibot.py` → `w10_finalize_agibot_trainset.py` → `a1_clean_dir.py` → `a2_make_idx2vid.py`, and the EVEWorld arm is trained by the `agi_*` recipe (`agi_aug_prep.py`, `agi_weightmap.py`, `agi_aug_trainer.py`, `agi_joint_config.py`).
- **Appendix "Implementation Details"**: training budget (777 AgiBot clips, 480×640, 93 frames at 16 FPS, raw step-50 checkpoint for generation) corresponds to the configs here, which checkpoint every 50 steps.
- **Appendix "Complete EWMBench Results"** (`app:ewmbench_full`, `tab:agibot_transfer_full`): the EWMBench evaluation consumes probes assembled by `agi_make_probe.py` through [`benchmarks/ewmbench/`](../../benchmarks/ewmbench/), with the semantics side pre-computed by `ewm_precaption.py` / `ewm_semantics_cpu.py`.
- The IGR copy-paste assets, weightmap, and TIA losses are AgiBot adaptations of the shared T4G method stack in [`eveworld/pipeline/`](../pipeline/) (paper Sections 4.1–4.2); the scripts here monkey-patch its grid/path constants to 640×480 (latent 24×30×40) instead of duplicating it.

## Contents

### AgiBot training-set construction

| File | What it does |
|---|---|
| `w9_extract_agibot.py` | Extracts sub-action clips from AgiBotWorld tars: decodes `head_color.mp4` (AV1 via PyAV), segments by `task_info` `action_config`, resamples to 93-frame 640×480@16 FPS mp4 + instruction txt. Excludes the 21 EWMBench test episodes (leakage guard, audit JSON). |
| `w10_finalize_agibot_trainset.py` | Per-task balanced downsampling (`--cap 150`, fixed seed, episode-diversity-first round robin) into `agibot_ewm_train_final/` via hard links + manifest. |
| `a1_clean_dir.py` | Builds `agibot_ewm_clean/` (777 clips) excluding the `*_trans.mp4` preview duplicates that would corrupt sorted-glob packing; asserts counts. |
| `a2_make_idx2vid.py` | Writes `_packidx2vid.json` (packing index → clip stem) and md5-verifies that the packed dataset matches the clean directory in sorted order. |

### Annotation, weightmaps, and IGR assets

| File | What it does |
|---|---|
| `agi_parse.py` | Authoritative per-clip instruction parse from `task_info` skill labels + verb templates → `{skill, category(TRANSFER\|STATE), arm, object, source, dest, state_part}`. `--selftest` reports parse coverage over all 777 clips. |
| `agi_detect.py` | GroundingDINO entity detection over the 777 clips (patches `t4g_detect` to 640×480 / 40×30 latent, injects `agi_parse` entities per clip, adds `state_cells` for STATE clips) → `t4g_anno/<name>.json`. |
| `agi_gripper_detect.py` | Dual-arm gripper detection with `left`/`right` side labels (by gripper center cell) → `gripper_anno/<name>.json`. |
| `agi_detect_dispatch.py` | 8-GPU shard dispatcher running the `agi_detect` then `agi_gripper_detect` rounds (`N_SHARDS` to override). |
| `agi_weightmap.py` | Dual-arm contractual weightmaps `(24,30,40)`: optical-flow motion masks ∪ side-aware gripper corridors ∪ detected object/dest/state cells; two cached variants (`motionauto` {1.0,3.0}, `skilltiered` {1.0,2.0,2.5,3.0}) with a ≤0.33 coverage cap and adaptive fallback; optional overlay viz. |
| `agi_zones_originfix.py` | Recomputes augmentation zones so TRANSFER clips get an origin-A zone anchored at the target's initial cell after it departs (CPU, in-place npz rewrite). |
| `agi_aug_prep.py` | IGR copy-paste assets (`patch`/`box`/`zones` npz per clip) wrapping `t4g_aug_prep`; extends `build_zones` so ±2-frame dilated dual-arm gripper corridors are carved out as unsafe (copies never paste onto arms). |
| `agi_aug_prep_dispatch.py` | 8-GPU shard dispatcher for `agi_aug_prep.py`. |

### Training

| File | What it does |
|---|---|
| `agi_aug_trainer.py` | `AgiAugTransform`: skill-conditioned zone bias for TRANSFER skills; STATE skills disable object pasting and keep only low-rate background pasting (`p_aug_state=0.15`). `AgiJointTrainer` is the runner entry that triggers registration. |
| `agi_joint_config.py` | Full recipe config: GigaWorld-0 pretrain base + `AgiAugTransform` + `motionauto` weightmap + TIA (`t4g_id_block`, `t4g_lambdas`); CAME-8bit, bf16, DeepSpeed ZeRO-2, seed 42. |
| `agi_wmaponly_config.py` | Minimal fail-fast arm: same base with copy-paste and TIA disabled, 150 steps — gate before enabling the full recipe. |
| `agi_joint_launch.sh` | Launcher: `check` (runs `agi_preflight.py` + py-compile) or `submit` (delegates to [`benchmarks/dreamgenbench/launch_gr1_train_kjob.sh`](../../benchmarks/dreamgenbench/launch_gr1_train_kjob.sh)); `ARM=wmaponly\|full`, forwards `T4G_W_LAT=40`/`T4G_WPIX=640` and optional `T4G_ID_BLOCK`/`T4G_WMAP_DIR`/`T4G_LAMBDAS`. |
| `agi_preflight.py` | Pre-submission validation: 777-clip clean dir free of `_trans`, packed size, idx2vid continuity, anno/assets/weightmap/gripper id sets, array shapes and weightmap value tiers, grid env consistency. |
| `agi_make_probe.py` | Assembles a probe model dir from a training checkpoint (raw or `--ema` transformer + pretrain VAE/text-encoder symlinks), named `ewm_apre_*` for the EWMBench generation harness. |

### WorldModelBench-style (WMB) adaptation line

| File | What it does |
|---|---|
| `w1_scan_oxe_index.py` | Indexes local Open X-Embodiment sources (bridge, taco_play, berkeley_autolab_ur5, jaco_play): tar shard, sample, instruction, frame count, first-frame dHash → per-dataset jsonl. |
| `w2_build_trainset.py` | Leakage-guarded sampling (drops episodes within dHash Hamming ≤12 of the 50 WMB robotics first frames, audit JSON; quotas 300/100/60/40) → 500 clips of 93 frames at 640×480@16 FPS + native instructions. |
| `w3_rewrite_prompts.py` | Rewrites native instructions into WMB style ("The robotic arm ...") via an OpenAI-compatible endpoint (`QWEN_BASE`); originals kept as `.txt.orig`. |
| `w4_detect.py` / `w4_dispatch.py` | GDINO detection over the WMB trainset (patched 640×480 grid; prefers `.txt.orig` prompts) → `t4g_anno/`; dispatcher shards over 8 GPUs. |
| `w5_make_wmap.py` | Weightmap cache {1.0,3.0} = DIS-flow motion ∪ detected target/dest/inventory cells, with overlay viz and coverage stats. |
| `w6_aug_prep.py` / `w6_dispatch.py` | Copy-paste augmentation assets wrapping `t4g_aug_prep` on the WMB trainset; dispatcher runs CPU shards. |
| `w7_make_probe.py` | Assembles `wmb_<vanilla\|apre>_s<step>` probe dirs (checkpoint transformer + pretrain VAE/text-encoder symlinks). |

### EWMBench evaluation helpers

| File | What it does |
|---|---|
| `ewm_precaption.py` | Zero-GPU caption pre-run: queries the endpoint (`QWEN_BASE`) for every video in the converted eval layout and caches `<model>_caption_responses.json` with official-style keys, so the GPU evaluation chain hits cache. |
| `ewm_semantics_cpu.py` | CPU scoring of the EWMBench semantics metrics (BLEU-4, CLIPScore, logic pass rate) from the cached caption responses + GT captions. |

## Usage

Typical AgiBot → EWMBench order (run from the `giga-world-0/` backbone checkout, as the launcher does):

```bash
# 1. Training set: extract -> finalize -> clean -> packing index
python eveworld/agibot/w9_extract_agibot.py
python eveworld/agibot/w10_finalize_agibot_trainset.py --cap 150
python eveworld/agibot/a1_clean_dir.py
python eveworld/agibot/a2_make_idx2vid.py          # then pack with the standard pack_data tooling

# 2. Annotations and IGR assets (GDINO stages need the detection environment)
python eveworld/agibot/agi_parse.py --selftest
python eveworld/agibot/agi_detect_dispatch.py      # entity + gripper rounds
python eveworld/agibot/agi_weightmap.py --variant both --viz-n 30
python eveworld/agibot/agi_aug_prep_dispatch.py
python eveworld/agibot/agi_zones_originfix.py

# 3. Train (preflight runs automatically); wmaponly gate first, then full
bash eveworld/agibot/agi_joint_launch.sh submit ARM=wmaponly
bash eveworld/agibot/agi_joint_launch.sh submit ARM=full T4G_ID_BLOCK=block<N>

# 4. Probe -> generation/evaluation
python eveworld/agibot/agi_make_probe.py --run agi_apre_full_s300 --step 50
# then benchmarks/ewmbench/kjob_ewmbench_8gpu_serial.sh with MODELS=ewm_apre_full_s50
python eveworld/agibot/ewm_precaption.py ewm_apre_full_s50
python eveworld/agibot/ewm_semantics_cpu.py pretrain ewm_apre_full_s50
```

The WMB line runs `w1` → `w2` → `w3` → `w4_dispatch.py` → `w5` → `w6_dispatch.py`, trains via `benchmarks/ewmbench/kjob_wmb_apre_train.sh`, and assembles probes with `w7_make_probe.py`.

## Notes

- **Paths are constants, not flags.** Most scripts here hardcode module-level roots under the internal `GAGI_ROOT` layout (`/data/datasets/gagi`, see [`eveworld/common/env.sh`](../common/env.sh)); adapt them (or keep the same layout) before running. Repo-level wrappers use `EVEWORLD_ROOT` for the repository root.
- **Cluster scripts are templates.** The `kjob_*` wrappers in [`benchmarks/ewmbench/`](../../benchmarks/ewmbench/) and `launch_gr1_train_kjob.sh` in [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/) are SLURM-style single-node 8-GPU jobs whose absolute paths (conda envs, data roots, runtimes) must be adapted to your site; the `*_dispatch.py` scripts here are their per-GPU shard fan-outs (`N_SHARDS`).
- **Resolution contract.** The AgiBot recipe is 640×480 with a 24×30×40 latent grid; `T4G_W_LAT=40` and `T4G_WPIX=640` must be exported wherever the T4G transform stack is imported (the launcher and prep scripts already do this — omitting them builds a (24,30,48) grid and trips shape asserts).
- **Dependencies.** Detection stages need GroundingDINO weights (`GDINO_PATH` in [`eveworld/pipeline/t4g_gdino.py`](../pipeline/t4g_gdino.py)) in the detection environment; `w9` needs PyAV for AV1; `w3`/`ewm_precaption.py` need an OpenAI-compatible endpoint (`QWEN_BASE`); see [`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md).
- **Data prerequisites.** AgiBotWorld raw tars + `task_info` for `w9`/`agi_parse.py`; a local Open X-Embodiment copy and the WMB robotics prompt images for the `w1`–`w7` line. Both lines enforce leakage guards (21 EWMBench test episodes excluded; dHash dedup against WMB first frames) and write audit JSONs.
