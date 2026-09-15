# WorldArena 1.0 Zero-Shot Evaluation

Reproducible pipeline for the zero-shot WorldArena 1.0 evaluation of EVEWorld and all compared models: video normalization, the eight public WorldArena component metrics (aggregated into EWMScore-local-8), and the cross-domain Model Laziness Rate (MLR) protocol.

## Paper mapping

- **Section 5.3 "Generalization Across Domains and Training Data" → "Zero-Shot Domain Generalization"**, **Table 2** (`tab:worldarena`, "Zero-shot results on WorldArena 1.0"): the eight component metrics (Image Quality, Aesthetic Quality, Dynamic Degree, Flow Score, Motion Smoothness, Subject Consistency, Background Consistency, Photometric Consistency) and the equal-weight overall EWMScore-local-8 are produced by `local_metric_eval.py` + `aggregate_core_scores.py`.
- **Appendix "MLR Evaluation Protocol"** (`app:mlr_protocol`, Algorithm 1): the deterministic GroundingDINO counting rule (initial frame + 24 uniformly sampled timestamps, gripper-overlap removal, two-consecutive-timestamp deviation rule) is implemented by `mlr_dispatch.py` + `mlr_eval.py`.
- **Appendix "Cross-Domain Evaluation"** (`app:mlr_cross_domain`), **Tables `tab:mlr_wa` / `tab:mlr_wa_permodel`**: the instruction-only mover parser (817/1,000 prompts), the common-eligible-set comparison, Wilson confidence intervals, and the paired exact McNemar test are produced by `mlr_dispatch.py` and `recompute_video_first_mlr.py`.
- **Appendix "Training Dynamics, Guidance, and Output Length"** and the RoboTwin/FlowWAM transfer arm additionally reuse the frozen MLR protocol through `mlr_cfg_flowwam_dispatch.py` and `mlr_flowwam_variants.py` (see [`../robotwin_flowwam/`](../robotwin_flowwam/)). The main-results DreamGenBench MLR pipeline lives in [`../dreamgenbench/`](../dreamgenbench/).

## Contents

| File | What it does |
|---|---|
| `prepare_evaluator.sh` | Creates a pinned git worktree of the public WorldArena 1.0 evaluator at a fixed commit, applies `worldarena_eval.patch`, verifies SHA-256 of the SEA-RAFT / VFIMamba / aesthetic checkpoints, and smoke-tests the import. |
| `worldarena_eval.patch` | Compatibility patch for the evaluator worktree: safetensors support in the SEA-RAFT loader, optional `motion_smoothness` import, and a `WORLDARENA_CACHE_DIR` override. |
| `eval_config.yaml` | Checkpoint paths consumed by `local_metric_eval.py` for each of the eight metrics (MUSIQ, CLIP/aesthetic head, RAFT, SEA-RAFT, DINO, VFIMamba). |
| `prepare_manifest.py` | Builds the isolated 1,000-prompt manifest from the benchmark summary JSON and audits every `<model>_test/` video directory (missing / extra / empty files); refuses to emit a manifest unless coverage is exact. |
| `wa_postprocess.py` | Converts side-by-side generation outputs into the official evaluation layout: crops the generated 640-wide half, trims to 121 frames, writes `<model>_test/fixed_scene_task_episodeK.mp4` at 24 FPS. |
| `normalize_xmodel_videos.py` | Re-encodes external (non-GigaWorld) model videos to the same contract (640×480, 121 frames, 24 FPS, libx264) with probe-verified output and a per-video report. |
| `local_metric_eval.py` | Runs one public WorldArena metric over the manifest under `torchrun`, with per-video coverage checks and atomic JSON output (`<model>/core/<metric>.json`). |
| `aggregate_core_scores.py` | Aggregates the eight per-metric JSONs into per-video CSVs and the `model_comparison.csv` / `.json` leaderboard ranked by EWMScore-local-8. |
| `mlr_dispatch.py` | MLR controller: parses mover objects from instructions with a fixed transport-verb + object-regex parser, builds sharded jobs, launches `mlr_eval.py` workers, and merges results with Wilson CIs and the pretrain/SFT/EVEWorld comparability gate. |
| `mlr_eval.py` | Per-shard MLR worker: GroundingDINO target counting on the condition image and 24 uniformly sampled frames per video, gripper-overlap filtering, two-consecutive-frame event rule. |
| `recompute_video_first_mlr.py` | Re-anchors the instance inventory on each generated video's own first frame, restricts to the common eligible set across models, and reports MLR, Wilson CIs, and exact McNemar paired tests (the `tab:mlr_wa` numbers). |
| `mlr_cfg_flowwam_dispatch.py` | Runs the frozen MLR protocol over the CFG/denoising-step sweep and the FlowWAM control-vs-EVEWorld held-out outputs. |
| `mlr_flowwam_variants.py` | Generic MLR runner for arbitrary FlowWAM variants given as `NAME=VIDEO_DIR` on the 250-row held-out manifest. |
| `kjob_worldarena1_eval_serial.sh` | SLURM-style cluster wrapper that loops models × metrics and calls `local_metric_eval.py` via `torchrun` (skips completed outputs). |
| `kjob_worldarena1_mlr.sh` | SLURM-style cluster wrapper that invokes `mlr_dispatch.py` with the training-environment Python. |

## Usage

All paths are driven by environment variables. `EVEWORLD_ROOT` must point to this repository root; data-side roots (`WA1_ROOT`, `VIDEO_ROOT`, `EVAL_ROOT`, checkpoint directories) follow your local storage layout — see `eveworld/common/env.sh` for the `GAGI_ROOT` convention used across the repo.

1. **Set up the evaluator** (once):

   ```bash
   EVEWORLD_ROOT=/path/to/EVEWorld \
   WORLDARENA_ROOT=/path/to/worldarena_eval_worktree \
     bash benchmarks/worldarena/prepare_evaluator.sh
   ```

   Edit `eval_config.yaml` so every checkpoint path matches your local copies.

2. **Prepare videos.** Generate with the inference settings of Appendix "Implementation Details" (step-250 checkpoint, seed 004, 30 denoising steps, CFG 7.0, 480×768, 93 frames at 16 FPS for GigaWorld-based models), then post-process into the official layout with `wa_postprocess.py`. For external I2V models, re-encode with `normalize_xmodel_videos.py` instead.

3. **Build the manifest** (audits all models' video coverage first):

   ```bash
   python benchmarks/worldarena/prepare_manifest.py \
     --summary /path/to/worldarena_summary.json \
     --video-root /path/to/eval_videos \
     --models pretrain round0 t4g_wmapA_pre_seed42_s250 \
     --output "$WA1_ROOT/manifests/track1_it2v.json" \
     --audit-output "$WA1_ROOT/manifests/track1_it2v_audit.json"
   ```

4. **Run the eight core metrics** (per model, per metric, 8 GPUs):

   ```bash
   torchrun --standalone --nproc-per-node=8 \
     benchmarks/worldarena/local_metric_eval.py \
     --worldarena-root "$WORLDARENA_ROOT" \
     --config benchmarks/worldarena/eval_config.yaml \
     --manifest "$WA1_ROOT/manifests/track1_it2v.json" \
     --expected-count 1000 \
     --video-dir "$VIDEO_ROOT/pretrain_test" \
     --metric flow_score \
     --output "$EVAL_ROOT/pretrain/core/flow_score.json"
   ```

   then aggregate the leaderboard:

   ```bash
   python benchmarks/worldarena/aggregate_core_scores.py \
     --eval-root "$EVAL_ROOT" --models pretrain round0 t4g_wmapA_pre_seed42_s250 \
     --manifest "$WA1_ROOT/manifests/track1_it2v.json" \
     --output-dir "$EVAL_ROOT/aggregate"
   ```

5. **Run MLR** (sharded over 8 GPUs) and re-anchor on generated first frames:

   ```bash
   python benchmarks/worldarena/mlr_dispatch.py \
     --manifest "$WA1_ROOT/manifests/track1_it2v.json" \
     --video-root "$VIDEO_ROOT" \
     --models pretrain round0 t4g_wmapA_pre_seed42_s250 \
     --output-dir "$WA1_ROOT/mlr_v2" --python /path/to/python --num-shards 8

   python benchmarks/worldarena/recompute_video_first_mlr.py \
     --input "$WA1_ROOT/mlr_v2/summary.json" \
     --output "$WA1_ROOT/mlr_v2/summary_video_first.json"
   ```

6. **Cluster submission.** `kjob_worldarena1_eval_serial.sh` and `kjob_worldarena1_mlr.sh` wrap steps 4–5 as SLURM-style jobs (`sbatch` headers, one 8-GPU node). They accept `KEY=VALUE` environment overrides on the command line.

## Notes

- **Cluster scripts must be adapted.** The `kjob_*` wrappers and `prepare_evaluator.sh` carry absolute defaults that point at internal storage (datasets, conda installations, evaluator worktrees). Override `EVEWORLD_ROOT`, `WORLDARENA_ROOT`, `WA1_ROOT`, `VIDEO_ROOT`, `EVAL_ROOT`, `CONDA_SH`, `CONDA_ENV`, and `TRAIN_PYTHON` for your site; the scripts are templates, not turnkey installers.
- **Dependencies.** The metric path needs the `WorldArena` conda environment (see [`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md)) plus the evaluator checkpoints verified in `prepare_evaluator.sh`. The MLR path needs GroundingDINO weights; set the model path in the detection stack (`GDINO_PATH` in [`eveworld/pipeline/t4g_gdino.py`](../../eveworld/pipeline/t4g_gdino.py)).
- **Detection stack location.** `mlr_eval.py` imports `t4g_detect`, `t4g_exam_v2`, and `t4g_gdino` from [`eveworld/pipeline/`](../../eveworld/pipeline/) (the import root is resolved relative to this file).
- **Data prerequisites.** A WorldArena 1.0 summary JSON (1,000 rows with `image`, `prompt`, `gt_path`) and generated videos named `fixed_scene_task_episodeK.mp4` under `<VIDEO_ROOT>/<model>_test/`. Every stage validates exact coverage and fails loudly rather than evaluating a partial set.
- The parser and detector settings (transport-verb/object lists, box thresholds, gripper-overlap rule, 24 sampled frames, two-consecutive-timestamp rule) are frozen protocol — they were fixed on a held-out development split before evaluation, so change them only for new experiments, not for reproducing the paper numbers.
