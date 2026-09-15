# PBench Robot Evaluation

Generation and evaluation pipeline for the PBench Robot physical-QA benchmark (NVIDIA PBench) on GigaWorld-0-based models: a Qwen-VL physical-QA judge (Domain score), VBench quality metrics, output-length sweeps, and the VideoPhy-2 PA-II physical-adherence evaluation for DreamGenBench videos.

## Paper mapping

- **Main text, Ablations and Analysis — Table `tab:horizon_compute_main` ("Sequence-length scaling and inference cost")**: 174 completed videos per duration at 3.8/5.8/9.8/15.8/19.8 s on a fixed eight-GPU layout (latency percentiles, throughput, peak memory, GPU-hours). Produced by the generation side of `run_pbench_robot_length_sweep.sh` (per-tier `generation_summary.json` + `gpu_memory_peak.json`).
- **Appendix "Training Dynamics, Guidance, and Output Length" (`app:training_sampling_sensitivity`) — Table `tab:pbench_horizon` ("PBench physical-QA accuracy versus output duration")**: Domain score and the Physical/Spatial/Temporal dimensions on PBench Robot (913 QA pairs over the same 174 videos per duration; judge: Qwen3.6-VL with thinking enabled). The Domain score comes from `eval_pbench_robot_qwen_vqa.py`; the quality side (eight VBench dimensions) from `eval_pbench_robot_vbench_quality.py`; the merged table from `summarize_pbench_quality_upto19.py`.
- **VideoPhy-2 (PA-II) physical-adherence scoring** of DreamGenBench generations (`setup_videophy_env.sh`, `run_videophy_pa2.py`, `kjob_dreamgenbench_videophy_pa2.sh`): feeds the PA-II analyses aggregated in [`../../eveworld/evaluation/`](../../eveworld/evaluation/) (e.g. `eval175_compare_pa2.py`).

## Contents

| File | What it does |
|---|---|
| `prepare_pbench_it2v.py` | Converts the `nvidia/PBench` parquet into GigaWorld-0 image-to-video input: extracts condition images and writes `pbench_<subset>_it2v.json` plus a `.metadata.jsonl` carrying prompts and `qa_pairs` for the judge. |
| `launch_pbench_robot_kjob.sh` | One-command generation launcher. `MODE=smoke` (default: 1 prompt, 1 step) or `MODE=full` (all 174 prompts, 30 steps); submits a cluster job through the GigaWorld-0 submit helper; prepares inputs first when `PREPARE_PBENCH_INPUT=1`. |
| `kjob_pbench_2gpu_serial.sh`, `kjob_pbench_8gpu_serial.sh` | SLURM-style payloads that loop named checkpoints (`pretrain`, `round0`, `t4g_*`) through `giga-world-0/scripts/inference.py` on 2 or 8 GPUs; skip models whose summary already exists. |
| `resume_pbench_robot_serving.sh` | Resumes an interrupted full run against an already-running GigaWorld-0 HTTP serving endpoint (via `giga-world-0/scripts/call_gigaworld0_serve.py`); infers the offset from the results file. |
| `eval_pbench_robot_qwen_vqa.py` | PBench physical-QA scorer. Samples frames per video, asks the yes/no questions through an OpenAI-compatible Qwen-VL endpoint (JSON-constrained answers, retries, resume), and writes `qwen_vqa_results.jsonl` + `qwen_vqa_summary.json` (question micro-accuracy, sample-macro `domain_score_like`, per-category tables). |
| `eval_pbench_robot_qwen_vqa.sh` | Env-var wrapper for the above (`METADATA_JSONL`, `VIDEO_DIR`, `EVAL_DIR`, `QWEN_BASE`, `QWEN_MODEL`, `CONCURRENCY`, `MODEL_MAX_TOKENS`, `DISABLE_THINKING`, ...). Sends HTTP requests only; no local judge weights. |
| `run_pbench_gen_eval_5models.sh` | Judges the 3.8 s generations of five checkpoints in parallel against one Qwen endpoint. |
| `run_pbench_qwen58_thinking32000_eval.sh` | Re-judges completed sweep generations with the paper judge configuration (thinking enabled, 32000 max tokens) into separate eval dirs, then writes summary and baseline-vs-thinking comparison CSV/JSON. |
| `run_pbench_qwen58_think32000_upto19_clean.sh` | Backfills/cleans Domain evals for the ≤19.8 s tiers of both the Pretrain and GR1/SFT sweeps: de-duplicates result JSONLs, drops error rows, refills only missing attempts. |
| `run_pretrain_pbench_15p8s_qwen58_think32000.sh`, `run_pretrain_pbench_19p8s_qwen58_think32000.sh` | Standalone single-duration re-judge for the two longest pretrained tiers. |
| `download_vbench_quality_checkpoints.sh` | Resumable download of every metric checkpoint: DINO repo + weights, CLIP ViT-B/32 and ViT-L/14, LAION aesthetic predictor, MUSIQ (pyiqa), AMT, ViCLIP, and the DreamSim ensemble. |
| `prepare_pbench_vbench_inputs.py` | Builds VBench inputs: crops the generated half of side-by-side outputs, links condition images, writes `pbench_robot_vbench_full_info.json` + `prepare_manifest.json`. |
| `eval_pbench_robot_vbench_quality.py` | Runs the eight quality dimensions (`i2v_subject`, `i2v_background`, `aesthetic_quality`, `imaging_quality`, `background_consistency`, `motion_smoothness`, `subject_consistency`, `overall_consistency`) through VBench / VBench2-beta-I2V, maps them to the paper's short names, and fuses them with the Domain summary into `quality_score` / `overall_score_like` (`pbench_robot_quality_overall_summary.json`). |
| `eval_pbench_robot_vbench_quality.sh` | Wrapper for the above: import prechecks, input preparation, per-dimension checkpoint check, GPU guard, then the Python eval. |
| `kjob_pbench_robot_vbench_quality.sh`, `launch_pbench_robot_vbench_quality_kjob.sh` | SLURM-style GPU payload + submitter for the quality eval. The payload refuses to run on the workspace host (`ALLOW_LOCAL_RUN=1` to override). |
| `run_pbench_quality_upto19_todo.sh`, `run_pbench_quality_upto19_parallel.sh` | Quality backfill for the ≤19.8 s sweep table: sequential or one-job-per-tier parallel submission, reusing a shared DreamSim model cache. |
| `summarize_pbench_quality_upto19.py` | Merges Domain + Quality summaries into the paper-style table (eight quality dims, Domain/Quality/Overall) as CSV + JSON. |
| `compare_pbench_robot_scores.py` | Prints local Domain/Quality/Overall against the reference paper numbers baked into the script, plus per-dimension quality metrics. |
| `run_pbench_robot_length_sweep.sh` | Resumable output-length sweep for the pretrained model (3.8–29.8 s at 16 FPS): submits generation jobs, waits for completion, runs Domain eval, optionally Quality (`RUN_QUALITY=1`), and writes a per-tier summary CSV/JSON with wall time and GPU-memory peaks. |
| `run_gr1_pbench_robot_length_sweep.sh` | Same sweep for the GR1-fine-tuned (Standard SFT) transformer over the five ≤19.8 s tiers; symlinks a composite model dir (GR1 transformer + pretrain text encoder/VAE) and writes a side-by-side comparison vs. the pretrained sweep. |
| `run_gr1_pbench_remaining_15p8s_19p8s.sh` | Completes only the 15.8 s / 19.8 s GR1 tiers, archiving stale incomplete runs (`STATUS_ONLY=1` for a dry run). |
| `setup_videophy_env.sh` | Clones the VideoPhy repository, builds a dedicated venv, installs its requirements, and prints the `videocon_physics` checkpoint download command. |
| `run_videophy_pa2.py` | PA-II inference: mPLUG-Owl `videocon_physics` entailment scoring ("Yes" vs "No" logits) over a video/caption CSV, with a transformers-compatibility patch. |
| `convert_videophy_pa2_csv.py` | Thresholds raw entailment scores (default 0.5) into the PA-II CSV (`video_path,prompt,prediction,raw_score`). |
| `kjob_dreamgenbench_videophy_pa2.sh`, `launch_dreamgenbench_videophy_pa2_kjob.sh` | End-to-end cluster job for DreamGenBench videos: builds the VideoPhy input CSV via [`../dreamgenbench/prepare_dreamgenbench_videophy_input.py`](../dreamgenbench/prepare_dreamgenbench_videophy_input.py), runs PA-II, converts the output, and logs GPU-memory usage. |

## Usage

All paths are driven by environment variables. `EVEWORLD_ROOT` must point to this repository root; the data-root convention (`GAGI_ROOT`) is documented in [`../../eveworld/common/env.sh`](../../eveworld/common/env.sh). The defaults baked into the scripts point at the authors' internal storage — override them for your site.

1. **Prepare PBench Robot inputs** from the `nvidia/PBench` parquet:

   ```bash
   python benchmarks/pbench/prepare_pbench_it2v.py \
     --parquet /path/to/pbench.parquet \
     --output-root "$GAGI_ROOT/pbench/giga_input" --subset robot
   ```

2. **Generate videos** (cluster job; smoke mode by default):

   ```bash
   bash benchmarks/pbench/launch_pbench_robot_kjob.sh            # smoke
   MODE=full bash benchmarks/pbench/launch_pbench_robot_kjob.sh  # all 174 prompts
   ```

   To continue an interrupted serving-based run: `URL=http://host:8000 SAVE_DIR=/path/to/run bash benchmarks/pbench/resume_pbench_robot_serving.sh`.

3. **Domain (physical QA) eval** against a Qwen-VL endpoint — e.g. `LIMIT=1` for a smoke run:

   ```bash
   QWEN_BASE=http://127.0.0.1:8000/v1 VIDEO_DIR=/path/to/robot_mp4s \
     bash benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh
   ```

4. **Quality (VBench) eval** — download checkpoints once, then submit the GPU job:

   ```bash
   bash benchmarks/pbench/download_vbench_quality_checkpoints.sh
   bash benchmarks/pbench/launch_pbench_robot_vbench_quality_kjob.sh
   ```

5. **Length sweep** (generation + Domain + optional Quality + summary table):

   ```bash
   MODE=full DATA_LIMIT=0 RUN_QUALITY=1 bash benchmarks/pbench/run_pbench_robot_length_sweep.sh
   bash benchmarks/pbench/run_gr1_pbench_robot_length_sweep.sh   # GR1/SFT arm + comparison
   ```

6. **VideoPhy-2 PA-II** (separate venv, one GPU job):

   ```bash
   bash benchmarks/pbench/setup_videophy_env.sh
   VIDEO_DIR=/path/to/generated_only_videos \
     bash benchmarks/pbench/launch_dreamgenbench_videophy_pa2_kjob.sh
   ```

## Notes

- **Cluster wrappers must be adapted.** The `kjob_*` / `launch_*` scripts are SLURM-style cluster templates (`#SBATCH` headers, `KEY=VALUE` overrides) with absolute defaults for internal storage, conda, and the job-submission helper. The submit helper lives at `giga-world-0/scripts/submit_gigaworld0_kjob.sh`; several launchers resolve it (and `REPO_DIR`) relative to their own location, which predates the move of this directory under `benchmarks/` — point them at the vendored [`giga-world-0/`](../../giga-world-0/) checkout for your site. The `kjob_*` payloads `cd` into `${EVEWORLD_ROOT}/giga-world-0` and invoke `./benchmarks/pbench/...`, so the EVEWorld tree must be reachable from there (e.g. symlinked or mirrored). The Python entry points (`prepare_*.py`, `eval_*.py`) have no such assumptions and run directly.
- **Dependencies.** Generation and judging use the main conda env (`CONDA_ENV`, default `giga_models`; see [`../../docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md)) plus `openai`, `opencv`, and `imageio-ffmpeg`. The VBench path additionally imports `vbench`, `vbench2_beta_i2v`, `clip`, `pyiqa`, `decord`, `dreamsim`, `open_clip`, `skvideo`, `torchmetrics`, and expects the `vbench_compat` shim (vendored at `giga-world-0/scripts/vbench_compat`) on `PYTHONPATH` — the eval wrapper prepends it relative to its resolved repo root. The VideoPhy-2 environment is intentionally a separate venv built by `setup_videophy_env.sh`.
- **Data prerequisites.** The `nvidia/PBench` parquet (174 `robot_*` rows used here); generated videos named `robot_*.mp4` / `<pbench_id>.mp4`; and a reachable Qwen-VL endpoint for the Domain judge (`QWEN_BASE`/`QWEN_MODEL`, `OPENAI_API_KEY` if the endpoint requires one). The paper judge configuration is thinking enabled with a 32000-token budget — see `run_pbench_qwen58_thinking32000_eval.sh`.
- **Model laziness vs. physical QA.** This pipeline measures broad physical/temporal quality degradation with output length; the instance-conservation metric (MLR) lives in [`../worldarena/`](../worldarena/) and [`../dreamgenbench/`](../dreamgenbench/).
