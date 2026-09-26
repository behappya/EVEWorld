# Benchmark Pipelines

Reproduction pipelines for every benchmark reported in the EVEWorld paper: DreamGenBench (main results), WorldArena 1.0 (zero-shot domain transfer), EWMBench (AgiBot training-distribution transfer), RoboTwin via the FlowWAM backbone (cross-backbone transfer), PBench (physical-QA length scaling), and the external I2V baselines.

## Paper mapping

| Directory | Paper artifact |
|---|---|
| [`dreamgenbench/`](dreamgenbench/) | Sections 5.1–5.2 and Table 1 (`tab:dreamgen_overall`, per-split `tab:dreamgen_main`): GR1 data packing, Standard SFT training, generation, Qwen-IF / Gemini-IF / GPT-IF judging, EVAL-175 campaigns. Also Section 5.4: CFG sensitivity (`fig:cfg`, `tab:cfg_main`) and sequence-length scaling (`tab:horizon_compute_main`). See its [README](dreamgenbench/README.md). |
| [`worldarena/`](worldarena/) | Section 5.3 zero-shot domain generalization (Table `tab:worldarena`, EWMScore-local-8) and Appendix cross-domain MLR (`tab:mlr_wa`, `tab:mlr_wa_permodel`). See its [README](worldarena/README.md). |
| [`ewmbench/`](ewmbench/) | Section 5.3 training-distribution transfer (`tab:agibot_transfer`): AgiBot detection/augmentation/packing, EWMBench generation (3 seeds x 21 episodes), official-layout conversion, and judging. |
| [`robotwin_flowwam/`](robotwin_flowwam/) | Section 5.3 backbone transfer (`tab:flowwam_transfer`): IGR/TIA retraining on FlowWAM, TIA layer probe, held-out RoboTwin generation, PSNR/SSIM/LPIPS/Flow-EPE fidelity metrics plus the frozen MLR protocol. |
| [`pbench/`](pbench/) | Appendix length-robustness analysis (`tab:pbench_horizon`): PBench-Robot physical QA (Qwen VQA), VBench quality metrics, VideoPhy PA-II, and per-length sweeps. |
| [`baselines/`](baselines/) | Table 1 external I2V baselines (CogVideoX1.5-5B-I2V, Wan2.2-TI2V-5B, Wan2.2-I2V-A14B, Cosmos-Predict2-2B): checkpoint download, DreamGenBench/EVAL-175/WorldArena inference, and judging under the same protocol as our models. |

## Contents

| Directory | What it does |
|---|---|
| [`dreamgenbench/`](dreamgenbench/) | Main benchmark pipeline on the GigaWorld-0 backbone: `kjob_pack_gr1_finetune_data.sh`, `kjob_train_gr1_finetune.sh` (Standard SFT), `kjob_gr1_dreamgen_generation.sh` (8-GPU generation), endpoint/official Qwen, Gemini, and GPT IF judges, EVAL-175 and CFG-grid kjobs, and score summarizers. |
| [`worldarena/`](worldarena/) | WorldArena 1.0 evaluator setup (`prepare_evaluator.sh` + patch), the eight component metrics (`local_metric_eval.py`, `aggregate_core_scores.py`), and the MLR protocol (`mlr_dispatch.py`, `mlr_eval.py`, `recompute_video_first_mlr.py`). |
| [`ewmbench/`](ewmbench/) | AgiBot-side data chain (`kjob_agibot_detect.sh`, `kjob_agibot_aug_prep.sh`, `kjob_agibot_pack.sh`) and EWMBench evaluation (`kjob_ewmbench_8gpu_serial.sh`, `ewmbench_layout_convert.py`, `kjob_ewmbench_eval_serial.sh`, `kjob_wmb_judge_serial.sh`). |
| [`robotwin_flowwam/`](robotwin_flowwam/) | FlowWAM training/probe/generation kjobs (`kjob_eve_flowwam_train.sh`, `kjob_flowwam_tia_probe.sh`, `kjob_flowwam_arm_generate.sh`), fidelity metrics (`flowwam_fidelity_metrics.py`, `flowwam_lpips_flow_epe.py`), GroundingDINO annotation and MLR (`kjob_flowwam_gdino_anno.sh`, `kjob_mlr_cfg_flowwam.sh`), plus checkpoint-screening and aggregation tools. |
| [`pbench/`](pbench/) | PBench input preparation (`prepare_pbench_it2v.py`), serial generation kjobs (`kjob_pbench_8gpu_serial.sh`), judges (`eval_pbench_robot_qwen_vqa.py`, `eval_pbench_robot_vbench_quality.py`, `run_videophy_pa2.py`), and summary tools. |
| [`baselines/`](baselines/) | `xmodel_download/dl_*.sh` fetch the external checkpoints; `xmodel_infer/` runs their DreamGenBench, EVAL-175, and WorldArena inference; `run_xmodel_eval175_judge.sh` judges baseline outputs. |

## Usage

All pipelines share the same conventions:

- `EVEWORLD_ROOT` must point to this repository root. Data-side roots default to the `GAGI_ROOT` convention (`/data/datasets/gagi`) defined in `eveworld/common/env.sh`; most scripts also accept explicit path overrides.
- `kjob_*.sh` scripts are SLURM-style cluster job payloads (`#SBATCH` headers, 1 or 8 GPUs) and `launch_*.sh` scripts are their submit-side wrappers. They take overrides as `KEY=VALUE` command-line arguments, e.g. `MAX_STEPS=1 GPU_IDS=0`.
- GPU payloads refuse to run on the workspace/login host; submit them through the matching `launch_*` wrapper, or set `ALLOW_LOCAL_RUN=1` on a GPU node for debugging.
- Typical flow per benchmark: generate videos (side-by-side layout) -> post-process into the benchmark's official layout -> run judges/metrics -> summarize. Generated videos feed downstream evaluation in other directories (e.g. EVAL-175 outputs feed the MLR stack in `eveworld/pipeline/` and [`worldarena/`](worldarena/)).

Start with the per-benchmark README where one exists ([`dreamgenbench/README.md`](dreamgenbench/README.md), [`worldarena/README.md`](worldarena/README.md)).

## Notes

- **Cluster scripts must be adapted.** Every `kjob_*` / `launch_*` script carries absolute defaults (dataset roots, conda installations, checkpoint paths) that point at internal storage. Override `EVEWORLD_ROOT`, `GAGI`/`GAGI_ROOT`, `CONDA_SH`, `CONDA_ENV`, `TRAIN_VENV`, and the data/output roots for your site; the scripts are templates, not turnkey installers.
- **Dependencies.** The `EVEWorld` conda environment covers generation and most judges (see [`docs/ENVIRONMENT.md`](../docs/ENVIRONMENT.md)); training uses an isolated venv (`giga-world-0/scripts/setup_gigaworld_train_venv.sh`), and some benchmarks need their own venv or external evaluator (e.g. `dreamgenbench/setup_dreamgenbench_eval_venv.sh`, `worldarena/prepare_evaluator.sh`, `pbench/setup_videophy_env.sh`).
- **Data prerequisites.** Benchmark inputs (GR1 fine-tuning split, DreamGenBench task JSONs, AgiBot episodes, RoboTwin held-out actions, WorldArena prompts) are not vendored into this repository; see the per-benchmark READMEs for the expected on-disk layouts. Every stage validates exact coverage and fails loudly rather than evaluating a partial set.
- **API judges.** Qwen-VL, Gemini, and GPT IF judges call OpenAI/GenAI-compatible endpoints; set the endpoint URL and token environment variables documented in each benchmark README. No credentials are stored in the repository.
