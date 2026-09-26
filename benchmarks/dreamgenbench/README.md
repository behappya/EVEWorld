# DreamGenBench: Standard SFT, Generation, and IF Judging

End-to-end DreamGenBench pipeline on the GigaWorld-0 backbone: pack the GR1 fine-tuning split, train the Standard SFT control (and EVEWorld variants configured from `eveworld/pipeline/`), generate videos, and judge instruction following with Qwen / Gemini / GPT evaluators. This directory also hosts the EVAL-175 campaigns, the CFG grid sweep, and the sequence-length sweeps.

## Paper mapping

- **Section 5.1 "Experimental Setup and Metrics"**: the 92-video GR1 fine-tuning split is packed by `kjob_pack_gr1_finetune_data.sh`; the Standard SFT recipe (200 steps, effective batch size 64, 93 frames at 480x768 / 16 FPS, bf16, EMA) is implemented by `kjob_train_gr1_finetune.sh` using the `configs.giga_world_0_video_gr1_finetune` base config from [`giga-world-0/`](../../giga-world-0/).
- **Section 5.2 "Main Results", Table 1** (`tab:dreamgen_overall`) **and the per-split table** (`tab:dreamgen_main`): EVAL-175 generation (`kjob_eval175_*.sh` over `eveworld/evaluation/` dispatchers) plus the Qwen-IF judges here; Gemini-IF runs the same Qwen-IF prompt protocol through `eveworld/evaluation/eval_gemini_dreamgen_qwen_protocol.py` (see `kjob_cfg_gemini_if.sh` for the wiring); GPT-IF is the third instruction-following judge of the original DreamGen protocol.
- **Section 5.4 "Ablations and Analysis"**: CFG sensitivity (`fig:cfg`, `tab:cfg_main`) is the 24-cell grid from `kjob_cfg_train_then_sweep.sh` + `kjob_cfg_grid_serial.sh` + `build_cfg_gemini_manifest.py` + `kjob_cfg_gemini_if.sh`; sequence-length scaling (`tab:horizon_compute_main`) uses the `run_*_length_*` / `run_short_length_*` sweeps.
- **Appendix MLR protocol** (`app:mlr_protocol`): videos generated here feed the GroundingDINO counting stack in [`eveworld/pipeline/`](../../eveworld/pipeline/) (`t4g_exam_v2.py`, `kjob_eval_t4g_exam_v2.sh`) and the frozen protocol runner in [`../worldarena/`](../worldarena/) (`mlr_eval.py`); MLR itself is not computed in this directory.

## Contents

| File | What it does |
|---|---|
| **Setup** | |
| `download_dreamgenbench_code.sh` | Clones the official [NVIDIA GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) DreamGenBench code (LFS smudge disabled). |
| `download_dreamgenbench_qwen25vl7b.sh` | Downloads `Qwen/Qwen2.5-VL-7B-Instruct` into the shared HF cache for the official local judge. |
| `setup_dreamgenbench_eval_venv.sh` | Creates an isolated venv (pinned transformers, `qwen-vl-utils`, `decord`) so the official judge does not touch the `EVEWorld` conda env. |
| **Data packing and training** | |
| `kjob_pack_gr1_finetune_data.sh` + `launch_gr1_pack_data_kjob.sh` | Packs `raw_data/*.mp4` + `*.txt` pairs into `packed_data/` with the T5-11B text encoder (`giga-world-0/scripts/pack_data.py`); guards against partial output via a work directory. |
| `kjob_train_gr1_finetune.sh` + `launch_gr1_train_kjob.sh` | Standard SFT: writes a runtime config from the base config module, trains with the isolated train venv, records GPU-memory samples, and verifies the `Step[MAX/MAX]` marker plus the final checkpoint. |
| **Generation** | |
| `prepare_gr1_dreamgen_inputs.py` | Builds the image-to-video input JSON (first frame + instruction per GR1 sample) consumed by all generation paths. |
| `kjob_gr1_dreamgen_generation.sh` + `launch_gr1_dreamgen_generation_kjob.sh` | 8-GPU batch generation with a fine-tuned checkpoint (EMA weights by default); writes side-by-side mp4s and `generation_summary.json`. |
| `launch_gr1_finetuned_serve_kjob.sh` + `run_gr1_dreamgen_generation.sh` | Alternative serving path: host a checkpoint as an HTTP service inside a kjob, then call it from the client script. |
| `kjob_dreamgen_multiseed_serial.sh` | Multi-seed generation for a single task/request (seed-sensitivity analysis). |
| `prepare_dreamgen_eval_videos.py` | Crops side-by-side outputs to generated-only videos (right half) with ffmpeg. |
| **IF / PA judging** | |
| `kjob_dreamgenbench_qwen_eval.sh` + `launch_dreamgenbench_qwen_eval_kjob.sh` | Official local judge: `dreamgenbench.eval_sr_qwen_whole` (Qwen-IF) and `dreamgenbench.eval_qwen_pa` (PA-I) with Qwen2.5-VL-7B; requires the setup scripts above. |
| `eval_dreamgenbench_qwen_api.py` / `.sh` | Endpoint judge: Qwen-VL behind an OpenAI-compatible server (`QWEN_BASE`); metrics `qwen_if,pa_i`; resumable per-video CSVs with retries. |
| `eval_dreamgenbench_gpt_if_api.py` / `.sh` | GPT-IF judge through a GenAI-compatible gateway (`DIFROST_*` env vars); async, resumable CSVs. |
| `kjob_cfg_gemini_if.sh` | Gemini-IF judging of the CFG grid via `eveworld/evaluation/eval_gemini_dreamgen_qwen_protocol.py`. |
| `run_completed_dreamgen_task_completion_eval.sh` | Batch Qwen-IF (optionally GPT-IF / PA-I) over all completed SFT/Pretrain length runs; skips finished CSVs and writes a summary table. |
| `run_missing_qwen58_evals.sh` / `run_dreamgen_gpt_if_upto19_missing.sh` | Resumable backfills for endpoint-Qwen and GPT-IF judgments. |
| **EVAL-175 campaigns** | |
| `kjob_eval175_gen.sh` / `kjob_eval175_serial_gen.sh` | Official-split generation (126 prompts across `eval175_gr1_{env,object,behavior}.json`), single seed; parallel or serial model x split placement. |
| `kjob_eval175_multiseed.sh` / `kjob_eval175_seed_sharded.sh` | Multi-seed and matched-seed sharded EVAL-175 generation. |
| `run_indomain_eval175_rejudge.sh` | Re-judges all models' EVAL-175 outputs with one judge endpoint for cross-run comparability. |
| **CFG grid and length sweeps** | |
| `kjob_cfg_train_then_sweep.sh` | Trains the seed-42 reproduction to step 300, assembles per-step anchor roots, then runs the CFG sweep. |
| `kjob_cfg_grid_serial.sh` | Serial generation over the 24-cell grid: steps {50..300} x CFG weights via `eveworld/evaluation/cfg_grid_serial_dispatch.py`. Default grid `1.0 2.5 5.0 7.0` (the paper's `w=7.0` setting); override with `CFG_VALUES="1.0 2.5 5.0 7.5"` for the wider grid. |
| `build_cfg_gemini_manifest.py` | Builds the 3,024-row Gemini-IF manifest for the grid and validates every video exists; `--cfg-values` must match the sweep grid. |
| `run_sft_pretrain_dreamgen_long_pa2_sweep.sh` / `run_pretrain_dreamgen_length_pa2_sweep.sh` / `run_short_length_generation_only.sh` / `run_short_length_dreamgen_only.sh` | Sequence-length sweeps (3.8s-29.8s at 16 FPS): 8-GPU generation per length plus VideoPhy PA-II (runner in [`../pbench/`](../pbench/)). |
| **Summaries** | |
| `summarize_dreamgenbench_csv.py` / `summarize_dreamgenbench_full.py` | Average binary Qwen-IF/GPT-IF/PA-I CSVs into a JSON summary; the "full" variant also thresholds VideoPhy PA-II and reports `PA = mean(PA-I, PA-II)`. |

EVAL-175 summaries (per-split aggregation, Wilson intervals) are produced by `eveworld/evaluation/eval175_summarize.py`.

## Usage

`EVEWORLD_ROOT` must point to this repository root. All `kjob_*` / `launch_*` scripts accept `KEY=VALUE` overrides; data roots default to the `GAGI_ROOT` convention (`/data/datasets/gagi`, see `eveworld/common/env.sh`).

1. **One-time setup** (only for the official local judge):

   ```bash
   ./benchmarks/dreamgenbench/download_dreamgenbench_code.sh
   ./benchmarks/dreamgenbench/download_dreamgenbench_qwen25vl7b.sh
   ./benchmarks/dreamgenbench/setup_dreamgenbench_eval_venv.sh
   ```

2. **Pack the GR1 split** (expects `raw_data/` with matched `*.mp4`/`*.txt` pairs and the pretrained `text_encoder/`):

   ```bash
   EVEWORLD_ROOT=/path/to/EVEWORLD ./benchmarks/dreamgenbench/launch_gr1_pack_data_kjob.sh
   ```

3. **Train Standard SFT** (8 GPUs; writes checkpoints under `OUTPUT_ROOT/experiments`):

   ```bash
   EVEWORLD_ROOT=/path/to/EVEWORLD ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh
   ```

   Use `MAX_STEPS=1 CONFIG_DRY_RUN=1` for a smoke test. EVEWorld variants reuse the same payload with a different `BASE_CONFIG_MODULE` from `eveworld/pipeline/`.

4. **Generate** (batch kjob path; auto-prepares the it2v JSON if missing):

   ```bash
   EVEWORLD_ROOT=/path/to/EVEWORLD CHECKPOINT_DIR=/path/to/checkpoint \
     ./benchmarks/dreamgenbench/launch_gr1_dreamgen_generation_kjob.sh
   ```

   Then crop to generated-only videos: `SOURCE_VIDEO_DIR=<save_dir> python benchmarks/dreamgenbench/prepare_dreamgen_eval_videos.py` (or use the serving path: `launch_gr1_finetuned_serve_kjob.sh` + `URL=http://<host>:8000 run_gr1_dreamgen_generation.sh`).

5. **Judge.** Official local Qwen: `VIDEO_DIR=<generated_only_dir> ./benchmarks/dreamgenbench/launch_dreamgenbench_qwen_eval_kjob.sh`. Endpoint Qwen: `VIDEO_DIR=<dir> QWEN_BASE=http://127.0.0.1:8000/v1 ./benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.sh`. GPT-IF: set `DIFROST_API_TOKEN` and run `eval_dreamgenbench_gpt_if_api.sh`.

6. **Summarize**: `python benchmarks/dreamgenbench/summarize_dreamgenbench_full.py --qwen-if-csv ... --output-json summary.json`.

## Notes

- **Cluster scripts must be adapted.** The `kjob_*` payloads carry `#SBATCH` headers and absolute defaults (dataset roots, conda paths, checkpoint paths) that point at internal storage; override `EVEWORLD_ROOT`, `GAGI`, `DATA_ROOT`, `OUTPUT_ROOT`, `CONDA_SH`, `CONDA_ENV`, `TRAIN_VENV`, etc. for your site. GPU payloads refuse to run on the workspace host unless `ALLOW_LOCAL_RUN=1`.
- **Dependencies.** Generation/judging run in the `EVEWorld` conda env ([`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md)); training uses the isolated venv from `giga-world-0/scripts/setup_gigaworld_train_venv.sh`; the official Qwen judge uses the venv from `setup_dreamgenbench_eval_venv.sh`. Endpoint judges need `openai`; GPT-IF additionally needs `google-genai`.
- **Data prerequisites.** The GR1 fine-tuning split (92 video/instruction pairs) and the GigaWorld-0 pretrained checkpoint (`transformer/`, `vae/`, `text_encoder/`) must be downloaded separately; EVAL-175 inputs live under the eval root's `giga_input/eval175_gr1_*.json`. Scripts validate counts (92 training samples, 126 eval prompts, 3,024 CFG cells) and fail loudly on gaps.
- **API credentials.** GPT-IF and Gemini-IF read `DIFROST_API_TOKEN` plus the `DIFROST_GENAI_BASE_URL` / `DIFROST_HOST` / `DIFROST_MODEL` gateway variables from the environment; no credentials are stored in this repository.
- The local 92-prompt runs (`run_completed_dreamgen_task_completion_eval.sh`, length sweeps) are in-domain diagnostics; the reported DreamGenBench numbers come from the EVAL-175 campaigns.
