# Cross-Model Baselines

Download, generate, and judge general-purpose image-to-video (I2V) models — **CogVideoX1.5-5B-I2V**, **Wan2.2-TI2V-5B**, **Wan2.2-I2V-A14B**, and **Cosmos-Predict2-2B** — as cross-model baselines showing that Model Laziness is a field-wide phenomenon, not specific to the GigaWorld-0 backbone.

## Paper mapping

- **Table 1 (`tab:dreamgen_overall`, Sec. 5.2 Main Results)** — the four "general video models" rows (MLR / Qwen-IF / Gemini-IF on DreamGenBench) come from the eval175 generation (`xmodel_infer/kjob_xmodel_eval175_serial.sh`) + judging (`run_xmodel_eval175_judge.sh`) pipeline.
- **Appendix "Model Laziness across Models" (`app:mlr_cross_model`, `tab:mlr_dgb`)** — cross-model MLR with Wilson CIs on the same generations.
- **Sec. 5.3 + Appendix "Cross-Domain Evaluation" (`tab:mlr_wa`, `tab:mlr_wa_permodel`)** — WorldArena 1.0 zero-shot transfer of the baselines, produced by `xmodel_infer/kjob_xmodel_worldarena1_serial.sh` together with [`../worldarena/`](../worldarena/).
- **Appendix "Training Dynamics, Guidance, and Output Length" (`tab:cross_model_laziness`)** — long-horizon process evaluation (3.8 s / 7.8 s tiers on 92 matched prompts); videos come from `xmodel_infer/run_all_xmodel_dreamgen.sh` / `run_xmodel_fill_missing.sh` and are scored with `eveworld/evaluation/tea/qwen_laziness.py`.
- **Appendix "Evaluation Stability" (`tab:dreamgen_gemini_sd`)** — repeated Gemini-IF judging reuses the eval175 baseline videos.

## Contents

| Path | What it does |
|---|---|
| `xmodel_download/_common.sh` | Shared downloader helpers: conda/`hf` CLI setup, resumable `dl_repo` wrapper (per-flag `--include/--exclude`), post-download `verify_paths` check. Weights land under `${GAGI_ROOT}/xmodels/` (default `/data/datasets/gagi/xmodels/`), outside git. |
| `xmodel_download/dl_wan22_ti2v_5b.sh` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (~10–12 GB, not gated). |
| `xmodel_download/dl_cogvideox15_5b_i2v.sh` | `zai-org/CogVideoX1.5-5B-I2V` (~10–11 GB, gated). |
| `xmodel_download/dl_wan22_i2v_a14b.sh` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (~55–65 GB, not gated). |
| `xmodel_download/dl_cosmos_predict25_2b.sh` | `nvidia/Cosmos-Predict2.5-2B` (gated; the Cosmos-Predict2-2B baseline). Non-standard repo layout, so verification only checks the directory is non-empty. |
| `xmodel_infer/xmodel_dreamgen_infer.py` | Single batch-I2V entry point. `--model-family {wan,wan_ti2v,cogvideox,cosmos}` selects the diffusers pipeline; reads a `(image, prompt, request_id)` manifest; multi-GPU data parallelism via `--gpu-ids`; skips existing mp4s (resumable); writes per-shard files plus a merged `generation_summary.json`. `--dry-run` validates inputs on a CPU-only host. |
| `xmodel_infer/kjob_xmodel_dreamgen_infer.sh` | SLURM payload (8×GPU node) running one model × one duration tier on the 92-prompt DreamGen manifest; all settings overridable as `KEY=VALUE` args. |
| `xmodel_infer/launch_xmodel_dreamgen_infer_kjob.sh` | Submitter for the above (via `kjobctl create slurm`); forwards env overrides into the job. |
| `xmodel_infer/run_all_xmodel_dreamgen.sh` | Sequential sweep: each downloaded model × duration tiers {5.8 s = 93 f, 9.8 s = 157 f, 15.8 s = 253 f @ 16 fps}, submitting one cluster job per pair and polling for its summary. Overrides: `SMOKE=1`, `DURS=`, `MODELS=`. |
| `xmodel_infer/run_xmodel_fill_missing.sh` | Idempotent gap-filler for the extra horizons used by the long-horizon process eval (3.8 s = 61 f, 7.8 s = 125 f; optional CogVideoX 15.8 s rerun via `INCLUDE_COG158=1`). |
| `xmodel_infer/kjob_xmodel_eval175_serial.sh` | Paper-protocol DreamGenBench generation: one 8-GPU node loops 3 models × 3 splits (`gr1_env` / `gr1_object` / `gr1_behavior`, 126 prompts total), 93 f @ 16 fps, seed 42. Per-split summaries make it resumable. |
| `xmodel_infer/kjob_xmodel_worldarena1_serial.sh` | WorldArena 1.0 chain per model: generate (125 f @ 24 fps) → normalize to 121 f ([`../worldarena/normalize_xmodel_videos.py`](../worldarena/normalize_xmodel_videos.py)) → core metrics ([`../worldarena/kjob_worldarena1_eval_serial.sh`](../worldarena/kjob_worldarena1_eval_serial.sh)) → MLR ([`../worldarena/mlr_dispatch.py`](../worldarena/mlr_dispatch.py), [`../worldarena/recompute_video_first_mlr.py`](../worldarena/recompute_video_first_mlr.py)) → aggregate ([`../worldarena/aggregate_core_scores.py`](../worldarena/aggregate_core_scores.py)). |
| `run_xmodel_eval175_judge.sh` | One-shot eval175 judging: format/seed audit + manifest build with `eveworld/evaluation/eval175_prepare.py`, then parallel Qwen-IF + PA-I judging of all three open models with [`../dreamgenbench/eval_dreamgenbench_qwen_api.py`](../dreamgenbench/eval_dreamgenbench_qwen_api.py), then prints per-model summaries. |

## Usage

```bash
# 0. Environment: repo root + data root (see eveworld/common/env.sh)
export EVEWORLD_ROOT=/path/to/EVEWorld
export GAGI_ROOT=/data/datasets/gagi        # cluster data root; adapt to your storage

# 1. Download weights (CPU-only, resumable; run several in parallel terminals)
bash benchmarks/baselines/xmodel_download/dl_wan22_ti2v_5b.sh
bash benchmarks/baselines/xmodel_download/dl_cogvideox15_5b_i2v.sh   # gated: hf auth login first
bash benchmarks/baselines/xmodel_download/dl_wan22_i2v_a14b.sh
bash benchmarks/baselines/xmodel_download/dl_cosmos_predict25_2b.sh  # gated: accept NVIDIA license + hf auth login

# 2. Multi-duration DreamGen generation (92 prompts; submits cluster jobs)
SMOKE=1 bash benchmarks/baselines/xmodel_infer/run_all_xmodel_dreamgen.sh  # 4-prompt smoke first
bash benchmarks/baselines/xmodel_infer/run_all_xmodel_dreamgen.sh
bash benchmarks/baselines/xmodel_infer/run_xmodel_fill_missing.sh          # 3.8 s / 7.8 s tiers

# 3. eval175 paper protocol: generate, then judge (needs a Qwen judge endpoint)
#    submit benchmarks/baselines/xmodel_infer/kjob_xmodel_eval175_serial.sh as a cluster job, then:
QWEN_BASE=http://127.0.0.1:8000/v1 bash benchmarks/baselines/run_xmodel_eval175_judge.sh

# 4. WorldArena 1.0 transfer chain (generate + core metrics + MLR), submitted as a cluster job
#    see benchmarks/baselines/xmodel_infer/kjob_xmodel_worldarena1_serial.sh
```

Single-run generation (one model, one tier) goes through the launcher, e.g.:

```bash
MODEL_FAMILY=wan_ti2v MODEL_PATH=${GAGI_ROOT}/xmodels/wan22_ti2v_5b \
RUN_NAME=wan22_ti2v_5b_5p8s NUM_FRAMES=93 \
bash benchmarks/baselines/xmodel_infer/launch_xmodel_dreamgen_infer_kjob.sh
```

## Notes

- **Cluster scripts.** All `kjob_*.sh` / `launch_*.sh` files are SLURM-style wrappers (`#SBATCH` headers, `kjobctl` submission, 8-GPU nodes) written for the authors' cluster. Absolute paths (`/data/datasets/gagi`, conda envs `giga_world1` / `giga_models` / `WorldArena`, `~/miniconda`) **must be adapted** to your environment before use. The generation payloads refuse to run on a GPU-less workspace host unless `ALLOW_LOCAL_RUN=1`; `xmodel_dreamgen_infer.py --dry-run` is the intended CPU-only sanity check.
- **Stale path in the judge script.** `run_xmodel_eval175_judge.sh` predates the repo reorganization: its `cd ${REPO}/eveworld/evaluation` assumed a copy of `eveworld/` inside `giga-world-0/`. In this repo the prepare script lives at `eveworld/evaluation/eval175_prepare.py`; point the script there (and run from the repo root) before executing. Its other dependency, [`../dreamgenbench/eval_dreamgenbench_qwen_api.py`](../dreamgenbench/eval_dreamgenbench_qwen_api.py), is current.
- **Judging prerequisites.** The judge script expects completed eval175 generations for `wan22_ti2v_5b`, `cogvideox15_5b_i2v`, and `wan22_i2v_a14b`, and an OpenAI-compatible Qwen endpoint (`QWEN_BASE`, default `http://127.0.0.1:8000/v1`); see [`../dreamgenbench/`](../dreamgenbench/) for judge serving. Wan models are audited at 93 f / 768×480; CogVideoX at its native 96 f / 1360×768.
- **Frame-count constraint.** For CogVideoX, `(num_frames - 1) // 4 + 1` must be even, so 61 / 93 / 125 / 157 / 253 frames are valid while 97 / 161 / 257 are not.
- **Model naming.** The Cosmos baseline is downloaded as `nvidia/Cosmos-Predict2.5-2B` (the infer family map also accepts `cosmos_predict2_2b_v2w`); the paper reports it as Cosmos-Predict2-2B. Its pipeline uses a passthrough safety checker because the official guardrail requires online downloads; inputs are fixed benchmark prompts.
- **Dependencies.** Generation needs the `giga_world1` conda env (diffusers ≥ 0.39 with all four I2V pipelines); judging uses the `giga_models` env; WorldArena core metrics use the separate WorldArena evaluator env (see [`../../docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md)). Downloading uses the `hf` CLI and is CPU-only.
