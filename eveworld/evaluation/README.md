# DreamGenBench Evaluation (`eveworld/evaluation`)

Generation dispatch, judge scoring, and statistical aggregation for the DreamGenBench
evaluation campaigns (internally named "EVAL-175") reported in the EVEWorld paper:
audited manifests, Qwen-IF / GPT-IF / Gemini-IF / PA-I / PA-II judging, multi-seed and
CFG-sweep generation, and process-laziness judge-agreement tooling.

## Paper mapping

- **Sec. 5.1–5.2 (setup & metrics), Table 1 (`tab:dreamgen_overall`), Table 2
  (`tab:dreamgen_main`)** — the `eval175_*` prepare → judge → summarize family, plus
  `eval_gemini_dreamgen_qwen_protocol.py` (Gemini-IF under the official protocol).
- **Table 4 (`tab:component_ablation`)** — `eval175_ablation_summarize.py` /
  `eval175_ablation_repeats_summarize.py` (fixed-seed ablation arms, repeated judging).
- **Appendix "Evaluation Stability" (`tab:dreamgen_gemini_sd`, `tab:matched_seed`)** —
  `eval175_multiseed_*` / `dreamgen_multiseed_*` generation, judge-repeat aggregators,
  `eval175_raw_ema_compare.py`, `failure_routes_gemini_serial.py`.
- **Appendix "Training Dynamics, Guidance, and Output Length" (`fig:cfg`,
  `tab:cfg_main`)** — `cfg_grid_serial_dispatch.py` + `cfg_grid_worker.py` +
  `summarize_cfg_gemini_repeats.py` (6 checkpoints × 4 CFG weights).
- **Appendix "Model Laziness and MLR"** — process-laziness instrumentation in `tea/` and
  `x11_process_judge.py`; the MLR detector itself lives in
  [`eveworld/pipeline/t4g_gdino.py`](../pipeline/t4g_gdino.py) and
  [`benchmarks/worldarena/mlr_eval.py`](../../benchmarks/worldarena/mlr_eval.py).

## Contents

### EVAL-175 judging pipeline (prepare → judge → summarize)

| File | What it does |
|---|---|
| `eval175_prepare.py` | Audits generated videos (frame count, resolution, fps probe), emits per-model/per-split scorer manifests and PA-II staging links. Expects the official splits `gr1_env`/`gr1_object`/`gr1_behavior` (29/50/47 prompts). |
| `eval175_qwen_local.py` | Official DreamGenBench Qwen-IF and PA-I scoring with a local Qwen2.5-VL-7B checkpoint. Resumable CSV output. |
| `eval175_gpt_if.py` | GPT-IF scoring of audited manifests through an OpenAI-compatible endpoint. |
| `eval175_summarize.py` | Strictly validates scorer CSVs against manifests; writes `eval175_scores.{json,csv,md}` (overall + per-split, vs. a baseline). |
| `eval175_compare_pa2.py` | Paired VideoPhy PA-II comparison (Wilcoxon + bootstrap CI), baseline vs. candidate. |
| `eval175_force_unresolved_zero.py` | Resolves missing/errored judge rows as zero while preserving a hashed audit trail. |
| `eval175_ablation_summarize.py` / `eval175_ablation_repeats_summarize.py` | Validate and summarize the fixed-seed component-ablation runs, single and repeated respectively (optional MLR summary merge). |
| `eval175_raw_ema_compare.py` | Builds a paired raw-weights vs. EMA table from two repeated-run summaries. |
| `eval175_kjob_qwen.sh`, `eval175_launch_qwen_kjob.sh` | SLURM-style cluster wrapper + launcher for the local Qwen judge. |

### Multi-seed generation (evaluation-stability campaigns)

| File | What it does |
|---|---|
| `eval175_multiseed_dispatch.py` / `eval175_multiseed_worker.py` | Shards inference seeds across the 8 GPUs of one node; each worker renders all 126 tasks for its seeds. |
| `eval175_seed_sharded_dispatch.py` | Generates each seed serially, sharding the 126 tasks over 8 GPUs per seed. |
| `eval175_multiseed_prepare.py` | Audits per-seed output directories and builds one Gemini manifest per seed. |
| `dreamgen_multiseed_dispatch.py` / `dreamgen_multiseed_worker.py` | Renders one DreamGenBench item (`--request-id`) across several seeds and models on one node. |

### Gemini judges

| File | What it does |
|---|---|
| `eval_gemini_dreamgen_qwen_protocol.py` | Gemini-IF: Gemini under the exact DreamGenBench Qwen-IF/PA-I protocol (reuses the protocol code from [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/)). |
| `eval_gemini_dreamgen_process.py` | Gemini physical-process judge (instruction completion, physical validity, target conservation, duplicate shortcut); writes `records.jsonl` + `summary.json`. |
| `gemini_consensus_judge.py` | Repeated-sample Gemini answering over a multiple-choice embodied-QA set stored in Lance format, with a served Qwen judge grading attempts into a per-question consensus summary (JSONL + Lance). |
| `failure_routes_gemini_serial.py` | Three resumable Gemini-IF repeats over the uniform negative-route failure set (1,048 manifest rows), with automatic error-retry passes. |

### CFG sweeps

| File | What it does |
|---|---|
| `cfg_grid_serial_dispatch.py` / `cfg_grid_worker.py` | Generates the training-step × CFG-weight grid (6 checkpoints × 4 weights, 126 prompts per cell); each checkpoint block uses all 8 GPUs via per-GPU workers. |
| `summarize_cfg_gemini_repeats.py` | Validates and aggregates repeated Gemini-IF runs over the 24-cell grid. |

### Dispatch chains and pools (x6–x13, l0, p0, j1)

| File | What it does |
|---|---|
| `x6_download_gr1_2b.sh` | Resumable Hugging Face download of the official `GigaWorld-0-Video-GR1-2b` checkpoint. |
| `x8_assemble_compare.py` | Assembles the unified multi-arm comparison folder (idempotent; re-run to fill in newly finished arms). |
| `x9_eval175_dispatch.py` | Core dispatcher: pins each (model × split) unit to one GPU; holds the `MODELS` registry imported by the other dispatchers. |
| `x11_process_judge.py` | Four-dimension process judge (count persistence, transport/teleport, termination, actor) against a served VLM; `calibrate` scores agreement against human labels, `score` judges a video directory. |
| `x12_eval175_autochain.sh` | Polls generation completion → builds manifests → runs the Qwen judge per arm. |
| `x13_eval175_serial_dispatch.py` | Runs model groups sequentially, calling `x9_eval175_dispatch.py` per batch. |
| `l0_longhorizon_pool.sh` | Generates the 7.8 s (125-frame) long-horizon baseline pool (92 prompts × 8 seeds). |
| `p0_probe_arm.sh` / `p0_probe_score.sh` | Fast checkpoint probe: generate on a prompt subset, score with `tea/qwen_laziness.py`, compare prompt-paired against the Round-0 pool. |
| `j1_score_pool.sh` | Batch-scores a generation pool with both `qwen_laziness.py` judge variants. |
| `launch_eval_kjob.sh` | Cluster orchestration entry point (prints the generation → process-metrics → stats plan). |

### `tea/` — process-laziness judges and agreement statistics

| File | What it does |
|---|---|
| `tea/qwen_laziness.py` | VLM process-laziness judge: checks the contact→grasp→transport→release→settle causal chain for physically impossible shortcuts (Q1–Q5) plus a 0–4 severity score; two prompt variants (`--judge a\|b`). |
| `tea/ncm.py` | Optical-flow motion-dynamics laziness signatures (TELE / JUMP / STILL / ROUGH); `score` a video directory, `validate` against real anchor videos. |
| `tea/compare_qwen.py` / `compare_qwen_repeats.py` / `compare_qwen_training_seeds.py` | Paired bootstrap statistics for laziness CSVs: single runs, judge-repeat averages, and prompt-clustered training/generation-seed comparisons. |
| `tea/compare_ncm.py` | Paired statistics for `ncm.py` score JSONs. |
| `tea/prepare_qwen_blind_review.py` | Builds a randomized, paired folder for a blinded human process audit. |

### Cross-campaign statistics

| File | What it does |
|---|---|
| `aggregate_stats.py` | Multi-model, multi-seed aggregation of process-metric JSONs: bootstrap 95% CIs plus a McNemar-style paired hint against the baseline. Pure CPU. |
| `run_stats.sh` | Thin wrapper that sources [`eveworld/common/env.sh`](../common/env.sh) and runs `aggregate_stats.py`. |

## Usage

Environment variables follow the shared contract in [`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md):
`EVEWORLD_ROOT` (repository root) and `GAGI_ROOT` (data root). Gemini access is configured
via `DIFROST_API_TOKEN`, `DIFROST_GENAI_BASE_URL`, and `DIFROST_MODEL`; OpenAI-compatible
judges via `OPENAI_API_KEY` / `OPENAI_BASE_URL`; a locally served Qwen judge via
`--qwen-base`. Typical pipeline:

```bash
# 1. Audit generations and build scorer manifests
python eveworld/evaluation/eval175_prepare.py \
  --generation-root "$GAGI_ROOT/eve_v2_outputs/eval175_gen" \
  --input-root      "$GAGI_ROOT/gr1_dreamgen_eval/giga_input" \
  --output-root     "$GAGI_ROOT/eve_v2_outputs/eval175_eval"

# 2a. Judge with a local Qwen2.5-VL (Qwen-IF + PA-I)
python eveworld/evaluation/eval175_qwen_local.py \
  --manifest   "$GAGI_ROOT/eve_v2_outputs/eval175_eval/manifests/<model>.jsonl" \
  --output-dir "$GAGI_ROOT/eve_v2_outputs/eval175_eval/results/<model>"

# 2b. Or judge with Gemini under the same protocol (Gemini-IF)
python eveworld/evaluation/eval_gemini_dreamgen_qwen_protocol.py \
  --manifest   "$GAGI_ROOT/eve_v2_outputs/eval175_eval/manifests/<model>.jsonl" \
  --output-dir "$GAGI_ROOT/eve_v2_outputs/gemini_eval/<model>"

# 3. Validate and summarize all models
python eveworld/evaluation/eval175_summarize.py \
  --manifest-root "$GAGI_ROOT/eve_v2_outputs/eval175_eval/manifests" \
  --results-root  "$GAGI_ROOT/eve_v2_outputs/eval175_eval/results" \
  --output-dir    "$GAGI_ROOT/eve_v2_outputs/eval175_eval/summary"
```

**Cluster-job caveat.** The `kjob_*` / `launch_*` wrappers and the `x6`–`x13`, `l0`,
`p0`, `j1` scripts are SLURM-style cluster launchers from the authors' infrastructure.
Their absolute data-root paths, conda activation lines, submit helpers, and the `MODELS`
registry in `x9_eval175_dispatch.py` must be adapted to your environment before use.
Generation workers import the vendored backbone; run them with `EVEWORLD_ROOT` on
`PYTHONPATH` (or from the repo root).

## Notes

- Dependencies: `torch` + `transformers` (local Qwen judge), `google-genai` and `openai`
  clients (API judges), `opencv`/`imageio` (video I/O), `numpy`/`scipy` (statistics), and
  `lance` + `pyarrow` (only for `gemini_consensus_judge.py`). Generation dispatchers call
  [`eveworld/method/scripts/generate_eag.py`](../method/scripts/generate_eag.py) on the
  vendored `giga-world-0` / `giga-models` stack.
- Data prerequisites: the DreamGenBench prompt JSONs and conditioning images
  (`eval175_gr1_{env,object,behavior}.json`), prepared by
  [`benchmarks/dreamgenbench/prepare_gr1_dreamgen_inputs.py`](../../benchmarks/dreamgenbench/prepare_gr1_dreamgen_inputs.py);
  the official protocol scripts live in [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/).
  The official GR1 fine-tune checkpoint is fetched by `x6_download_gr1_2b.sh`.
- `x11_process_judge.py calibrate` additionally requires a human-labeled calibration pack
  (grid images + gold labels) that is not shipped in this repository; `score` mode only
  needs a served VLM endpoint.
- All scorers are resumable (`--resume` is on by default) and write outputs atomically,
  so interrupted judging runs can simply be re-executed.
