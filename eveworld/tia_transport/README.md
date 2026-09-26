# TIA Transport Adapter (`tia_transport/`)

Isolated architecture experiment for the TIA **cross-frame instance transport**: a
zero-initialized transport adapter is inserted after the correspondence-probed
Transformer block of GigaWorld-0, trained with the *unchanged* full EVEWorld
objective in a paired control-vs-transport campaign, and scored with multi-seed /
multi-checkpoint evaluation campaigns. Code name: **CIC-Transport** (the adapter
serves the contrastive instance-correspondence objective, `L_TIA`).

## Paper mapping

- **Section 4.2 "Temporal Instance Alignment"** (`sec:tia`) and **Appendix "TIA
  Cross-Frame Alignment"** (`app:tia_details`, Eq. for the transport update and
  Algorithm `alg:tia_transport`): `cic_transport_transformer.py` implements the
  dense local matcher — each cell of frame `t-1` queries a local window in frame
  `t`, cosine similarities become softmax transport weights at temperature τ, the
  projected previous-frame features are splatted and mass-normalized, and the
  update is written back as a bounded `γ·tanh(W_out(·))` residual with `W_out`
  zero-initialized (identity before training; first frame unchanged).
- **Appendix "TIA Layer Selection"** (`app:tia_layer_selection`, Table
  `tab:tia_layer_probe`): `block22` is the insertion point recorded in the code
  provenance of the paired campaign; the paper table quotes the same probe run
  with **block 23** as its minimum-error row. Both are shipped side by side —
  `cic_transport_config.py` keeps `block22` untouched (the campaign preflight is
  bit-exact against it), while `cic_transport_b23_config.py` sets both
  `t4g_id_block` and `cic_transport_after_block` to `block23` for a fresh run.
- **Appendix "Implementation Details"** (`app:training_setup`): the shipped
  defaults — rank 64, 7×7 window (`window_radius=3`), τ=0.07, γ=0.1 — match
  `cic_transport_config.py`.
- **Appendix "Evaluation Stability"** (`app:evaluation_stability`): the
  three-repeat Qwen-IF/Gemini-IF audit scripts quantify judge-repeat variance.

The contrastive correspondence loss and the IGR objective themselves live in the
shared trainers ([`../pipeline/t4g_joint_trainer.py`](../pipeline/t4g_joint_trainer.py),
[`../pipeline/t4g_corr_trainer.py`](../pipeline/t4g_corr_trainer.py)); this
directory swaps in the transport-carrying transformer so the architecture change
is isolated against an exactly paired control run.

## Contents

### Core implementation

| File | What it does |
|---|---|
| `cic_transport_transformer.py` | `CICTransportAdapter` (local matching + zero-init residual transport, sentinel stats) and `CICTransportGigaWorld0Transformer3DModel` (registers the adapter as a forward hook after a selected block; saves/loads a `cic_transport_config.json` sidecar; preserves the training RNG so paired runs stay aligned). |
| `cic_transport_trainer.py` | `CICTransportJointTrainer`: full EVEWorld objective (IGR + TIA losses, region weighting, probes) with the transformer as the only changed component; asserts the TIA identity block equals the transport insertion block; logs adapter sentinel stats. |
| `cic_transport_config.py` | Paired full-training config (seed 42, 300 steps): historical EVEWorld config plus the `cic_transport_*` model keys only. |
| `cic_transport_b23_config.py` | Side-by-side twin of the above with the insertion point at `block23` (paper table's minimum-error block) and its own `project_dir`; the campaign preflight does not cover it — launch it through [`../pipeline/t4g_joint_launch.sh`](../pipeline/t4g_joint_launch.sh) instead. |
| `cic_transport_pipeline.py` | Inference loaders (`CICTransportGigaWorld0Pipeline`, EAG variant) that rebuild the transformer from a checkpoint via its sidecar. |
| `test_cic_transport.py` | CPU smoke tests: zero-init identity, soft-splat motion tracking, gradient staging, RNG preservation, checkpoint round trip, activation-checkpoint wrapping. Run with `python -m eveworld.tia_transport.test_cic_transport`. |

### Paired training campaigns

| File | What it does |
|---|---|
| `cic_transport_campaign.py` | Fail-closed `preflight` / `node-preflight` / `audit` CLI for the paired seed-42 runs: protected-path guards, strict 92-sample data check, SHA-256-pinned historical runtime config, normalized runtime-drift diff, per-checkpoint weight/sidecar audit. It deliberately registers only the two bit-exact variants (`control`, `transport`) — alternative insertion points run through the generic joint launcher. |
| `cic_transport_launch.sh` / `cic_transport_kjob_train.sh` | Host-side launcher (`check` / `submit-control` / `submit-transport` / `submit-pair` / `audit`) and the 8-GPU SLURM payload, which runs node-preflight then delegates to [`../../benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh`](../../benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh). |
| `cic_transport_seed6666_s400_config.py` / `cic_transport_seed6666_s400_campaign.py` / `cic_transport_seed6666_s400_launch.sh` / `cic_transport_seed6666_s400_kjob.sh` | Training-seed replication (seed 6666) extended to 400 steps, with its own preflight/audit chain and optional resume. |

### Generation campaigns (EVAL-175, 126-item DreamGenBench eval)

| File | What it does |
|---|---|
| `eval175_transport_worker.py` | Runs the established EVAL-175 worker ([`../evaluation/eval175_multiseed_worker.py`](../evaluation/eval175_multiseed_worker.py)) with the transport pipeline loader patched in. |
| `cic_transport_eval175_launch.sh` / `cic_transport_eval175_kjob.sh` | Seed-004 generation + manifest audit for four checkpoints: control/transport × steps 50/100. |
| `cic_transport_eval250_launch.sh` / `cic_transport_eval250_kjob.sh` | Same for six checkpoints: control/transport × steps 150/200/250. |
| `cic_transport_seed6666_eval250_launch.sh` / `cic_transport_seed6666_eval250_kjob.sh` | Same for the seed-6666 replication checkpoints (steps 150–250). |
| `cic_transport_multiseed70_dispatch.py` | Injects a transport model into the shared multi-seed dispatcher ([`../evaluation/eval175_multiseed_dispatch.py`](../evaluation/eval175_multiseed_dispatch.py)) process-locally via `CIC_TRANSPORT_MODEL_NAME` / `CIC_TRANSPORT_MODEL_DIR`. |
| `cic_transport_multiseed70_launch.sh` / `cic_transport_multiseed70_kjob.sh` | 70-seed generation for `transport_raw_s150` and `transport_raw_s200` in two frozen ranges (seeds 1–35, 36–70). |

### Judge repeats and result aggregation

| File | What it does |
|---|---|
| `cic_transport_gemini_repeats.sh` → `cic_transport_eval175_results.py` | Three Gemini-IF repeats over the step-50/100 checkpoints, then a fail-closed 504-row audit with paired win/loss/tie statistics. |
| `cic_transport_eval250_gemini_repeats.sh` → `cic_transport_eval250_results.py` | Same for steps 150/200/250 (756 rows per repeat). |
| `cic_transport_eval250_qwen_repeats.sh` → `cic_transport_eval250_qwen_results.py` | Three thinking-off Qwen-IF repeats and audit for steps 150–250. |
| `cic_transport_seed6666_eval250_qwen_repeats.sh` → `cic_transport_seed6666_eval250_qwen_results.py` | Qwen-IF repeats for the seed-6666 replication. |
| `cic_transport_multiseed70_qwen_batch.sh` → `cic_transport_multiseed70_qwen_results.py` | Frozen multi-seed Qwen batches for `transport_raw_s150`. |
| `cic_transport_remaining_qwen_repeats.sh` → `cic_transport_remaining_qwen_results.py` | Completes the remaining s150 and all s200 Qwen runs, then compares all 70 seeds. |

### Reference arms

Pretrain and Standard SFT references evaluated under the identical protocol:
`cic_transport_pretrain_eval175_{launch,kjob}.sh` + `cic_transport_pretrain_gemini_repeats.sh`
+ `cic_transport_pretrain_results.py` (pretrain at seed 004);
`cic_transport_pretrain_seed005_025_*` and `cic_transport_pretrain_seed040_*`
(pretrain at seeds 005/025/040); `cic_transport_qwen_reference_{repeats,results}.*`
(Pretrain and Standard SFT Qwen references); `cic_transport_reference_seed049_062_*`
and `cic_transport_reference_seed066_*` (Pretrain/SFT at seeds 049/062/066,
compared against Transport s150 and s200 respectively).

## Usage

`EVEWORLD_ROOT` must point to this repository root; dataset, checkpoint, and
output roots follow the `GAGI_ROOT` convention (see
[`../common/env.sh`](../common/env.sh)). All campaigns write under
`${GAGI_ROOT}/eve_v2_outputs/eve_cic_transport_v1/` and refuse to overwrite
existing outputs.

```bash
# Paired training campaign (preflight is fail-closed)
python -m eveworld.tia_transport.test_cic_transport                 # CPU smoke tests
python -m eveworld.tia_transport.cic_transport_campaign preflight --variant transport
bash eveworld/tia_transport/cic_transport_launch.sh submit-pair     # control + transport
bash eveworld/tia_transport/cic_transport_launch.sh audit           # checkpoint audit

# Generation + judging (after training)
bash eveworld/tia_transport/cic_transport_eval175_launch.sh submit  # four checkpoints, seed 004
bash eveworld/tia_transport/cic_transport_gemini_repeats.sh         # 3 Gemini-IF repeats + audit

# Alternative insertion point (paper table's block 23), outside the strict campaign:
# the generic joint launcher prints its plan, `submit` hands it to SLURM.
# OUT_ROOT/RUN_NAME mirror the twin config's project_dir so they stay in sync.
BASE_CONFIG_MODULE=eveworld.tia_transport.cic_transport_b23_config \
OUT_ROOT=/data/datasets/gagi/eve_v2_outputs/eve_cic_transport_v1/cic_transport_b23_seed42_s300 \
RUN_NAME=cic_transport_b23_seed42_s300 \
  bash eveworld/pipeline/t4g_joint_launch.sh submit
```

> **Cluster caveat.** All `launch_*.sh` / `kjob_*.sh` scripts are SLURM-style
> wrappers for an 8-GPU node and carry cluster-specific absolute paths
> (`${GAGI_ROOT}` layout, conda env `EVEWorld`, the training venv). Adapt
> these paths before running elsewhere. Generation payloads refuse to run on a
> workspace host unless `ALLOW_LOCAL_RUN=1` is set.

## Notes

- Depends on PyTorch, diffusers, and the vendored
  [`../../giga-models/`](../../giga-models/) / [`../../giga-world-0/`](../../giga-world-0/)
  snapshots (`PYTHONPATH` must include the repo root and both snapshots; the
  scripts arrange this themselves).
- Training preflight requires the packed GR1 fine-tuning data, the T4G
  annotation pack (`_packidx2vid.json` and per-video JSONs), IGR augmentation
  assets, and the GigaWorld-0 pretrain transformer/VAE; `cic_transport_campaign.py
  preflight` verifies all of them and enforces the strict 92-sample check. See
  [`../../docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md) for environment setup.
- Generation and manifest building reuse the shared EVAL-175 tooling in
  [`../evaluation/`](../evaluation/); judge repeats call
  `eval_gemini_dreamgen_qwen_protocol.py` (Gemini) or a local OpenAI-compatible
  Qwen endpoint (`QWEN_BASE`, e.g. `http://127.0.0.1:8000/v1`).
- The adapter does not support sequence parallelism; training is validated in
  bf16 full-parameter mode (fp8 training is rejected by the trainer).
