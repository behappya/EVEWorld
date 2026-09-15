# eveworld/method/ — Alternative-Design Training Stack

Training and sampling stack for the **alternative anti-laziness designs** evaluated as negative controls in Appendix D of the paper ("Alternative Designs"): Causal Frontier, LAD-LoRA, EAG sampling guidance, and the earlier CPC-style contrastive prototype, together with the parameter-matched LoRA controls and the held-out comparison harness used to score them. The shipped EVEWorld method (IGR + TIA) lives in [`../pipeline/`](../pipeline/) and [`../tia_transport/`](../tia_transport/); this directory keeps the rejected routes reproducible.

## Paper mapping

| Code | Paper |
|---|---|
| `eve_frontier_trainer.py`, `eve_frontier_loss.py`, `pipeline_frontier.py`, `configs/eve_frontier_mvp.py` | Appendix D, "Causal Frontier": prefix + active-block factorization, active-block-only denoising loss, irreversible block commits at inference |
| `eve_lad_lora_trainer.py`, `configs/eve_lad_lora.py` | Appendix D, "LAD-LoRA": frozen transition-energy regularizer on the model's own x0 prediction, distilled into a rank-64 LoRA |
| `eag.py`, `pipeline_eag.py`, `scripts/generate_eag.py` | Appendix D, "EAG": sampling-time guidance along the negative transition-energy gradient |
| `lam/`, `scripts/train_lam.py`, `scripts/encode_latents.py` | The frozen latent-action transition-energy model E shared by EAG and LAD-LoRA (Appendix D) |
| `eve_trainer.py`, `losses/`, `configs/eve_causal_fullft.py` | Development-stage CPC-style contrastive prototype (margin ranking against synthetic lazy negatives); part of the exploration that motivated the negative-control study |
| `configs/eve_joint_lora.py`, `scripts/heldout_main_serial_chain.sh` | Parameter-matched joint-LoRA control and the held-out train/validate/generate/score protocol behind the Appendix D comparisons |

## Contents

| Path | What it does |
|---|---|
| `eve_trainer.py` | `EveCausalTrainer`: full-parameter trainer adding margin-ranked contrastive terms (`cpc_*`) between real sequences and synthetic lazy negatives in latent space |
| `eve_frontier_trainer.py` | `EveFrontierTrainer`: LoRA trainer for the frontier factorization; losses `frontier_edm` + optional reference-centered anti-skip hinge (`frontier_skip`, requires `train_mode="lora"`) |
| `eve_frontier_loss.py` | Pure-tensor frontier mechanics shared by trainer, pipeline, and tests: `FrontierConfig`/`FrontierSlice`, batch preparation, `CommittedHistory`, boundary guard, skip losses |
| `eve_lad_lora_trainer.py` | `EveLadLoraTrainer`: EDM main loss + sigma-gated soft-top-k transition-energy regularizer (`lad_reg`) with per-step EDM/LAD gradient balancing; fails fast unless DeepSpeed gradient clipping is active |
| `eag.py` | `EAGGuidance`: wraps a frozen latent action model as a transition-energy function (soft top-k aggregation) with a sigma-scheduled, norm-bounded guidance step |
| `pipeline_eag.py` | `EAGGigaWorld0Pipeline`: inference pipeline inserting EAG guidance into the parent sampling loop; `eag_weight=0` is exactly the baseline sampler |
| `pipeline_frontier.py` | `FrontierGigaWorld0Pipeline`: sequential committed-frontier I2V generation (one active latent block at a time, optional boundary commit guard) |
| `heldout_io.py` | Shared manifest-row selection and image preparation for held-out generation |
| `configs/` | GigaTrain config modules: `eve_causal_fullft.py` (+ `eve_ablation_no_cpc.py`, `eve_ablation_random_neg.py`), `eve_frontier_mvp.py` (env-tunable `FRONTIER_*`), `eve_lad_lora.py` (env `W_LAD`/`LAD_*`/`LAM_CKPT`), `eve_joint_lora.py` (matched control; requires `HELDOUT_TRAIN_INDICES`), `deepspeed_zero2_clip.json` (ZeRO-2 + `gradient_clipping: 1.0`) |
| `losses/` | `lazy_negatives.py` (teleport / excision / shuffle / freeze-jump constructors + random-permutation control) and `causal_process_loss.py` (composable CPC / dynamics-consistency / progress losses) |
| `lam/` | `latent_action_model.py`: Genie-style latent action model (inverse dynamics → codebook-quantized actions → forward dynamics) used as the frozen energy model |
| `scripts/` | Entry points and cluster wrappers — see below |
| `tests/` | `test_eve_frontier.py`: CPU unit tests for the frontier mechanics |

## Usage

All training is config-driven through the GigaTrain launcher: a config module sets `runners = ["eve.<TrainerName>"]` (resolved via the lazy attributes in [`../__init__.py`](../__init__.py)) and a wrapper submits the job through the shared GR1 payload [`benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh`](../../benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh). Set `EVEWORLD_ROOT` to the repository root first.

```bash
export EVEWORLD_ROOT=/path/to/EVEWorld

# LAD-LoRA (8 GPUs, one node); W_LAD=0.0 is the plain-LoRA control
bash eveworld/method/scripts/launch_lad_lora_train_kjob.sh W_LAD=0.1 SEED=6666

# Causal Frontier MVP overfit probe (8 fixed train rows)
bash eveworld/method/scripts/launch_frontier_mvp_kjob.sh FRONTIER_BLOCK_SIZE=4

# CPC-style full fine-tune (+ ablation configs under configs/)
bash eveworld/method/scripts/launch_eve_train_kjob.sh MAX_STEPS=2000
```

Launchers accept trailing `KEY=VALUE` arguments and export them as environment variables. `kjob_*.sh` / `launch_*.sh` are SLURM-style wrappers (SBATCH headers, pod polling) whose absolute paths — `/data/datasets/gagi/...`, venv locations, scheduler integration — **must be adapted to your cluster**.

**LAD pretraining** (prerequisite for EAG and LAD-LoRA; one GPU suffices):

```bash
python eveworld/method/scripts/encode_latents.py --video-dir <clean_videos> --out latents.pt
python eveworld/method/scripts/train_lam.py --latents latents.pt --out lam.pt --steps 8000
```

`train_lam.py` ends with a go/no-go check that real transitions score lower than synthetic lazy ones; `scripts/diag_lam_aggregation.py` compares error aggregations (mean/max/p90/top-k).

**Guided / frontier generation:**

```bash
# EAG (weight 0 = baseline sampler)
python eveworld/method/scripts/generate_eag.py --data-path gr1_dreamgen_it2v.json \
    --save-dir out/eag --transformer <ckpt> --vae <vae> --text-encoder <te> \
    --lam lam.pt --eag-weight 0.03

# Committed-frontier rollout from a frontier checkpoint
python eveworld/method/scripts/generate_frontier.py --data-path <json> \
    --split-manifest <manifest.jsonl> --expected-split test --save-dir out/frontier \
    --transformer <ckpt> --text-encoder <te> --vae <vae> --lora <lora> --seed 0
```

**Held-out comparison.** The serial controller `scripts/heldout_main_serial_chain.sh` runs the leakage-safe protocol (train → validation-based checkpoint selection → frozen-test generation → scoring) for the `joint_lora` / `frontier_only` / `eve` arms; see [`scripts/HELDOUT_MAIN.md`](scripts/HELDOUT_MAIN.md) for the exact four-phase commands.

**Tests** (CPU-only):

```bash
python -m unittest eveworld.method.tests.test_eve_frontier
```

## Notes

- All trainers inherit `PhysicsLatentGigaWorld0Trainer` from [`../alternatives/physlatent/`](../alternatives/physlatent/) and run against the vendored [`giga-world-0/`](../../giga-world-0/) backbone and [`giga-models/`](../../giga-models/) framework; the environment is documented in `docs/ENVIRONMENT.md`.
- LAD-LoRA requires DeepSpeed (the trainer asserts the configured gradient clipping matches `max_grad_norm`) and a frozen LAD checkpoint (`LAM_CKPT`).
- Data prerequisites: packed GR1 fine-tuning data, GigaWorld-0 pretrained weights, and — for held-out runs — the frozen split manifests under [`../data_curation/splits/`](../data_curation/splits/).
- Generated videos are scored with the tooling in [`../evaluation/`](../evaluation/) and the benchmark pipelines under [`benchmarks/`](../../benchmarks/).
