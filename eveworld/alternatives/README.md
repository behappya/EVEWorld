# Alternative Designs (Negative Controls)

Reference implementations of the five alternative routes that were evaluated
before the final IGR + TIA design of EVEWorld. Each route attacks Model
Laziness at a different point — dynamics proxy, training objective, sampling,
rollout factorization, or inference-time repair — and each is a **negative
result**: none improved process laziness and instruction following jointly.
The code is kept for reproducibility of the paper's matched-control
comparisons; these routes are **not** part of the EVEWorld method.

## Paper mapping

- **Experiments → "Ablations and Analysis" → "Comparison with Failed
  Alternatives"**, Table `tab:failure_compare_main`: per-route relative change
  in MLR (process laziness) and Gemini-IF (instruction fidelity) against its
  own matched control under identical prompts and seeds.
- **Appendix "Alternative Designs"** (`app:failed_alternatives`): full route
  descriptions, matched-control MLR pairs (Table `tab:failed_routes`), and
  per-route pseudocode (`alg:physlatent`, `alg:eag`, and the LAD-LoRA /
  Causal Frontier / ICH-D algorithm boxes).

| Route | Intervention | Matched control | MLR control → route | Rel. ΔMLR | Rel. ΔIF |
|---|---|---|---|---|---|
| PhysicsLatent | dynamics proxy + training | Pretrain 5.8s | 17.39% → 16.18% | −7.0% | −11.9% |
| EAG | dynamics proxy + test-time | weight 0 | 15.38% → 38.46% | +150.0% | +10.0% |
| LAD-LoRA | dynamics proxy + training | LoRA control | 3.85% → 3.85% | 0.0% | −6.1% |
| Causal Frontier | training + causal rollout | control50 | 30.77% → 41.67% | +35.4% | −53.8% |
| ICH-D | train + test + repair | Pretrain (4 seeds) | 11.23% → 13.09% | +16.6% | −30.6% |

PhysicsLatent is the only route that lowered MLR at all, at the cost of
instruction fidelity; the comparison motivates the direct restoration (IGR)
and correspondence (TIA) supervision used by the final method.

## Route-to-code map

| Route | Code |
|---|---|
| **PhysicsLatent** | [`physlatent/`](physlatent/) — physics-aware process tokens appended to the conditioning sequence, weak-label auxiliary heads, pseudo-label generation. See [`physlatent/README.md`](physlatent/README.md). |
| **EAG** | [`../method/eag.py`](../method/eag.py) (transition energy + guidance step), [`../method/pipeline_eag.py`](../method/pipeline_eag.py) (guided sampler), [`../method/scripts/generate_eag.py`](../method/scripts/generate_eag.py) (paired generation) |
| **LAD-LoRA** | [`../method/eve_lad_lora_trainer.py`](../method/eve_lad_lora_trainer.py) (trainer), [`../method/configs/eve_lad_lora.py`](../method/configs/eve_lad_lora.py) (rank-64 LoRA config) |
| **Causal Frontier** | [`../method/eve_frontier_loss.py`](../method/eve_frontier_loss.py) + [`../method/eve_frontier_trainer.py`](../method/eve_frontier_trainer.py) + [`../method/pipeline_frontier.py`](../method/pipeline_frontier.py) (committed-frontier MVP); [`../method/eve_trainer.py`](../method/eve_trainer.py) + [`../method/losses/causal_process_loss.py`](../method/losses/causal_process_loss.py) + [`../method/losses/lazy_negatives.py`](../method/losses/lazy_negatives.py) (counterfactual-process variant); configs [`eve_frontier_mvp.py`](../method/configs/eve_frontier_mvp.py) / [`eve_causal_fullft.py`](../method/configs/eve_causal_fullft.py) |
| **ICH-D** | `t4g_ich_*` family in [`../pipeline/`](../pipeline/): learned-detector arms ([`t4g_ich_trainer.py`](../pipeline/t4g_ich_trainer.py)), offline frozen-probe fitting ([`t4g_ich_d_fit.py`](../pipeline/t4g_ich_d_fit.py)), D-arm trainer ([`t4g_ich_d_trainer.py`](../pipeline/t4g_ich_d_trainer.py)), inference wrapper ([`t4g_ichD_generate.py`](../pipeline/t4g_ichD_generate.py)) |

## Route notes and entry points

**EAG — energy-guided sampling.** A frozen latent-action model
([`../method/lam/latent_action_model.py`](../method/lam/latent_action_model.py),
self-supervised pretraining via `train_lam.py` on caches from
`encode_latents.py`) scores the predicted clean latent with a soft top-3
transition energy; after each CFG update the sampler takes one normalized,
sigma-scheduled step down the energy. `--eag-weight 0` reproduces the matched
baseline at the same seed.

```bash
# paired baseline (w=0) + EAG (w=0.03) submission at a shared seed
bash eveworld/method/scripts/launch_eag_generate_kjob.sh
LIMIT=8 DRY_RUN=1 bash eveworld/method/scripts/launch_eag_generate_kjob.sh  # smoke / print only
```

**LAD-LoRA — training-time transition-energy regularizer.**
`EveLadLoraTrainer` (runner `eveworld.EveLadLoraTrainer`) keeps the EDM denoising
loss and adds the same soft top-3 transition energy of the denoised output as
a sigma-gated regularizer, re-weighted every step so the energy gradient
stays at a fixed fraction (target 0.1, capped at 1.0) of the denoising
gradient. The backbone stays frozen; only the rank-64 attention LoRA is
trained (DeepSpeed required). `W_LAD=0` is the pure-LoRA matched control.

```bash
W_LAD=0.1 SEED=6666 bash eveworld/method/scripts/launch_lad_lora_train_kjob.sh
# serial W_LAD ablation chain: eveworld/method/scripts/lad_lora_serial_chain.sh
```

**Causal Frontier — committed-prefix blockwise generation.** The clip is
factorized into blocks of four latent frames; the model only ever sees the
clean committed prefix (marked by a near-zero noise level and a mask channel)
plus the noisy active block, and the EDM loss applies to the active block
only. `FrontierGigaWorld0Pipeline` generates block by block and commits each
block irreversibly. `eve_frontier_loss.py` also implements the
reference-centered anti-skip ranking loss (immediate-next vs. later/terminal
candidate, centered on the frozen base model); CPU mechanics tests live in
[`../method/tests/test_eve_frontier.py`](../method/tests/test_eve_frontier.py).
The earlier counterfactual-process variant (`eve_trainer.py`, runner
`eveworld.EveCausalTrainer`, config `eve_causal_fullft.py`) instead keeps
full-sequence training and margin-ranks the real process against
laziness-corrupted negatives (teleport / excision / freeze-jump). Entry
points: `launch_frontier_mvp_kjob.sh` (8-GPU train-only overfit probe),
`launch_frontier_generate_kjob.sh` (blockwise generation), and
`launch_frontier_eval_kjob.sh` (next-vs-future preference probe), all under
`eveworld/method/scripts/`.

**ICH-D — inference-time deletion residual.** A frozen linear probe (artifact
of `t4g_ich_d_fit.py`; 16-dim novelty + correspondence features collected
from blocks 10/12/16/22) estimates per-cell duplicate probability; a gated,
zero-initialized residual then removes the flagged content from the block-22
representation. The learned-detector arms (`t4g_ich_trainer.py` with configs
`t4g_ich_1L/3L_config.py` and the `*_pw_launch.sh` re-weighted reruns) and
the layer-fusion ablation (`t4g_ich_fusion_ablation.py`) document the route
to the frozen design. `t4g_ichD_generate.py` restores `ich_d.*` weights and
hooks into a standard generation pass; `t4g_ichD_gen_kjob.sh` runs the
92-prompt × 4-seed evaluation pool. ICH-D builds on the self-case restoration
pipeline in [`../pipeline/`](../pipeline/) (`t4g_selfcase_trainer.py`,
`t4g_aug_*`, `selfcase_g1*.py`); the eraser was ultimately dropped from the
final model (see `t4g_final_trainer.py`).

## Usage

- **Environment.** Set `EVEWORLD_ROOT` to the repository root; data and
  output paths follow the `GAGI_ROOT` convention in
  [`../common/env.sh`](../common/env.sh). Entry points expect `PYTHONPATH` to
  include the repo root, [`../../giga-world-0/`](../../giga-world-0/) (pinned
  backbone) and [`../../giga-models/`](../../giga-models/) (training
  framework), plus the training venv created by
  `giga-world-0/scripts/setup_gigaworld_train_venv.sh`.
- **Cluster jobs.** All `kjob_*.sh` / `launch_*_kjob.sh` scripts are
  SLURM-style cluster wrappers left in their development state: they carry
  absolute dataset/checkpoint paths and a `kjob` submission command that must
  be adapted to your site. Training routes reuse the DreamGenBench GR1
  training payload in [`../../benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/),
  which also hosts the MLR / instruction-following evaluation behind the
  matched-control numbers.

## Notes

- **Status: negative controls.** For the method that worked (IGR + TIA), see
  the core pipeline in [`../pipeline/`](../pipeline/) and
  [`../tia_transport/`](../tia_transport/).
- **Module naming.** The EAG / LAD-LoRA / Frontier trainers and configs
  subclass the PhysLatent trainer and import it as
  `eveworld.alternatives.physlatent` (see [`physlatent/`](physlatent/)).
  Runner strings of the form `eveworld.X` resolve to the top-level
  `eveworld` package (see [`../__init__.py`](../__init__.py)).
- **Dependencies.** GroundingDINO (detection and MLR), a pretrained
  latent-action checkpoint for the EAG / LAD-LoRA routes, the DreamGen GR1
  fine-tuning split, and the pinned GigaWorld-0 / GigaModels snapshots.
  Environment details: [`../../docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md).
