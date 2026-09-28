# Design alternatives

The instruction-grounding designs that were prototyped and measured before the
released method settled on its two components: IGR, the instruction-grounded
reconstruction weighting applied inside the diffusion loss, and TIA, the
temporal-instruction alignment adapter attached at one probed Transformer
block. Each subdirectory records the mechanism, the run settings and the
matched control.

## What was compared

Each route was measured on DreamGenBench against its own matched control: the
closest configuration without the component under test, generated from the
same prompts with the same seeds. Two metrics decide a route. MLR, the Model
Laziness Rate, is the percentage of detector-eligible rollouts whose adjusted
target-instance count deviates persistently from the count in the conditioning
state, restricted to the shared eligible subset `U_63` (63 of the 126 prompts).
Gemini-IF is the instruction-following score
of the generated clip. The protocol, including the occlusion rule that
exempts robot-supported disappearances, is documented in
[`../../docs/evaluation.md`](../../docs/evaluation.md).

| Route | Intervention point | Matched control | Pairs |
|---|---|---|---|
| PhysicsLatent | dynamics proxy, training | Pretrain, 5.8 s | 92 |
| EAG | dynamics proxy, sampling | weight 0 | 16 |
| LAD-LoRA | dynamics proxy, training | LoRA control | 32 |
| Causal Frontier | objective, rollout | control50 | 16 |
| ICH-D | training, sampling, repair | Pretrain, 4 seeds | 368 |

The DreamGenBench evaluation pool is small — 92 matched pairs for
PhysicsLatent, 32 for LAD-LoRA, 368 for ICH-D, and 16 each for EAG and Causal
Frontier, where 13 of the 16 were detector-eligible for EAG.

## Routes

- [`physics_latent/`](physics_latent/) — eight process tokens appended to the
  conditioning sequence, trained with weak physical labels on top of the
  pretrained backbone.
- [`eag/`](eag/) — an energy-guided sampler: a frozen latent-action model
  scores the predicted clean latent and the sampler steps down that energy.
- [`lad_lora/`](lad_lora/) — the same transition energy as a training-time
  regulariser on a rank-64 attention LoRA.
- [`causal_frontier/`](causal_frontier/) — training and generation over
  committed four-latent blocks, with the frontier weighting in
  [`loss.py`](causal_frontier/loss.py).
- [`ich_d/`](ich_d/) — a frozen linear duplicate probe gating a residual that
  deletes repeated content, with the probe fit recorded in
  [`fit.py`](ich_d/fit.py).

## Re-running a route

A route is cheap to re-check on the metrics side: its arm is generated with the
released sampler settings, judged by both instruction-following judges and
scored by the MLR evaluator, exactly like the released arms, so only the
training or sampling change differs. The eligible-set bookkeeping is shared
with the main tables, which is what makes a route's number comparable with a
released row.

```bash
python scripts/evaluate/eval_mlr.py --config configs/eval/mlr/dreamgen.yaml \
    --pred-dir outputs/alternatives/eag/w0.03/generated_only \
    --output outputs/alternatives/eag/w0.03/mlr.json
python scripts/evaluate/eval_instruction_following.py \
    --config configs/eval/instruction_following/gemini_if.yaml \
    --pred-dir outputs/alternatives/eag/w0.03/generated_only \
    --output outputs/alternatives/eag/w0.03/gemini_if.json
```

The same pair of commands scores any other route once its predictions are
written under its own output directory. Generation uses the shared inference
entry point with the route's own conditioning or guidance flags, and the
matched control of each route is the same command with the component switched
off. The floor and ceiling for a
comparison are the two released arms of Table 6: the standard SFT baseline and
the full model.

## Reading order

Each route note states what the route does, how it was run and what its control
was. The notes are self-contained, and the shared protocol is
documented once, in [`../../docs/evaluation.md`](../../docs/evaluation.md) for
the metrics and [`../../docs/reproduction.md`](../../docs/reproduction.md) for
the commands.
