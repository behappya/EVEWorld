# Design alternatives

The instruction-grounding designs that were prototyped and measured before the
released method settled on its two components: IGR, the instruction-grounded
reconstruction weighting applied inside the diffusion loss, and TIA, the
temporal-instruction alignment adapter attached at one probed Transformer
block. Every route here is a negative result — none of them improved process
laziness and instruction following at the same time — and each subdirectory
records the mechanism, the run settings and the numbers that ended it.

## What was compared

Each route was measured on DreamGenBench against its own matched control: the
closest configuration without the component under test, generated from the
same prompts with the same seeds. Two metrics decide a route. MLR, the Model
Laziness Rate, is the share of the sampled timestamps at which a target the
instruction asks for is missing, restricted to the shared eligible subset
`U_63` (63 of the 126 prompts, coverage 50.00%). Gemini-IF is the
instruction-following score of the generated clip. The protocol, including the
occlusion rule that exempts robot-supported disappearances, is documented in
[`../../docs/evaluation.md`](../../docs/evaluation.md).

| Route | Intervention point | Matched control | MLR, control → route | Rel. ΔMLR | Rel. ΔIF | Pairs |
|---|---|---|---|---|---|---|
| PhysicsLatent | dynamics proxy, training | Pretrain, 5.8 s | 17.39 → 16.18 | −7.0 | −11.9 | 92 |
| EAG | dynamics proxy, sampling | weight 0 | 15.38 → 38.46 | +150.0 | +10.0 | 16 |
| LAD-LoRA | dynamics proxy, training | LoRA control | 3.85 → 3.85 | 0.0 | −6.1 | 32 |
| Causal Frontier | objective, rollout | control50 | 30.77 → 41.67 | +35.4 | −53.8 | 16 |
| ICH-D | training, sampling, repair | Pretrain, 4 seeds | 11.23 → 13.09 | +16.6 | −30.6 | 368 |

The relative changes carry 95% confidence intervals that are wide next to the
movements themselves: MLR moves by −7.0% [−51.5, +69.1] for PhysicsLatent,
+150.0% [0.0, +500.0] for EAG, 0.0% [0.0, 0.0] for LAD-LoRA, +35.4%
[−42.9, +300.0] for Causal Frontier and +16.6% [−19.4, +70.8] for ICH-D, and
the instruction-following intervals are as wide on the smaller pools. The
DreamGenBench evaluation pool is small — 92 matched pairs for PhysicsLatent, 32
for LAD-LoRA, 368 for ICH-D, and 16 each for EAG and Causal Frontier, where 13
of the 16 were detector-eligible for EAG — so only a large, one-directional
movement is readable, and no route produced one.

## Outcome

PhysicsLatent is the only route that lowered MLR at all, and it paid for the
movement with instruction fidelity. Every other route either raised MLR or
lowered instruction following, and the two largest pools say so with intervals
that exclude a flat result: ICH-D, on 368 matched pairs, moved MLR from 11.23%
to 13.09% while its Gemini-IF fell from 45.83 to 31.79 with an interval of
[−40.1, −20.2], and Causal Frontier lost more than half of its
instruction-following score for a rise in MLR. Full per-route numbers, with the
absolute scores and intervals, are in the route notes below.

The pattern is the reason the released design does not win either metric
indirectly through a proxy model, a detector or a repair pass. It supervises
the two properties the metrics probe directly inside the backbone: IGR restores
the disturbed region in the reconstruction loss, and TIA aligns the target's
hidden states across frames at the probed block. Both train with the model
instead of being bolted onto it, and both improve the released row at once in
[`../../results/paper/tables/table6_ablation.md`](../../results/paper/tables/table6_ablation.md).

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

Each route note states what the route does, how it was run, what its control
was and how it ended. The notes are self-contained, and the shared protocol is
documented once, in [`../../docs/evaluation.md`](../../docs/evaluation.md) for
the metrics and [`../../docs/reproduction.md`](../../docs/reproduction.md) for
the commands.
