# Table 6 — Component ablation on DreamGenBench

All four variants share the initialization, the 92-clip training set and the 250-step
optimization budget, so the differences isolate IGR and TIA. Gemini-IF is broken down over
the three generalization splits of the benchmark: unseen environments, unseen objects and
unseen behaviors.

| Variant | IGR | TIA | Env ↑ | Object ↑ | Behavior ↑ | Overall ↑ | MLR (%) ↓ |
|---|---|---|---|---|---|---|---|
| Standard SFT | | | 51.72 | 42.00 | 67.02 | 53.57 | 11.11 |
| IGR only | ✓ | | 49.43 | 36.67 | **69.50** | 51.85 | 4.76 |
| TIA only | | ✓ | 55.17 | 42.67 | 68.79 | 55.29 | 7.94 |
| **EVEWorld** | ✓ | ✓ | **72.41** | **50.33** | 64.89 | **60.85** | **1.59** |

## Metrics

- **Env**, **Object**, **Behavior**, **Overall** — Gemini-IF instruction-following accuracy in
  percent, higher is better, on the environment, object and behavior splits and on the union
  of the three; the judge and its rubric are described in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md).
- **MLR (%)** — Model Laziness Rate, lower is better, on the shared eligible set `U_63` and
  under the same rule as [Table 1](table1_dreamgen.md).

IGR alone carries the instance-consistency gain (MLR 11.11% to 4.76%) while costing a little
instruction following; TIA alone carries the instruction-following gain (Overall 53.57% to
55.29%) with a smaller MLR gain; together they give the best combined result, 1.59% MLR and
60.85% Overall, with the Environment and Object splits showing the largest movement.

## Reproduction

`bash scripts/reproduce/table6_ablation.sh` trains and evaluates the four variants and writes
these rows; the shared training settings are in
[`../../../docs/reproduction.md`](../../../docs/reproduction.md) and the per-variant configurations
are under `configs/ablations/`.
