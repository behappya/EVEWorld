# Item-level scores

One JSONL file per arm and benchmark, one row per evaluated item: the intermediate product
that the aggregate tables are computed from. A score can therefore be traced back to the clip
that produced it without re-running generation, and a table can be recomputed from the rows
alone.

The harnesses under `scripts/evaluate/` create the files as they run, under the evaluation
output root (`outputs/` by default, which the root `.gitignore` keeps out of the repository):

```
outputs/evaluation/item_level/dreamgen/mlr_eveworld.jsonl     # rows, one per clip
outputs/evaluation/item_level/dreamgen/mlr_eveworld.json      # aggregate written beside them
```

## Layout

- `dreamgen/` — DreamGenBench clips: MLR events, the two instruction-following scores and the
  eligible set the MLR denominator is read from.
- `worldarena/` — the WorldArena prompts: MLR events and the instruction-following scores,
  pooled over the prompts every compared arm can be judged on.
- `ewmbench/` — the AgiBot clips of the EWMBench transfer arm, three generation repeats per
  clip, with the EWMBench component scores in place of the instruction-following columns.
- `robotwin/` — the held-out RoboTwin episodes, with reconstruction metrics against the
  recorded reference clip next to the MLR columns.

Each harness writes the keys it can produce and leaves the rest out or `null`; a row never
carries a value from a stage that did not run.

## Schema

One JSON object per line, with these keys:

- `request_id` (str) — the benchmark item: a DreamGen clip stem, a WorldArena
  `fixed_scene_task_episodeN` id, a RoboTwin `task_<task>_episode_<n>` row, or an AgiBot clip.
- `model` (str) — the arm that generated the clip, e.g. `standard_sft` or `eveworld`.
- `mlr` (float or null) — the clip's Model Laziness Rate in percent, as returned by
  `eveworld.evaluation.mlr.metric.clip_mlr`; null when the MLR pass did not run.
- `eligible` (bool) — whether the item belongs to the MLR denominator: the instruction parsed
  to a target and the first frame contains it.
- `events` (list of bool) — one flag per sampled timestamp, true where a deviation event was
  recorded after the persistence (`k` consecutive timestamps) and occlusion (`tau_occ`)
  rules; the same array the aggregate pools.
- `coverage` (float) — the share of eligible items in the benchmark, in percent, repeated on
  every row so a single row identifies the denominator it belongs to.
- `qwen_if` (float or null) — instruction-following score of the row in percent, judged by
  Qwen2.5-VL-7B.
- `gemini_if` (float or null) — the same quantity judged independently by Gemini.
- `psnr`, `ssim`, `lpips`, `flow_epe` (float or null) — reconstruction metrics of the row
  against the recorded reference clip, written on the benchmarks that have one.
- `protocol` (str) — the frozen protocol revision that produced the row, so rows scored under
  different revisions are never pooled by accident.

## From rows to tables

Group the rows by `model`, keep the rows whose `eligible` is true, and compute
`metric.aggregate` over each group: it returns the mean `mlr` over the eligible clips, the
`coverage`, the eligible and total clip counts, and the number of eligible clips with at least
one event. For a paired comparison across models, restrict every model to the shared eligible
set with `metric.common_eligible` first — that is how the paired comparisons of the paper are
computed, where both arms are scored on the items both can be judged on rather than on their
own eligibilities.

## Regenerating

```bash
python scripts/evaluate/eval_mlr.py \
    --config configs/eval/mlr/dreamgen.yaml \
    --pred-dir outputs/eval_dreamgen/eveworld \
    --output outputs/evaluation/item_level/dreamgen/mlr_eveworld.json
```

`--output` names the aggregate; the per-clip rows are written to the `.jsonl` beside it. The
flags each harness accepts are listed in [`evaluation.md`](evaluation.md); the wrappers in
`scripts/reproduce/` chain the arms of a table and write the rows on the way.
