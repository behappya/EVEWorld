# MLR sensitivity: the occlusion threshold and the persistence window

The main protocol fixes two knobs of the missing-locality rule: the
occlusion-overlap threshold `tau_occ`, which decides when an under-count is
excused as robot-supported occlusion, and the persistence requirement `k`,
which decides how long a deviation must last before it counts as an event.
This study recomputes the published table over `tau_occ in {0.10, 0.15, 0.20,
0.25}` and `k in {1, 2, 3}` from the cached per-clip records of one evaluation
run, holding the generated videos, detections, tracks, masks, queries and the
eligible set fixed.

## The rule

Every timestamp carries a per-instance detection count and, when the exemption
is needed, an occlusion ratio per expected instance. An under-count at a
timestamp is exempted when every instance missing there is occluded, and the
occluded test is strict: `occlusion_ratio > tau_occ`, so a ratio exactly at the
threshold is not occluded. This is what
`eveworld.evaluation.mlr.adjust_counts` implements once the ratios are turned
into witnesses. The persistence filter is the package's own `detect_events`,
which flags a clip when the remaining deviations form a run of at least `k`
consecutive timestamps; requiring two timestamps instead of one can only shrink
the event set.

The two directions the table explores act on different mechanisms. Lower
`tau_occ` exempts more under-counts as robot-supported occlusions, while larger
`tau_occ` keeps more of them as MLR candidates; increasing `k` removes
short-lived deviations. The main protocol sets `tau_occ = 0.15`, motivated by
prior amodal-segmentation protocols that use approximately 15--17%
occluded-area criteria to identify nontrivial occlusion.

## How it was run

The sweep is a CPU job over cached per-clip records, so no videos are
regenerated and no detection is re-run: each grid cell is a directory of one
JSON file per clip under a name such as `tau_0.15_k2`, and every file fixes the
counts, the ratios and the eligibility flag produced by the recorded run. Run
it with the package importable, as in the released schedule:

```bash
PYTHONPATH=src python experiments/analysis/mlr_sensitivity/sweep.py \
    --root outputs/mlr_sweep/eveworld --baseline-root outputs/mlr_sweep/sft
```

`--baseline-root` is optional and points at a second sweep with the same
layout; with it, every cell is printed next to the matching baseline cell and
the relative reduction. Without it the script prints the same tables without
the comparison. Clips carrying an `error` field are excluded exactly as in a
normal evaluation run. The first table counts each eligible clip once through
the share of clips with at least one event; the second table reports the mean
per-clip event share, which weights every clip by how much of it was flagged
instead of counting it once.

## Outcome

Events / eligible with the MLR in parentheses, on the shared 63-clip eligible
set:

| `tau_occ` | SFT `k=1` | SFT `k=2` | SFT `k=3` | EVEWorld `k=1` | EVEWorld `k=2` | EVEWorld `k=3` |
|---|---|---|---|---|---|---|
| 0.10 | 11/63 (17.46) | 5/63 (7.94) | 4/63 (6.35) | 2/63 (3.17) | 1/63 (1.59) | 1/63 (1.59) |
| 0.15 | 14/63 (22.22) | 7/63 (11.11) | 5/63 (7.94) | 3/63 (4.76) | 1/63 (1.59) | 1/63 (1.59) |
| 0.20 | 17/63 (26.98) | 9/63 (14.29) | 6/63 (9.52) | 3/63 (4.76) | 1/63 (1.59) | 1/63 (1.59) |
| 0.25 | 20/63 (31.75) | 10/63 (15.87) | 7/63 (11.11) | 4/63 (6.35) | 3/63 (4.76) | 1/63 (1.59) |

Across all twelve configurations the relative reduction against the standard
SFT baseline stays between 70.0% and 88.9% and the comparison never reverses.
Under the main setting, `tau_occ = 0.15` and `k = 2`, MLR falls from 7/63
(11.11%) to 1/63 (1.59%), an 85.7% relative reduction. The 87.5% quoted in the
abstract compares against the pretrained GigaWorld-0 backbone (12.70% to
1.59%) and is a different baseline; the two figures are not interchangeable.

## Reading the record

- [`sweep.py`](sweep.py) is the whole experiment: it reads one directory per
  grid cell, validates every per-clip record, rebuilds the exemption with
  `adjust_counts` and the persistence filter with `detect_events`, aggregates
  through the package's own `aggregate`, and prints the markdown tables plus a
  one-line summary of the grid, the eligible count per cell, the clip errors
  and the reduction range.
- The metric definitions, the frozen detector thresholds and the shared
  eligible set are in [`../../../docs/evaluation.md`](../../../docs/evaluation.md);
  the study is scheduled among the additional analyses in
  [`../../../docs/reproduction.md`](../../../docs/reproduction.md).
- The eligible set alternates with the seed-repeat spread in
  [`../seed_stability/`](../seed_stability/), and the long-horizon study in
  [`../long_horizon/`](../long_horizon/) measures how the same backbones hold
  up over much longer clips.
