# Results

The released numbers of the paper and the per-item scores behind them. Everything stored
here is text: the paper tables rendered as markdown, the pointers to the figure files, and
the schema of the per-item rows. The rollouts, decoded frames and row files themselves are
written into the gitignored `outputs/` scratch tree, so what is committed stays reviewable
and diffable while the large artefacts stay out of the repository.

## Layout

- `paper/tables/table1_dreamgen.md` — DreamGenBench, all models, MLR and instruction following.
- `paper/tables/table3_worldarena.md` — WorldArena 1.0 transfer of the two post-trained arms.
- `paper/tables/table4_agibot.md` — EWMBench after AgiBot post-training, the cross-distribution arm.
- `paper/tables/table5_robotwin.md` — RoboTwin held-out episodes, the cross-backbone arm.
- `paper/tables/table6_ablation.md` — component ablation of IGR and TIA on DreamGenBench.
- `paper/figures/README.md` — which camera-ready figure each released image comes from.
- `item_level/README.md` — the JSONL schema written by every evaluation run.

## Paper tables

Each table is the markdown rendering of the corresponding table of the paper, with the
same rows, the same columns and the same emphasis, so a reviewer can compare a table here
against a table in the paper without re-reading either. The metric definitions are stated
under each table; the protocols behind them, including the occlusion rule of the Model
Laziness Rate, are in [`../docs/evaluation.md`](../docs/evaluation.md).

## Item-level scores

`item_level/<benchmark>/` holds one JSONL file per model and benchmark, one row per evaluated
item. The rows are produced by `scripts/evaluate/*` and are the intermediate product that the
tables aggregate: each row carries the request id, the model, the raw metric values, the
eligibility flag and the protocol parameters, so a table can be recomputed from the rows
without re-running generation. The directories are committed with a `.gitkeep` because the
`*.jsonl` files are ignored globally by the root `.gitignore`; the schema is in
[`item_level/README.md`](item_level/README.md).

## Paper figures

The released images are rendered from the camera-ready vector figures. `paper/figures/README.md`
records which source file each image comes from and the command that re-renders them, so a
figure can be regenerated at another resolution without keeping the intermediate PDFs in the
repository.

## Regenerating the numbers

Every artefact has one wrapper; each wrapper takes the published settings from `configs/` and
writes into `results/` or the scratch tree.

```bash
bash scripts/reproduce/table1_dreamgen.sh    # Table 1, DreamGenBench
bash scripts/reproduce/table3_worldarena.sh  # Table 3, WorldArena 1.0
bash scripts/reproduce/table4_ewmbench.sh    # Table 4, EWMBench (AgiBot)
bash scripts/reproduce/table5_robotwin.sh    # Table 5, RoboTwin (FlowWAM)
bash scripts/reproduce/table6_ablation.sh    # Table 6, component ablation
bash scripts/reproduce/figure4_cfg.sh        # Figure 4, CFG sensitivity
```

The wrappers read the datasets named by `DREAMGEN_DATA_ROOT`, `AGIBOT_DATA_ROOT`,
`WORLDARENA_DATA_ROOT` and `ROBOTWIN_DATA_ROOT`, and the weights under
`EVEWORLD_CHECKPOINT_ROOT`. The inputs, the compute budget and the expected numbers of each
wrapper are documented in [`../docs/reproduction.md`](../docs/reproduction.md); the checkpoints
are listed in [`../checkpoints/README.md`](../checkpoints/README.md).

## Committed versus gitignored

Committed: the five table files, the figure pointer, the item-level schema and the empty
item-level directories. Gitignored: everything under `outputs/`, where the wrappers drop
generated clips, per-clip records and logs, and every `*.jsonl`, `*.npz`, `*.mp4` and image
file anywhere in the tree. A checkout therefore always holds the numbers and never the media
that produced them; re-running a wrapper recreates the media from the datasets and the
checkpoints.
