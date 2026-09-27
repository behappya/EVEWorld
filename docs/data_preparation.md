# Data preparation

The datasets, benchmarks and backbones are downloaded from their upstream
projects; nothing but text lands in this repository. `data/` holds the split
files that define which clips are used, the per-clip metadata that the training
and evaluation code reads, and the prompt templates used by the judges.
Generated clips, decoded frames and caches stay under the data roots pointed to
by the `.env` variables.

## Datasets

| Asset | Role | Size |
|---|---|---|
| DreamGen GR1 fine-tuning split | Post-training data for the GigaWorld-0 arms. | 92 clips, 480 x 768, 93 frames at 16 FPS. |
| DreamGenBench | Main benchmark: Model Laziness Rate and instruction following. | 126 prompts, shared eligible set `U_63` of 63 clips (coverage 50.00%). |
| AgiBot | Cross-distribution training for the EWMBench row. | 777 clips at 480 x 640. |
| EWMBench | AgiBot evaluation: 21 test episodes, three seeds each. | 21 episodes, seeds 42 / 43 / 44. |
| RoboTwin | Cross-backbone evaluation on the FlowWAM recipe. | 50 tasks x episodes 45-49 = 250 held-out episodes. |
| WorldArena 1.0 | Zero-shot cross-domain evaluation on unseen prompts. | 1,000 prompts, 157 eligible (coverage 19.22%). |
| PBench Robot | Physical-commonsense robustness across output lengths. | 913 QA items. |

The DreamGenBench eligible set is shared with the instruction-following judges,
so MLR and both IF scores describe the same 63 clips. The other 63 prompts are
still generated and judged, but they are not part of the MLR denominator; the
coverage number reports that difference.

The AgiBot split excludes the 21 EWMBench test episodes so that the EWMBench
row is not evaluated on clips the model has seen. The exclusion is written to
the metadata as an audit trail, and the same guard applies to the EWMBench
generation manifests.

The RoboTwin held-out episodes are never part of the 2,250 training episodes.
Checkpoint screening uses a further fixed slice of episode 45 from every task;
the reported numbers use all 250 held-out episodes.

## Where the prepared files go

| Dataset | Split files | Per-clip metadata |
|---|---|---|
| DreamGenBench / GR1 | `data/splits/dreamgen/*.txt` | `data/metadata/dreamgenbench/<vid>.json` |
| AgiBot | `data/splits/agibot/*.txt` | `data/metadata/agibot/<vid>.json` |
| RoboTwin | `data/splits/robotwin/*.txt` | `data/metadata/robotwin/<episode>.json` |
| WorldArena 1.0 | `data/splits/worldarena/*.txt` | `data/metadata/worldarena/<request_id>.json` |
| PBench Robot | `data/splits/pbench/eval.txt` | `data/metadata/pbench/` |

A split file lists one clip id per line. The metadata carries the prompt,
the expected instance count per sampled timestamp and the occlusion
annotations used by the MLR protocol; for training clips it also carries
the IGR annotation described below. Judge and audit templates live in
`data/prompts/` (`qwen_if.txt`, `gemini_if.txt`, `vlm_audit.txt`).

PBench Robot is the one row without a preparation step: the question-answer
pairs ship with the upstream dataset ([`third_party.md`](third_party.md)) and
the horizon sweep is scored straight from the rollouts written under
`--pred-dir`, so the two paths above only name where the upstream clips and
questions are mounted before a sweep is run.

## IGR annotation schema

Each training clip gets one JSON file per video under the annotation directory.
The annotation is produced offline, before training, and is the only input the
IGR corruption builder needs besides the clip itself.

```json
{
  "vid": "570",
  "prompt": "move the red block from the table to the shelf",
  "target_name": "red block",
  "b_name": "shelf",
  "gate_enabled": true,
  "gate_reason": "target tracked over the interaction window",
  "t_arrival": 41,
  "n_lat": 24,
  "H_lat": 30,
  "W_lat": 48,
  "target_cell_0": [12, 9],
  "inventory_cells": [[12, 9], [11, 9]],
  "distractor_cells": [[7, 31]],
  "a_cells": [[9, 24]],
  "n_inventory": 2,
  "n_detected_frames": 91,
  "per_lat_frame": [
    {"target_cell": [12, 9], "b_cells": [[7, 31]], "detected": true},
    {"target_cell": [12, 10], "b_cells": [[7, 31]], "detected": true}
  ]
}
```

The excerpt shows the first two of the `n_lat` entries of `per_lat_frame`;
every entry has the same three keys. Cells are `(row, col)` on the latent grid
and are `null` when the object is not detected in that frame. AgiBot
annotations add a `state_cells` key to each frame entry. Clips without a usable
target track set `gate_enabled` to false and are excluded from IGR corruption;
their training samples fall back to the clean clip with a unit weight map.

Two caches sit next to the annotations. The IGR asset cache
`<assets_dir>/<vid>.npz` stores the patch sampled from the most reliable
observation of the target together with its box and the candidate zones, and
the weight-map cache `<wmap_dir>/<vid>.npy` stores the precomputed float
arrays. Precomputed weight maps are validated on load: all values must be
finite and the unique values must be a subset of the configured value set.

## Latent geometry

Annotations, weight maps and TIA windows are expressed on the latent grid of
the recipe, at a cell size of 16 pixels with a VAE stride of 8:

| Recipe | Grid (T, H, W) | Clip |
|---|---|---|
| DreamGenBench / GR1, GigaWorld-0 | 24 x 30 x 48 | 93 frames at 480 x 768. |
| AgiBot, GigaWorld-0 | 24 x 30 x 40 | 480 x 640. |
| RoboTwin, FlowWAM | 31 x 30 x 40 | 121 frames at 480 x 640. |

The weight map is strictly binary, `3.0` on the disturbed interaction region
and `1.0` elsewhere, and is normalised to unit mean before it is used as a loss
weight.

## Commands

```bash
python scripts/prepare/prepare_dreamgen.py --data-root "$DREAMGEN_DATA_ROOT" --output-dir data
python scripts/prepare/prepare_agibot.py --data-root "$AGIBOT_DATA_ROOT" --output-dir data
python scripts/prepare/prepare_robotwin.py --data-root "$ROBOTWIN_DATA_ROOT" --output-dir data
python scripts/prepare/build_mlr_metadata.py --data-root "$WORLDARENA_DATA_ROOT" --output-dir data
```

All four scripts take the same three options: `--data-root` for the upstream
data tree, `--output-dir` for the directory the splits and metadata are written
to (default `data/`), and `--limit` to cap the number of clips processed, which
is useful for a smoke run. `prepare_dreamgen.py` also builds the IGR
annotations, `prepare_agibot.py` applies the 21-episode leakage guard — the
EWMBench checkout comes from `--ewmbench-root` or `EWMBENCH_DATA_ROOT`, with
the built-in episode list used when neither is available — and
`build_mlr_metadata.py` writes the expected-count metadata the evaluator
compares against.

## Licences

Every dataset, benchmark and pretrained model stays under its original
licence — DreamGen and DreamGenBench (NVIDIA), WorldArena, AgiBot
(AgiBotTech), EWMBench, RoboTwin, FlowWAM, the Wan / CogVideoX / Cosmos
model licences, and GroundingDINO (IDEA). This repository contains only
code, splits and metadata.
See [`third_party.md`](third_party.md) for the corresponding table.
