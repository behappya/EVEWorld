# EVE Held-out Main Experiment

This workflow never submits more than one cluster job at a time. Every training
job requests one node with eight GPUs. Every generation job starts eight
single-GPU shards on that same node; with the default two generation seeds,
each seed receives four shards.

The final test data must not be the old 92 exposed GR1 rows. Prepare:

- a data JSON list whose rows contain `source_file_name`, `image`, and `prompt`;
- a frozen JSONL manifest whose matching rows contain `file_name` and
  `"split": "test"`;
- a source/task description and hash outside the generated output directory.

## 1. Train

The default matrix is three methods by three training seeds. Runs are serial.
For the current 72-row train split, 50 epochs become 450 optimization steps,
with checkpoints at steps 150, 300, and 450.

```bash
cd giga-world-0
PHASE=train \
RUN_TAG=eve_heldout_v1 \
bash eveworld/method/eveworld/method/scripts/heldout_main_serial_chain.sh
```

This writes `checkpoint_candidates.tsv` below
`/data/datasets/gagi/giga_world_0_outputs/eve/heldout_main/eve_heldout_v1/`.

The train phase is restart-safe. A completed logical run is discovered and
skipped. An interrupted run is preserved, and the replacement starts from
step 0 in a sibling `_retryN` directory. The candidate TSV is rebuilt through
a temporary file and replaced only after the full training matrix completes.

## 2. Select checkpoints on validation only

Validation generation is also serial and uses all eight GPUs. Judge B severity
is the primary selection value, completeness is the tie-break, and an earlier
checkpoint wins any remaining tie.

```bash
PHASE=validate \
RUN_TAG=eve_heldout_v1 \
VAL_DATA_PATH=/data/datasets/gagi/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json \
VAL_MANIFEST=eveworld/data_curation/splits/frontier_20260715/val.jsonl \
EXPECTED_VAL_SPLIT=val \
bash eveworld/method/eveworld/method/scripts/heldout_main_serial_chain.sh
```

The frozen result is `selected_checkpoints.tsv`. Do not edit it after test
generation starts.

## 3. Generate the frozen test

```bash
PHASE=generate \
RUN_TAG=eve_heldout_v1 \
SELECTED_CHECKPOINTS=/data/datasets/gagi/giga_world_0_outputs/eve/heldout_main/eve_heldout_v1/selected_checkpoints.tsv \
TEST_DATA_PATH=/path/to/frozen_test_inputs.json \
TEST_MANIFEST=/path/to/frozen_test.jsonl \
EXPECTED_TEST_SPLIT=test \
bash eveworld/method/eveworld/method/scripts/heldout_main_serial_chain.sh
```

Each checkpoint is one eight-GPU job. The controller submits the next job only
after the current job leaves the active queue and its dispatch summary exists.

## 4. Score

```bash
SELECTED_CHECKPOINTS=/data/datasets/gagi/giga_world_0_outputs/eve/heldout_main/eve_heldout_v1/selected_checkpoints.tsv \
GEN_ROOT=/data/datasets/gagi/giga_world_0_outputs/eve/heldout_main/eve_heldout_v1/generation \
bash eveworld/method/eveworld/method/scripts/score_heldout_main.sh
```

Scoring runs Judge B, Judge A, TEA, and paired comparisons against Joint Base.
It also reports `Frontier-only vs Joint-LoRA` and `EVE vs Frontier-only`, which
are the two comparisons needed for the method claim. Cross-training-seed Qwen
statistics bootstrap prompts, report each training seed's direction, and do
not duplicate the Joint Base rows as independent observations.
