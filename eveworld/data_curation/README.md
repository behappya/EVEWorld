# Data Curation: Splits, Manifests, and Leakage Control

Curates the GR1 fine-tuning data used for post-training and enforces the train/evaluation hygiene behind the paper's results: prompt-stratified held-out splitting, a frozen development/test-partition manifest, manifest validation with packed-index export into the training configs, and download of the IDM weights used by evaluation probes.

## Paper mapping

- **Section 5.1 "Experimental Setup and Metrics"**: "We post-train on the official DreamGen GR1 fine-tuning split, which contains 92 robot-manipulation videos paired with instructions." `scripts/freeze_frontier_split.py` partitions those rows into the frozen development split (`train`/`val`) plus a small clean held-out candidate set, committed under `splits/frontier_20260715/`.
- **Appendix "Reproducibility and Limitations"**: "Training and evaluation data are disjoint … detector thresholds, checkpoints, and guidance settings are selected on development data before final evaluation." The frozen manifests and `scripts/heldout_manifest_tool.py` make that policy machine-checkable; `splits/frontier_20260715/split_summary.json` records the leakage policy (`train`/`val` are development-only; the clean candidate rows are never used for tuning or checkpoint selection).
- **Section 5.1 metrics / evaluation tooling**: `scripts/download_idm_gr1.sh` fetches the official DreamGen/GR00T inverse-dynamics model (`seonghyeonye/IDM_gr1`) that anchors the IDM-based evaluation probe (inferring actions from generated videos).
- The split indices produced here are consumed by the training configs in [`eveworld/method/configs/`](../method/configs/) (`eve_frontier_mvp.py`, `eve_joint_lora.py`) through `filters.py`, and the resulting checkpoints are evaluated under [`benchmarks/`](../../benchmarks/).

## Contents

| File | What it does |
|---|---|
| `filters.py` | `select_data_indices`: a GigaTrain/GigaDatasets dataset filter that keeps packed rows by their stable `data_index`, so a frozen training subset can be selected without copying the packed videos. Referenced by name from the training configs. |
| `scripts/make_holdout.py` | Generic prompt-stratified held-out splitter: buckets prompts by task template (`pick up X from A to B`), then draws a seeded train/eval split per bucket so evaluation covers diverse objects/actions without leakage. Writes `train.jsonl`, `eval.jsonl`, `split_summary.json`. |
| `scripts/freeze_frontier_split.py` | Freezes the development and clean held-out manifests. The 92-row old manifest was exposed to earlier feasibility runs, so it is development-only; the GR1 metadata rows absent from it (8 rows) are reserved as a clean candidate test set. Emits `train.jsonl` / `val.jsonl` / `test_candidate.jsonl` with `split` and `prior_training_exposure` provenance fields, plus `split_summary.json`; fails closed if any input changes shape. |
| `scripts/heldout_manifest_tool.py` | Manifest utility: `count` (row count after split validation), `indices` (comma-separated unique `packed_index` values for the training configs), `validate-eval` (checks ids, duplicates, split labels, and membership against a packed-data JSON). |
| `scripts/download_idm_gr1.sh` | Idempotent, resumable download of the `seonghyeonye/IDM_gr1` Hugging Face repo into a local probe directory, with a post-download weight-file self-check. `DRY_RUN=1` prints the plan only. |
| `scripts/run_data.sh` | Orchestration skeleton: runs the held-out split step and lists the network-dependent dataset TODOs (official DreamGenBench task lists, unlabeled manipulation videos, cross-domain process benchmarks). |
| `splits/frontier_20260715/` | The committed frozen split summary (seed, counts, clean-candidate ids, leakage policy). The JSONL manifests themselves are regenerated artifacts produced by `freeze_frontier_split.py`. |

## Usage

Configure the shared environment first (see [`eveworld/common/env.sh`](../common/env.sh)): `GAGI_ROOT`, `GR1_DATA_ROOT`, and `EVE_OUT` locate the GR1 download and the curation outputs.

1. **Freeze the development split** (validates that the old 92-row manifest and the GR1 metadata are exactly as expected; refuses to proceed otherwise):

   ```bash
   python3 eveworld/data_curation/scripts/freeze_frontier_split.py \
     --old-manifest "$GR1_DATA_ROOT/raw_data/manifest.jsonl" \
     --metadata "$GR1_DATA_ROOT/raw_hf/metadata.csv" \
     --raw-root "$GR1_DATA_ROOT/raw_hf/gr1" \
     --out-dir eveworld/data_curation/splits/frontier_20260715 \
     --val-count 20 --seed 20260715
   ```

2. **Validate a manifest / export training indices**:

   ```bash
   python3 eveworld/data_curation/scripts/heldout_manifest_tool.py count \
     --manifest eveworld/data_curation/splits/frontier_20260715/train.jsonl \
     --expected-split train

   export FRONTIER_TRAIN_INDICES="$(
     python3 eveworld/data_curation/scripts/heldout_manifest_tool.py indices \
       --manifest eveworld/data_curation/splits/frontier_20260715/train.jsonl \
       --expected-split train)"
   ```

   With `FRONTIER_TRAIN_INDICES` set, `eveworld/method/configs/eve_frontier_mvp.py` wires `eveworld.data_curation.filters.select_data_indices` into the training dataloader so only the frozen rows are seen.

3. **Generic stratified holdout** (any metadata CSV with `file_name`/`text` columns):

   ```bash
   python3 eveworld/data_curation/scripts/make_holdout.py \
     --meta "$GR1_DATA_ROOT/raw_hf/metadata.csv" \
     --out-dir "$EVE_OUT/data/splits" --eval-frac 0.3 --seed 0
   ```

4. **Download the IDM_gr1 evaluation weights** (idempotent; safe to re-run):

   ```bash
   bash eveworld/data_curation/scripts/download_idm_gr1.sh            # default destination
   DEST=/your/path DRY_RUN=1 bash eveworld/data_curation/scripts/download_idm_gr1.sh
   ```

## Notes

- **Dependencies.** The Python tools use only the standard library. The download script needs a Hugging Face CLI (`hf` / `huggingface-cli`, or any Python with `huggingface_hub`); set `HF_ENDPOINT` for a mirror if your network requires one.
- **Data prerequisites.** `freeze_frontier_split.py` expects the GR1 fine-tuning download (`raw_data/manifest.jsonl` with the packed mp4s, `raw_hf/metadata.csv`, `raw_hf/gr1/` videos) — see the data-packing job `kjob_pack_gr1_finetune_data.sh` in [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/). Its defaults point at the repo's `GAGI_ROOT` storage layout; pass explicit flags to adapt.
- **Frozen means frozen.** The committed `splits/frontier_20260715/split_summary.json` fixes seed, counts, and the clean-candidate ids. Do not re-run the freeze with different inputs when reproducing paper numbers; create a new dated split directory for new experiments.
- **Cluster jobs.** Packing, training, and generation that consume these splits run through the repo's `kjob_*` / `launch_*` SLURM-style wrappers (e.g. under [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/)); their absolute paths must be adapted to your cluster before submission.
- **Downstream.** Training configs live in [`eveworld/method/configs/`](../method/configs/); evaluation of the trained checkpoints is orchestrated per benchmark under [`benchmarks/`](../../benchmarks/).
