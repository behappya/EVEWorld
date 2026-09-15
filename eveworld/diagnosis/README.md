# Diagnosis: Characterizing Model Laziness

CPU-only diagnostic stage that quantifies **Model Laziness** — the manipulated target duplicating, disappearing, or changing discontinuously mid-rollout — before and alongside the full detector-based MLR stack. It extracts cheap event timelines from generated videos, computes process-faithfulness metrics, tests whether those failures are orthogonal to existing physics/quality scores, supports human annotation for validation, and builds the frame montages used as qualitative evidence.

## Paper mapping

- **Section 3 "Why Embodied World Models Become Lazy"** (Problem Statement / Causes): `scripts/event_extract.py` + `metrics/process_metrics.py` implement the lightweight process-faithfulness probes (premature completion, motion-before-contact, teleportation, completeness, cause-before-effect ordering) used to diagnose laziness patterns in baseline and Standard SFT rollouts.
- **Section 5.1 "Experimental Setup and Metrics"**: `metrics/orthogonality.py` provides the correlation analysis between process faithfulness and existing physics/quality scores (e.g. PBench, Qwen-IF) that motivates reporting a dedicated instance-conservation metric — none of the standard benchmarks measures it directly.
- **Appendix "MLR Evaluation Protocol" (`app:mlr_protocol`, manual-inspection checks)**: `annotation/annotate.py` + `annotation/agreement.py` produce the human labels, inter-annotator Cohen's κ, and the automatic-vs-human calibration behind the manual checks.
- **Figure 1 (teaser) and the appendix qualitative case figures** (surplus-instance removal and qualitative-comparison montages): `scripts/extract_laziness_frames.py` builds the chronological frame montages that exhibit duplication, disappearance, and premature-completion cases.
- The headline, deterministic **MLR** metric (GroundingDINO target counts on the initial frame and 24 sampled timestamps; two-consecutive-timestamp rule, Algorithm 1 in the appendix) is implemented in [`benchmarks/worldarena/mlr_eval.py`](../../benchmarks/worldarena/mlr_eval.py) with [`mlr_dispatch.py`](../../benchmarks/worldarena/mlr_dispatch.py); the DreamGenBench main-results pipeline lives in [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/). This directory is the complementary, GPU-free diagnosis stage; its composite `LAZINESS_RATE` is a heuristic development signal, not the paper's reported MLR.

## Contents

| File | What it does |
|---|---|
| `metrics/process_metrics.py` | Per-video process metrics from `*.events.json`: PCR (premature completion), MBC (motion-before-contact), TELE (single-frame teleportation), COMP (required events present), CBE (cause-before-effect ordering), plus a composite `LAZINESS_RATE`; writes a summary + per-video JSON. |
| `metrics/orthogonality.py` | Scatters process score (1 − lazy) against a physics/quality score column from a CSV/JSON, reports Pearson/Spearman correlations and the fraction of high-physics videos that still cheat; writes `*.json` + `*.png`. |
| `scripts/event_extract.py` | Pure-CPU event-timeline extractor (OpenCV heuristics: foreground motion + cheap connected-component target tracking). Emits normalized event times (`t_action_start`, `t_first_contact`, `t_object_motion`, `t_success`, `t_terminal_stable`), per-frame signals, and `max_jump` per video. |
| `scripts/extract_laziness_frames.py` | Cuts 4–6 representative frames from the generated (right) half of side-by-side evaluation videos and stitches them into a horizontal montage with per-frame timestamps; three modes: `--survey` (uniform sweep), `--id` (reproduce a curated `REGISTRY` case), `--video --frames/--times` (ad-hoc trial). |
| `scripts/run_p0.sh` | Orchestrates the whole stage: event extraction for baseline/SFT video sets → process metrics → orthogonality → prints the human-annotation and go/no-go checklist. |
| `annotation/annotate.py` | Interactive CLI annotation of five laziness dimensions per video (`premature`, `mbc`, `teleport`, `incomplete`, `not_executable`), writing JSONL with `--resume` support. |
| `annotation/agreement.py` | `kappa`: Cohen's κ per dimension between two annotators; `calib`: agreement/Spearman between human `lazy_any` labels and the automatic `lazy` flag (pass criteria: agreement ≥ 0.7 or Spearman ≥ 0.6). |
| `requirements.txt` | CPU-only dependencies: numpy, opencv-python-headless, matplotlib, scipy (CoTracker optional for a stronger tracker). |

## Usage

Install the lightweight dependencies and configure the shared environment first:

```bash
pip install -r eveworld/diagnosis/requirements.txt
# Review eveworld/common/env.sh: GAGI_ROOT, EVE_VIDEO_ROOT, EVE_OUT, GR1_DATA_ROOT
```

End-to-end (pure CPU, runs locally):

```bash
BASELINE_DIR=/path/to/baseline_videos \
SFT_DIR=/path/to/sft_videos \
PHYS_CSV=/path/to/per_video_physics.csv PHYS_COL=pbench_domain \
  bash eveworld/diagnosis/scripts/run_p0.sh
```

Step by step:

```bash
# 1. Event extraction (batch over a video directory)
python3 eveworld/diagnosis/scripts/event_extract.py \
  --video-dir /path/to/videos --meta /path/to/metadata.csv --out-dir $EVE_OUT/p0/events_baseline
#    (--crop 0.5,1.0 keeps only the generated right half of side-by-side mp4s)

# 2. Process metrics
python3 eveworld/diagnosis/metrics/process_metrics.py \
  --events-dir $EVE_OUT/p0/events_baseline --out $EVE_OUT/p0/metrics_baseline.json --tag baseline

# 3. Orthogonality against an existing physics/quality score table
python3 eveworld/diagnosis/metrics/orthogonality.py \
  --process $EVE_OUT/p0/metrics_baseline.json \
  --physics per_video_physics.csv --phys-col pbench_domain \
  --out-prefix $EVE_OUT/p0/ortho_baseline

# 4. Human annotation (two annotators), then agreement + calibration
python3 eveworld/diagnosis/annotation/annotate.py --video-dir /path/to/sampled --out lab_A.jsonl --resume
python3 eveworld/diagnosis/annotation/agreement.py kappa --a lab_A.jsonl --b lab_B.jsonl
python3 eveworld/diagnosis/annotation/agreement.py calib --labels lab_A.jsonl \
  --metrics $EVE_OUT/p0/metrics_baseline.json

# 5. Qualitative montages
python3 eveworld/diagnosis/scripts/extract_laziness_frames.py --survey 16 --video /path/to/side_by_side.mp4
python3 eveworld/diagnosis/scripts/extract_laziness_frames.py --video /path/to/side_by_side.mp4 \
  --frames 0,44,78,100,130
python3 eveworld/diagnosis/scripts/extract_laziness_frames.py --list --id-prefix gigaworld0/
```

## Notes

- **Environment variables.** `run_p0.sh` sources [`eveworld/common/env.sh`](../common/env.sh) and honors `EVE_VIDEO_ROOT`, `EVE_OUT`, `BASELINE_DIR`, `SFT_DIR`, `GR1_META`, `PHYS_CSV`/`PHYS_COL`, and `PYBIN`. Defaults follow the repo's `GAGI_ROOT` storage convention; override them for your site.
- **Heuristic detector.** `event_extract.py` deliberately uses a cheap OpenCV baseline tracker; the ordering-based metrics are designed to be robust to its imprecision. Thresholds such as `TELEPORT_THR` in `process_metrics.py` should be calibrated against the annotation set (`agreement.py calib`) before drawing conclusions, and the tracker can be swapped for CoTracker/SAM2.
- **Data prerequisites.** Generated videos (optionally side-by-side `[condition | generation]` mp4s), plus a prompt metadata CSV with `file_name`/`text` columns for prompt-aware analysis. The curated `REGISTRY` cases in `extract_laziness_frames.py` reference cluster storage paths; re-register cases against your own video copies before using `--id` / `--all`.
- **Cluster jobs.** Training and generation that feed this stage are launched through the `kjob_*` / `launch_*` SLURM-style wrappers elsewhere in the repo (e.g. under [`benchmarks/dreamgenbench/`](../../benchmarks/dreamgenbench/)); their absolute paths must be adapted to your cluster. The diagnosis stage itself is CPU-only and needs no GPU.
- **Downstream.** Laziness evidence gathered here motivates the IGR/TIA training in [`eveworld/method/`](../method/) and complements the detector-based MLR evaluation under [`benchmarks/`](../../benchmarks/).
