# Evaluation

Every reported number comes from generated clips scored by the harness under
[`src/eveworld/evaluation/`](../src/eveworld/evaluation/). The protocols are
frozen: detector thresholds, sampled timestamps, judging settings and guidance
weights were fixed on development data before the final runs, so a comparison
is only meaningful when both sides use the same settings. Each section below
gives what the metric measures, the protocol behind the released runs, the
implementing module, and the command that runs it. Generation is described in
[`inference.md`](inference.md); the end-to-end wrappers live in
`scripts/reproduce/` and are documented in [`reproduction.md`](reproduction.md).

## Model Laziness Rate

**What it measures.** MLR is the percentage of detector-eligible rollouts that
exhibit a persistent deviation of the adjusted target-instance count from its
conditioning-state count. The conditioning state `N_0(x)` is the instance
count observed in the first frame of the clip itself, so the metric needs
neither a prompt-specified quantity nor a reference future. Under-counts
attributable to the robot occluding the object are exempt; over-counts are
always violation evidence.

**Protocol.** Each sampled timestamp is scored by counting movable-object
instances with Grounding-DINO and by tracking masks with SAM2, which also
supplies the robot-occlusion mask. For target instance `i` at timestamp `t`
and robot mask `M_robot(t)`, the occlusion ratio is

```
r_occ(i, t) = |M_target(i, t) ∩ M_robot(t)| / |M_target(i, t)|
```

The instance is exempt at that timestamp when `r_occ > tau_occ = 0.15`. The
comparison is strict: an exact tie is not an exemption. A timestamp whose
instance count falls below the frame-0 inventory is skipped only when *every*
missing instance is exempt. An exemption resets the persistence counter rather
than extending it, so an under-count that occlusion already explains cannot
accumulate into an event. A deviation becomes an event once it appears at two
consecutive sampled timestamps (`k = 2`); the per-clip outcome is binary and
the benchmark score averages over the eligible clips:

```
MLR      = 100 * (1 / |D+|) * sum_{x in D+} e(x)
coverage = 100 * |D+| / |D|
```

`D+` holds the clips whose sampled timelines contain at least one prompted
instance (`N_0(x) > 0`); a clip whose first frame shows no prompted object is
ineligible and leaves both numerator and denominator. Coverage is reported
next to every MLR value for that reason. All models in a comparison are scored
on the same eligible set, released per benchmark under `data/metadata/`:
DreamGenBench, WorldArena and RoboTwin each ship the eligible request ids and
their counts, and `scripts/prepare/build_mlr_metadata.py` regenerates them
from the raw manifests. The number of sampled timestamps is not fixed by the
paper; the released protocol samples 24 timestamps per clip in round fashion.

**Defaults.** These are the frozen values of the WorldArena protocol, exposed
in [`configs/eval/mlr/`](../configs/eval/mlr/) and in
[`configs/eval/worldarena.yaml`](../configs/eval/worldarena.yaml).

| Key | Value | Role |
|---|---|---|
| `deviation_mode` | `symmetric` | Under-counts and over-counts both deviate; only under-counts can be exempt. |
| `occlusion_rule` | `paper_overlap` | The `r_occ` rule above. |
| `sample_mode` | `round` | Evenly spaced timestamps across the clip. |
| `frame_count` | `24` | Sampled timestamps per clip. |
| `persistence` | `2` | Consecutive sampled timestamps required for an event. |
| `tau_occ` | `0.15` | Occlusion-ratio threshold for an exemption. |
| `reliable_area_ratio` | `0.50` | A mask below half its temporal median area carries no visibility evidence and does not vote on an exemption. |
| `reliable_logit`, `presence_logit` | `0.0` | Detector confidence gates for a reliable and a present instance. |
| `object_topk` | `6` | Retained object detections per frame. |
| `object_box_threshold` | `0.35` | Box threshold of the counting stage. |
| `min_center_distance` | `60.0` | Merging distance in pixels: detections closer than this are one instance. |
| `gripper_overlap_threshold` | `0.35` | Gripper overlap above which a detection is treated as the robot rather than the target. |
| `contact_margin_ratio` | `0.035` | Contact margin used by the gripper test. |
| `robot_prompt` | `robot gripper` | Prompt of the robot-side detector. |
| `robot_topk`, `robot_box_threshold` | `3`, `0.15` | Robot detections retained and their box threshold. |
| locator | `box_thr=0.20`, `text_thr=0.15` | Grounding-DINO thresholds of the grounding stage. |
| `sam2_model_config` | `sam2.1_hiera_t.yaml` | SAM2 tracker configuration; the path is resolved inside the SAM2 checkout. |

**Modules.**

| Module | Contents |
|---|---|
| [`mlr/detector.py`](../src/eveworld/evaluation/mlr/detector.py) | `InstanceCounter` (Grounding-DINO wrapper), `count_instances`, `expected_count`. |
| [`mlr/merge.py`](../src/eveworld/evaluation/mlr/merge.py) | `merge_detections`, `merge_track_masks`, `align_timestamps`, `build_metadata` producing `expected_counts`, `timestamps`, `instances`. |
| [`mlr/occlusion.py`](../src/eveworld/evaluation/mlr/occlusion.py) | `occlusion_ratio`, `is_occluded`, `occlusion_table`, `adjust_counts`. |
| [`mlr/persistence.py`](../src/eveworld/evaluation/mlr/persistence.py) | `PersistenceTracker`, `PersistenceResult`, `detect_events`. |
| [`mlr/metric.py`](../src/eveworld/evaluation/mlr/metric.py) | `clip_mlr`, `mean_mlr`, `coverage`, `aggregate`. |
| [`mlr/vlm_audit.py`](../src/eveworld/evaluation/mlr/vlm_audit.py) | Optional vision-language audit of the flagged records, driven by `data/prompts/vlm_audit.txt`. |

**Command.** The summary written to `--output` carries `selected_prompts`,
`records`, `eligible`, `coverage`, `events`, `mlr` and `wilson_95`, plus an
`errors` list for clips whose metadata could not be resolved. On DreamGenBench:

```bash
python scripts/evaluate/eval_mlr.py \
    --config configs/eval/mlr/dreamgen.yaml \
    --pred-dir outputs/dreamgen_eveworld/generated_only \
    --metadata data/metadata/dreamgenbench \
    --output outputs/evaluation/item_level/dreamgen/mlr_eveworld.json
```

`--limit` caps the number of clips for a smoke run. Use
`configs/eval/mlr/worldarena.yaml` and `configs/eval/mlr/robotwin.yaml` for
the other two protocols; the pair comparison between two models on the common
eligible set uses `exact_mcnemar(discordant_a, discordant_b)` on the
discordant clip counts.

**Sensitivity.** The protocol can be recomputed on the same generated videos
for `tau_occ in {0.10, 0.15, 0.20, 0.25}` and `k in {1, 2, 3}`, holding
detections, tracks, masks, queries and the eligible set fixed. The sweep is
regenerated by `experiments/analysis/mlr_sensitivity/`.

## Instruction following

**What it measures.** Qwen-IF and Gemini-IF ask a vision-language judge whether
a generated clip satisfies the instruction that conditioned it, and report the
percentage of clips that pass. The per-clip verdicts are also summarized by
syntax, spatial relation, object attributes and counting, which is what makes
the two judges complementary to MLR: a clip can keep its instance count and
still ignore the instruction, and the reverse.

**Protocol.** Each clip is decoded to 49 frames at JPEG quality 85 and sent
with the instruction and the judge prompt. Both judges run at temperature 0.0
with thinking disabled, four concurrent requests, at most eight in flight,
three retries, a 600-second timeout and a 32,000-token output budget, with
`--resume` so a rerun skips clips already scored. The Qwen judge
(`DEFAULT_MODEL = "qwen-plus"`) talks to an OpenAI-compatible endpoint
resolved from `QWEN_BASE_URL` or the eval config's `base_url`
(`http://127.0.0.1:8000/v1`), with the DashScope compatible endpoint as the
module default and the key from `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`. The
Gemini judge (`DEFAULT_MODEL = "gemini-2.5-flash"`) goes through the GenAI
gateway named by `GEMINI_BASE_URL` or `DIFROST_GENAI_BASE_URL` (with
`DIFROST_API_TOKEN`, `DIFROST_HOST` and `DIFROST_THINKING_LEVEL`), uses the
public Gemini API when no gateway is configured, and takes its key from
`GEMINI_API_KEY` or `GOOGLE_API_KEY`. Both judge prompts are files:
`data/prompts/qwen_if.txt` and `data/prompts/gemini_if.txt`.

**Modules.**

| Module | Contents |
|---|---|
| [`instruction_following/qwen_if.py`](../src/eveworld/evaluation/instruction_following/qwen_if.py) | `QwenIFJudge`, `build_judge_prompt`, `parse_verdict`, `score_records`, `summarize`. |
| [`instruction_following/gemini_if.py`](../src/eveworld/evaluation/instruction_following/gemini_if.py) | `GeminiIFJudge` and the same helpers. |

Both modules import the vendor SDK lazily inside the client class, so the
package imports and the other metrics stay usable without either SDK.

**Command.** DreamGenBench scores both judges over the same generated
directory; the `--limit` flag caps the clips.

```bash
python scripts/evaluate/eval_instruction_following.py \
    --config configs/eval/instruction_following/qwen_if.yaml \
    --pred-dir outputs/dreamgen_eveworld/generated_only \
    --output outputs/evaluation/item_level/dreamgen/qwen_if_eveworld.json

python scripts/evaluate/eval_instruction_following.py \
    --config configs/eval/instruction_following/gemini_if.yaml \
    --pred-dir outputs/dreamgen_eveworld/generated_only \
    --output outputs/evaluation/item_level/dreamgen/gemini_if_eveworld.json
```

The `configs/eval/instruction_following/` configs carry the judging settings
listed above; the endpoint credentials stay in the environment. The CFG grid
of Figure 4 was judged with the same protocol, at a fixed generation seed and
30 inference steps, which is why the guidance sweep is comparable across its
cells. The three generalization splits of the benchmark are judged in the
same pass and summarized separately as the Environment, Object and Behavior
columns of the ablation table.

## RoboTwin reconstruction metrics

**What it measures.** Cross-backbone transfer is scored on the held-out
RoboTwin episodes with four reconstruction metrics and MLR. PSNR and SSIM
measure per-frame fidelity of the generated rollout against the recorded one
and are averaged over the clip; LPIPS measures perceptual distance on the same
frames; Flow-EPE is the end-point error between the optical flows that RAFT
extracts from the generated and the recorded rollout, and it is the metric
that isolates motion fidelity from appearance.

**Protocol.** The evaluation set is 250 rows: 50 tasks times episodes 45-49,
with the episode-45 slice held apart as the development split used for layer
and checkpoint selection. Clips are generated from the FlowWAM backbone with
`--flow-cond robot_only --cfg-scale 5.0 --steps 40 --tia-inject off
--full-traj direct --seed 42`, so the condition is the robot-only flow of the
recorded trajectory and the target is reproduced by the model. PSNR is
computed with a data range of 255 on per-frame values before averaging, and
Flow-EPE averages the per-pixel error over pixels with a valid flow. MLR uses
the same frozen protocol as above, with a 24-timestamp sample per clip.

**Modules.**

| Module | Contents |
|---|---|
| [`robotwin/psnr.py`](../src/eveworld/evaluation/robotwin/psnr.py) | `psnr`, `psnr_clip`, `ssim_clip`, `lpips_clip`. |
| [`robotwin/flow_epe.py`](../src/eveworld/evaluation/robotwin/flow_epe.py) | `compute_flow`, `flow_epe`, `epe_map`. |
| [`robotwin/evaluate.py`](../src/eveworld/evaluation/robotwin/evaluate.py) | `evaluate_clip`, `aggregate`, `main`; the aggregate is written to `aggregate/table5.json`. |

**Command.** `--pred-dir` holds the generated clips, `--target-dir` the
recorded episodes; both are matched on the request id.

```bash
python scripts/evaluate/eval_robotwin.py \
    --config configs/eval/mlr/robotwin.yaml \
    --pred-dir outputs/robotwin_eveworld/generated_only \
    --target-dir data/robotwin/heldout \
    --output outputs/evaluation/item_level/robotwin/table5.json
```

The reproduction wrapper for this table is
`scripts/reproduce/table5_robotwin.sh`.

## EWMBench

**What it measures.** EWMBench scores the AgiBot post-training transfer set
with the official metric suite: Motion, Semantics and Overall are the official
aggregate scores, and DYN, HSD, nDTW and Scene are component metrics for
dynamic degree, hand-object interaction, trajectory similarity and scene
consistency. This is the arm that tests cross-distribution generalization,
since the AgiBot clips come from a different data source than the DreamGen
fine-tuning set.

**Protocol.** The generator produces 777 AgiBot clips at `480 x 640` from the
step-50 checkpoint, three repeats per clip at generation seeds 42, 43 and 44.
The filmstrip is written in the layout the official scorer expects,

```
eval_layout/<model>_dataset/<task>/<episode>/<1|2|3>/video/frame_%05d.jpg
```

where the `<1|2|3>` directory is the repeat index and the seeds map 42 -> 1,
43 -> 2, 44 -> 3. The AgiBot source clips are read from `data/agibot`, or from
`AGIBOT_DATA_ROOT` when that variable points elsewhere. MLR beside the EWMBench
scores uses the same frozen protocol.

**Modules.**

| Module | Contents |
|---|---|
| [`ewmbench/evaluate.py`](../src/eveworld/evaluation/ewmbench/evaluate.py) | `load_scores`, `evaluate`, `aggregate`, `main`, a thin adapter over the external benchmark repository. |

The metric keys reported by the adapter are `motion`, `semantics`, `dyn`,
`hsr`, `ndtw`, `scene` and `overall`; the paper's tables label the fourth
component HSD, and the two names refer to the same hand-object score.

**Command.**

```bash
python scripts/evaluate/eval_ewmbench.py \
    --config configs/eval/ewmbench.yaml \
    --pred-dir eval_layout/eveworld_dataset \
    --output outputs/evaluation/item_level/ewmbench/ewmbench_eveworld.json
```

## WorldArena 1.0

**What it measures.** WorldArena 1.0 evaluates zero-shot generalization to
unseen scenes and embodiments with eight core metrics and an Overall score,
the equal-weight mean EWMScore-local-8. Image Quality, Aesthetic Quality,
Dynamic Degree, Flow Score, Motion Smoothness, Subject Consistency,
Background Consistency and Photometric Consistency are computed locally, and
MLR is reported beside them on the shared eligible set.

**Protocol.** The manifest holds 1,000 prompts; the subset eligible under the
shared detector-eligible filter is released under
`data/metadata/worldarena/`. Clips are generated with the step-250
checkpoint, 30 denoising steps and CFG 7.0 at 93 frames and `480 x 768`, and
scored by the local metric implementation. The overall score is the
unweighted mean of the eight core metrics.

**Modules.**

| Module | Contents |
|---|---|
| [`worldarena/evaluate.py`](../src/eveworld/evaluation/worldarena/evaluate.py) | `load_scores`, `evaluate`, `aggregate`, `main`. |

**Command.**

```bash
python scripts/evaluate/eval_worldarena.py \
    --config configs/eval/worldarena.yaml \
    --pred-dir outputs/worldarena_eveworld/generated_only \
    --output outputs/evaluation/item_level/worldarena/table3.json
```

MLR on this benchmark is computed on the intersection across all evaluated
models, which is why the eligible count differs from the DreamGenBench set.

## PBench

**What it measures.** PBench Robot probes physical commonsense with
question-answer pairs over generated rollouts: Domain is the overall
physical-QA accuracy, and Phys, Space and Time are the physical, spatial and
temporal reasoning accuracies. It complements MLR by measuring broader process
correctness rather than instance count alone.

**Protocol.** Rollouts are generated at seven horizon tiers, 61, 93, 157, 253,
317, 413 and 477 frames at 16 FPS, which correspond to 3.8, 5.8, 9.8, 15.8,
19.8, 25.8 and 29.8 seconds. Each tier is scored with the same 913 questions
over 174 videos, and the summary averages the five durations from 3.8 s to
19.8 s plus a mean over the three longest tiers. Cross-trial robustness
replays the 3.8 s and 5.8 s tiers with three inference seeds under a matched
budget.

**Modules.**

| Module | Contents |
|---|---|
| [`pbench/evaluate.py`](../src/eveworld/evaluation/pbench/evaluate.py) | `load_scores`, `evaluate`, `aggregate`, `main`, a thin adapter over the external benchmark repository. |

**Command.** The `--pred-dir` tree holds one directory per horizon tier;
the horizon sweep has no item-level directory, so the run keeps its summary in
the scratch tree.

```bash
python scripts/evaluate/eval_pbench.py \
    --config configs/eval/pbench.yaml \
    --pred-dir outputs/pbench_eveworld \
    --output outputs/pbench_eveworld/horizon.json
```

## Running a full evaluation

The harnesses share a small command-line surface, and each benchmark adds its
own flags:

| Flag | Meaning |
|---|---|
| `--config` | Evaluation config under `configs/eval/`. |
| `--pred-dir` | Directory of generated clips or the benchmark layout. |
| `--target-dir` | Recorded reference clips, for the reconstruction metrics. |
| `--metadata` | Detector-eligible metadata and prompt targets. |
| `--output` | Destination file for the aggregated summary. |
| `--limit` | Cap the number of scored clips. |

The four item-level benchmarks, DreamGenBench, WorldArena, EWMBench and
RoboTwin, write per-clip records under
`outputs/evaluation/item_level/<benchmark>/` with the aggregate beside them,
so a table cell can be traced back to the clip that produced it. The
end-to-end wrappers in `scripts/reproduce/` chain generation, judging and
metrics per paper artefact; see [`reproduction.md`](reproduction.md) for their
inputs, compute and per-artefact protocols.
