# Data

Split files, per-clip metadata and judge templates for the benchmarks of the paper.
This directory holds text only. The clips, decoded frames and caches stay under the
dataset roots named by the `.env` variables (`DREAMGEN_DATA_ROOT`, `AGIBOT_DATA_ROOT`,
`ROBOTWIN_DATA_ROOT`, `WORLDARENA_DATA_ROOT`), so nothing binary is committed here and
the split files are meaningful on their own: a split row names a clip, it does not carry it.

## Layout

`prepare_dreamgen.py` regenerates the three DreamGenBench files, `prepare_agibot.py` the AgiBot
splits, `build_mlr_metadata.py` the four WorldArena ones and `prepare_robotwin.py` the two
RoboTwin splits:

- `splits/dreamgen/train.txt` — the 92 GR1 fine-tuning clips that post-train the GigaWorld-0 arms.
- `splits/dreamgen/val.txt` — 12 clips held out of the post-training set for checkpoint selection.
- `splits/dreamgen/test.txt` — the 126 DreamGenBench clips behind the evaluation prompts.
- `splits/worldarena/eval.txt` — the 1,000 requests of the frozen WorldArena 1.0 manifest.
- `splits/worldarena/train.txt` — 16 requests that froze the detector thresholds and occlusion rule.
- `splits/robotwin/heldout.txt` — 250 held-out episodes, 50 tasks x episodes 45-49.
- `splits/robotwin/dev.txt` — the episode-45 slice, 50 rows, used to screen checkpoints.
- `metadata/dreamgenbench/target_queries.json` — prompt, target, source and destination per clip.
- `metadata/dreamgenbench/eligible_ids.json` — the 63 clips of the eligible set `U_63`.
- `metadata/worldarena/parsed_targets.json` — prompt, target and mover per request.
- `metadata/worldarena/eligible_ids.json` — the 157 requests eligible for Model Laziness Rate.
- `prompts/qwen_if.txt`, `prompts/gemini_if.txt` — instruction-following judge templates.
- `prompts/vlm_audit.txt` — audit template of the optional vision-language check of flagged records.

Every file here is versioned, and the media it refers to is not. The split files are
manifests of ids: media is copied into place by the dataset preparation scripts and stays
under the roots above. PBench Robot is the exception: nothing is prepared for it here, and
`splits/pbench/eval.txt` plus `metadata/pbench/` only name where the clip list and the 913
question-answer pairs of the upstream dataset are mounted before the horizon sweep is scored.

## Splits

**DreamGenBench.** One line per clip stem, zero-padded to five digits, with a comment header
that states the role and the count. `train.txt` lists the 92 clips behind
[Table 1](../results/paper/tables/table1_dreamgen.md) and
[Table 6](../results/paper/tables/table6_ablation.md); `val.txt` lists 12 clips that were
held out of that post-training set; `test.txt` lists the 126 clips behind the benchmark's
evaluation prompts, of which 63 belong to the shared eligible set `U_63`.

**WorldArena 1.0.** `eval.txt` lists the 1,000 request ids `fixed_scene_task_episode1` to
`fixed_scene_task_episode1000` in manifest order. 157 of them are eligible for Model
Laziness Rate; `train.txt` is not a training set but the 16 requests whose detector
thresholds were frozen before the final evaluation, and the remaining requests are evaluated
with that frozen protocol.

**RoboTwin.** `heldout.txt` lists 250 rows, `task_<task>_episode_<45..49>` for each of the 50
tasks, ordered by task and then by episode, and [Table 5](../results/paper/tables/table5_robotwin.md)
reports all 250. `dev.txt` is the episode-45 row of every task: a 50-row screen that is
disjoint from the 200 confirmation rows of episodes 46-49 and from the 2,250 training
episodes of episodes 0-44.

## Metadata

The metadata carries what the prompts do not: which object an instruction asks for, which
surface it starts on, where it goes, and which agent has to move it. It is the input of the
Model Laziness Rate denominator, so the eligibility of a prompt is a property of the parsed
metadata rather than of the prompt text. The schemas and the eligibility rules are in
[`metadata/README.md`](metadata/README.md).

## Prompts

`prompts/qwen_if.txt` and `prompts/gemini_if.txt` are the two instruction-following judges.
Both receive the instruction, the requirement and the system answer, and both return a score
from 0 to 100 with a per-category breakdown over object presence, spatial relation, attribute,
count, action and no shortcut; the two files differ in wording and in structure because the
two judges are maintained independently. `prompts/vlm_audit.txt` drives the optional review
of flagged records and returns a verdict, a confidence and a reason. The templates are plain
text and can be replaced without touching the runner.

## Regenerating the versioned files

```bash
python scripts/prepare/prepare_dreamgen.py --data-root "$DREAMGEN_DATA_ROOT" --output-dir data
python scripts/prepare/prepare_agibot.py  --data-root "$AGIBOT_DATA_ROOT"  --output-dir data
python scripts/prepare/prepare_robotwin.py --data-root "$ROBOTWIN_DATA_ROOT" --output-dir data
python scripts/prepare/build_mlr_metadata.py --data-root "$WORLDARENA_DATA_ROOT" --output-dir data
```

All four scripts take `--data-root`, `--output-dir` (default `data`) and `--limit` for a smoke
run. `prepare_dreamgen.py` also builds the IGR annotations of the training clips, and
`build_mlr_metadata.py` writes the expected instance counts that the evaluator compares
against. See [`../docs/data_preparation.md`](../docs/data_preparation.md) for the upstream
layouts each root must have.
