# Metadata

The parsed instructions behind the split files: for every prompt, which object an instruction
targets, where it starts and where it has to end up, and which agent is supposed to move it.
Model Laziness Rate counts a request only when the instruction asks for a relocation, and that
property is read from this metadata rather than from the prompt text, so the eligibility of a
prompt is decided here and not by the judge.

## Layout

`scripts/prepare/prepare_dreamgen.py` writes the two DreamGenBench files and
`scripts/prepare/build_mlr_metadata.py` the two WorldArena ones:

- `dreamgenbench/target_queries.json` — parsed `target`, `source` and `destination` of all 126
  DreamGenBench prompts.
- `dreamgenbench/eligible_ids.json` — the 63 clips of the shared eligible set `U_63`, with the
  counts and the coverage they imply.
- `worldarena/parsed_targets.json` — parsed `target` and `mover` of all 1,000 WorldArena requests.
- `worldarena/eligible_ids.json` — the 157 requests eligible for Model Laziness Rate, with counts.

The two `target_queries.json` and `parsed_targets.json` files are objects keyed by request id, so
the eligible list is a subset of the keys of its companion file. Both `eligible_ids.json` files
hold `eligible` (the sorted list of ids in manifest order), `num_eligible`, `num_total` and
`coverage` (a percentage rounded to two decimals); the WorldArena file adds `num_resolved`.

## DreamGenBench

Each entry of `dreamgenbench/target_queries.json` maps a five-digit clip id to an object with five
keys:

| Key | Type | Meaning |
|---|---|---|
| `prompt` | string | The instruction the clip is generated from. |
| `target` | string or null | The object the instruction manipulates. |
| `source` | string or null | The surface the object rests on when the clip starts. |
| `destination` | string or null | The surface or container the object has to reach. |
| `eligible` | bool | `true` when the instruction is a relocation, see below. |

Three entries in full:

```json
"00106": {"prompt": "Lift the milk carton off the tray and put it in the drawer.",
          "target": "milk carton", "source": "tray", "destination": "drawer", "eligible": true}
"00107": {"prompt": "Shake the green apple while it stays above the stove.",
          "target": "green apple", "source": "stove", "destination": null, "eligible": false}
"00113": {"prompt": "Keep the frying pan steady in front of the camera.",
          "target": "frying pan", "source": null, "destination": null, "eligible": false}
```

**Eligibility rule.** `eligible` is `true` exactly when both `source` and `destination` parsed, so
the instruction moves the target from one surface to another. The remaining 63 clips split into 54
instructions with a `source` and no `destination` (a hold or a lift in place, e.g. `00107`) and 9
scene instructions with neither (e.g. `00113`); none of them is eligible, because there is no
arrival surface to disappear from and a lazily generated clip would be indistinguishable from a
correct one. The eligible 63 of 126 clips are the shared `U_63` set.

## WorldArena

Each entry of `worldarena/parsed_targets.json` maps a request id of the form
`fixed_scene_task_episode<N>` to an object with three keys:

| Key | Type | Meaning |
|---|---|---|
| `prompt` | string | The instruction the request is generated from. |
| `target` | string or null | The movable object the instruction manipulates. |
| `mover` | string or null | Which agent has to perform the motion. |

`mover` takes four values: `both_grippers`, `left_gripper`, `right_gripper` or `robot_base`;
`null` means the parser found no agent that has to move. Three entries in full:

```json
"fixed_scene_task_episode1": {"prompt": "Advance the robot base until the fridge is within reach.",
                              "target": null, "mover": "robot_base"}
"fixed_scene_task_episode3": {"prompt": "Hand the blue mug from the left gripper to the right one.",
                              "target": "blue mug", "mover": "both_grippers"}
"fixed_scene_task_episode4": {"prompt": "Wait until the stove is clear.", "target": null, "mover": null}
```

Across the 1,000 requests: 494 name `both_grippers`, 190 `robot_base`, 69 `left_gripper` and 64
`right_gripper`, while 183 carry no mover at all. 197 requests have a target, 143 have neither
field and the remaining 660 name only a mover.

**Eligibility rule.** A request is eligible when `target` and `mover` both parsed: a movable
object exists and some agent has to move it, which is the pair of conditions under which a missing
object can be blamed on the generator instead of on the scene. The rule selects 157 requests, of
which 40 have a target but no mover and 660 have a mover but no target. The 157 ids are the
denominator of the WorldArena Model Laziness Rate.

## Coverage

The two benchmarks disagree on the denominator, and both files state theirs explicitly:

- DreamGenBench: `coverage = 100 * 63 / 126 = 50.0`. Every clip is resolved, so the eligible set
  is half of the benchmark and `num_total` doubles as the denominator.
- WorldArena: `coverage = 100 * 157 / 817 = 19.22`. `num_resolved` is 817, the number of
  requests whose `mover` parsed; the 183 requests without a mover fall outside the denominator
  because the protocol cannot say which agent should have acted.

## Regenerating the versioned files

```bash
python scripts/prepare/prepare_dreamgen.py --data-root "$DREAMGEN_DATA_ROOT" --output-dir data
python scripts/prepare/build_mlr_metadata.py --data-root "$WORLDARENA_DATA_ROOT" --output-dir data
```

`prepare_agibot.py` and `prepare_robotwin.py` write the splits of the other two datasets and need
no metadata. All four scripts take `--data-root`, `--output-dir` (default `data`) and `--limit`
for a smoke run. The upstream layouts the roots must have are described in
[`../../docs/data_preparation.md`](../../docs/data_preparation.md).
