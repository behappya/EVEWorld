# Inference

Generation is evaluation-only work: the instruction parser, the grounder and
the corruption builder belong to training, and at sampling time the model
receives the first frame, the instruction and the benchmark-specific control
input. For RoboTwin that control input is the robot-only optical flow; for the
other arms the first frame and the instruction are enough.

## Entry points

| Script | Backbone | Arms |
|---|---|---|
| [`scripts/inference/infer_gigaworld.py`](../scripts/inference/infer_gigaworld.py) | GigaWorld-0. | DreamGenBench, WorldArena 1.0, EWMBench, PBench, the CFG grid. |
| [`scripts/inference/infer_flowwam.py`](../scripts/inference/infer_flowwam.py) | FlowWAM. | The held-out RoboTwin episodes. |

Both read the `inference:` block of their config and accept overrides for the
checkpoint, the step count, the guidance scale and the output directory. The
request list comes from the split recorded in the config's `data.split` field,
which is [`data/splits/dreamgen/test.txt`](../data/splits/dreamgen/test.txt)
with the 126 DreamGenBench request ids for that arm; the 12-clip
`data/splits/dreamgen/val.txt` slice is the development set used for
checkpoint selection and is not part of the reported numbers.

## Paper settings

| Arm | Checkpoint | Denoising steps | CFG | Geometry | Notes |
|---|---|---|---|---|---|
| DreamGenBench | Raw step-250. | 30 | 7.0 | 93 frames at `480 x 768`, 16 FPS. | First frame and instruction. |
| WorldArena 1.0 | Raw step-250. | 30 | 7.0 | 93 frames at `480 x 768`, 16 FPS. | Zero-shot prompts. |
| EWMBench (AgiBot) | Raw step-50. | 30 | 7.0 | 93 frames at `480 x 640`, 16 FPS. | Benchmark seeds 42 / 43 / 44. |
| PBench | Raw step-250. | 30 | 7.0 | 61 to 477 frames at 16 FPS. | Seven horizon tiers. |
| RoboTwin | Final-epoch LoRA. | 40 | 5.0 | 121 frames at `480 x 640`, 24 FPS. | Robot-only flow, seed 42, direct full trajectory. |

The RoboTwin rows are generated with `--flow-cond robot_only`,
`--tia-inject off`, `--full-traj direct` and `--seed 42`; the same values sit
in the config as `inference.flow_cond` and `inference.tia_inject`. The FlowWAM
entry point generates 121 frames at `640 x 480` and 24 FPS, and its defaults
are 25 denoising steps with `--sigma-shift 5.0`, so the RoboTwin rows override
the step count to 40.

CFG itself is studied separately over six training steps and four guidance
weights at a fixed generation seed, with the output written under
`<output-dir>/step_%03d/cfg_<tag>/generated_only/`. The grid is run by
`scripts/reproduce/figure4_cfg.sh` and described in
[`reproduction.md`](reproduction.md).

## Flags

| Flag | Meaning |
|---|---|
| `--config` | YAML file under `configs/`; required. |
| `--checkpoint` | Checkpoint or LoRA adapter to load, overriding the config. |
| `--prompt-file` | Text file listing the prompts or request ids to generate, one per line. |
| `--num-steps` | Override `inference.num_steps`. |
| `--cfg-scale` | Override `inference.cfg_scale`. |
| `--output-dir` | Override `output_dir`; defaults to the value in the config. |
| `--limit` | Cap the number of clips, for smoke runs. |
| `--shard-index`, `--num-shards` | Generate one slice of the request list. |

The FlowWAM entry point adds `--flow-cond`, `--tia-inject` (`on` or `off`),
`--full-traj` (`on`, `off` or `direct`) and `--seed`.

## Output layout

Each request produces one clip:

```
<output-dir>/generated_only/<request_id>.mp4
```

The request id is the benchmark's own identifier, so a finished directory can
be matched against a split file directly. The EWMBench arm instead dumps frames
for the metric suite under
`eval_layout/<model>_dataset/<task>/<episode>/<1|2|3>/video/frame_%05d.jpg`,
with generation seeds 42, 43 and 44 mapped to repeats 1, 2 and 3.

## Sharding and resume

`--num-shards` splits the request list into contiguous ranges and
`--shard-index` selects the range to generate, which lets one run fan out over
several GPUs or several machines. The shards are independent and write into the
same `generated_only/` directory; no coordination is needed beyond that.

A run also resumes by itself: clips whose mp4 file already exists in the output
directory are skipped, so an interrupted run is restarted with the same
command. Pass `--limit` on a first invocation to check the wiring on a handful
of prompts before spending the full budget.

## Worked example

Generate the 126 DreamGenBench request ids with the step-250 checkpoint, one
shard per GPU:

```bash
CKPT="${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}/dreamgen_eveworld/checkpoint-250.pt"

for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python scripts/inference/infer_gigaworld.py \
        --config configs/paper/gigaworld/dreamgen/eveworld.yaml \
        --checkpoint "$CKPT" \
        --output-dir outputs/dreamgen_eveworld \
        --num-shards 8 --shard-index "$i" &
done
wait
```

The 126 ids split into six shards of 16 and two of 15, and the eight processes
write their clips into `outputs/dreamgen_eveworld/generated_only/`. Once the
directory is complete, the judges score it with
[`evaluation.md`](evaluation.md)'s instruction-following commands, and
`scripts/reproduce/table1_dreamgen.sh` runs generation, both judges and the MLR
pass in one go.

To reproduce the CFG grid rather than a single setting, run
`bash scripts/reproduce/figure4_cfg.sh`; it sweeps the six training steps and
four guidance weights and writes one `generated_only/` tree per cell, which is
the layout [`reproduction.md`](reproduction.md) expects for Figure 4.
