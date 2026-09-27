# Checkpoints

The default destination of every released weight the repository reads. No weight is ever
committed: `.gitignore` keeps `/checkpoints/*` out of version control and un-ignores this file, so
the tracked tree carries the download instructions and the files themselves stay on disk.

[`scripts/setup/download_models.sh`](../scripts/setup/download_models.sh) is the single entry
point. It honours `EVEWORLD_CHECKPOINT_ROOT`, which defaults to `checkpoints`, and the table below
is the prose copy of its `MODELS` list: a checkpoint added in one file has to be added in the
other. [`../docs/checkpoints.md`](../docs/checkpoints.md) documents the flags, the verification
steps and the failure modes in full.

## Fetching the weights

```bash
bash scripts/setup/download_models.sh --dry-run
bash scripts/setup/download_models.sh
bash scripts/setup/download_models.sh --only sam2.1-hiera-tiny
EVEWORLD_CHECKPOINT_ROOT=/mnt/checkpoints bash scripts/setup/download_models.sh
```

Transfers resume, complete files are skipped, and nothing needs a GPU, so the script runs on a
storage node as well as on a training host. Set the path variables afterwards from the block the
script prints at the end; [`../.env.example`](../.env.example) carries the same variable names
with empty values.

## What is downloaded

Each entry below names the value `--only` expects, the Hub repository it reads and what it is for:

- `giga-world-0-video-pretrain-2b` — `open-gigaai/GigaWorld-0-Video-Pretrain-2b`. The 2B video
  backbone, pretrained; the initialisation of every GigaWorld-0 arm (Tables 1, 3, 4 and 6) and
  the weights behind the zero-shot GigaWorld-0 row of Table 1.
- `giga-world-0-video-gr1-2b` — `open-gigaai/GigaWorld-0-Video-GR1-2b`. The GR1-tuned variant of
  the same 2B backbone; fetched for completeness, no recipe initialises from it.
- `wan2.2-ti2v-5b` — `Wan-AI/Wan2.2-TI2V-5B-Diffusers`. The 5B Diffusers-format base model the
  FlowWAM Stage-1 weights load on top of; needed by Table 5.
- `flowwam-stage1` — `YixiangChen/FlowWAM`. The released Stage-1 weights that every FlowWAM
  recipe initialises from; Table 5.
- `flowwam-robotwin` — `YixiangChen/FlowWAM`. The released RoboTwin fine-tune together with the
  action-normalisation statistics the RoboTwin recipes normalise with; the reference point for a
  local run of Table 5.
- `grounding-dino-swin-t-ogc` — `ShilongLiu/GroundingDINO`. The Swin-T open-vocabulary locator
  that finds the movable object; part of every MLR row (Tables 1, 3, 5 and 6).
- `sam2.1-hiera-large`, `sam2.1-hiera-tiny` — `facebook/sam2.1-hiera-large` and
  `facebook/sam2.1-hiera-tiny`. The Hiera-L and Hiera-T mask trackers; `SAM2_CHECKPOINT` selects
  the one the tracker loads and the MLR defaults use the tiny pair.
- `qwen2.5-vl-7b-instruct` — `Qwen/Qwen2.5-VL-7B-Instruct`. The 7B vision-language model behind
  the local Qwen instruction-following judge (Tables 1, 3 and 6).

The list names the scale rather than a byte count, since the Hub pages state the exact sizes; the
two `flowwam` entries read the same repository with different file lists, so
`--only flowwam-stage1` and `--only flowwam-robotwin` can be fetched independently. The Gemini
judge is reached over an endpoint and has no weight file here.

## Layout on disk

Everything lands under `${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}`, one subdirectory per download,
with the structure of the Hub repository kept below it:

```text
checkpoints/
├── giga-world-0-video-pretrain-2b/transformer/{config.json,diffusion_pytorch_model.safetensors}
├── giga-world-0-video-gr1-2b/transformer/
├── wan2.2-ti2v-5b/
├── flowwam/
│   ├── flowwam_worldarena_stage1.safetensors
│   ├── flowwam_robotwin.safetensors
│   └── flowwam_robotwin_action_norm_stats.npz
├── groundingdino/
│   ├── groundingdino_swint_ogc.pth
│   └── GroundingDINO_SwinT_OGC.cfg.py
├── sam2/
│   ├── sam2.1_hiera_large.pt
│   ├── sam2.1_hiera_l.yaml
│   ├── sam2.1_hiera_tiny.pt
│   └── sam2.1_hiera_t.yaml
└── qwen2.5-vl-7b-instruct/
```

The two FlowWAM entries read the same Hub repository with different file lists, so
`--only flowwam-stage1` and `--only flowwam-robotwin` can be run independently and both write into
`flowwam/`. A run's own training checkpoints do not land here: they are written as
`checkpoint-<step>.pt` next to a `trainer_state.json` in the run's output directory, which lives
under the gitignored `outputs/` tree. Nothing in a training or evaluation run writes into this
directory, so a downloaded tree stays as it was fetched.

## Related pages

- [`../docs/checkpoints.md`](../docs/checkpoints.md) — the downloader in full, the layout, the
  variables each path lands in, verification and troubleshooting.
- [`../docs/installation.md`](../docs/installation.md) — where the path variables are set.
- [`../docs/reproduction.md`](../docs/reproduction.md) — the artefact map that names the checkpoint
  behind each paper row.
