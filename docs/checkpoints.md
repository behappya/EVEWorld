# Checkpoints

[`scripts/setup/download_models.sh`](../scripts/setup/download_models.sh) is
the single entry point for every weight the repository reads: the two
GigaWorld-0 video backbones, the Wan2.2-TI2V-5B base model, the released
FlowWAM checkpoints, the GroundingDINO locator, the SAM2.1 tracker and the
local Qwen judge. Everything lands under
`${EVEWORLD_CHECKPOINT_ROOT:-checkpoints}`, so a single variable moves the
whole tree onto a larger volume.
[`checkpoints/README.md`](../checkpoints/README.md) describes the same layout
in prose; the table inside the downloader is its executable copy, and a
checkpoint added in one file has to be added in the other.

## What the downloader fetches

| `--only` name | Hub repository | Directory under the root | Files |
|---|---|---|---|
| `giga-world-0-video-pretrain-2b` | `open-gigaai/GigaWorld-0-Video-Pretrain-2b` | `giga-world-0-video-pretrain-2b/` | `transformer/config.json`, `transformer/diffusion_pytorch_model.safetensors` |
| `giga-world-0-video-gr1-2b` | `open-gigaai/GigaWorld-0-Video-GR1-2b` | `giga-world-0-video-gr1-2b/` | the same two files |
| `wan2.2-ti2v-5b` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `wan2.2-ti2v-5b/` | the whole repository |
| `flowwam-stage1` | `YixiangChen/FlowWAM` | `flowwam/` | `flowwam_worldarena_stage1.safetensors` |
| `flowwam-robotwin` | `YixiangChen/FlowWAM` | `flowwam/` | `flowwam_robotwin.safetensors`, `flowwam_robotwin_action_norm_stats.npz` |
| `grounding-dino-swin-t-ogc` | `ShilongLiu/GroundingDINO` | `groundingdino/` | `groundingdino_swint_ogc.pth`, `GroundingDINO_SwinT_OGC.cfg.py` |
| `sam2.1-hiera-large` | `facebook/sam2.1-hiera-large` | `sam2/` | `sam2.1_hiera_large.pt`, `sam2.1_hiera_l.yaml` |
| `sam2.1-hiera-tiny` | `facebook/sam2.1-hiera-tiny` | `sam2/` | `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_t.yaml` |
| `qwen2.5-vl-7b-instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2.5-vl-7b-instruct/` | the whole repository |

The two GigaWorld-0 entries are the pretrained and the GR1 variant of the same
video backbone. Every recipe under `configs/paper/gigaworld/` initialises from
`Video-Pretrain-2B`, so the pretrained entry is the one training needs; the GR1
variant is fetched for completeness. `wan2.2-ti2v-5b` is the Diffusers-format
base model that the FlowWAM Stage-1 weights are loaded on top of, and the two
FlowWAM entries read the same Hub repository with different file lists, so
`--only flowwam-stage1` and `--only flowwam-robotwin` can be run independently
and both write into `flowwam/`. Both SAM2.1 sizes are downloaded;
`SAM2_CHECKPOINT` selects the one the tracker loads, and the MLR defaults use
the tiny pair.

An entry whose file list is empty downloads the whole repository, which is what
happens for the Wan base model and the Qwen judge.

## Running the downloader

```bash
bash scripts/setup/download_models.sh --dry-run
bash scripts/setup/download_models.sh --only sam2.1-hiera-tiny
bash scripts/setup/download_models.sh --checkpoint-root /mnt/checkpoints
EVEWORLD_CHECKPOINT_ROOT=/mnt/checkpoints bash scripts/setup/download_models.sh
```

| Flag | Meaning |
|---|---|
| `--checkpoint-root DIR` | Destination for this invocation; same effect as exporting `EVEWORLD_CHECKPOINT_ROOT`. The script changes into the repository root first, so a relative value resolves against the repository, not against the directory it was called from. |
| `--only NAME` | Fetch only the named entries; repeat the flag to select several. |
| `--dry-run` | Print the `huggingface-cli download` command for every entry and exit without touching the disk. Works without the CLI installed. |

Nothing in the download requires a GPU, so it can run on a login or storage
node. Transfers are resumable and files that already exist are skipped, which
makes a re-run after an interruption cheap. The CLI is `huggingface-cli` by
default and `hf` when the former is absent; set `HF_CLI` to override. When
`hf_transfer` is on `PATH` the script reports that it will use it.

Before the first transfer the script echoes the resolved root, and after the
last one it prints the variables that point at what was written:

```text
GW0_MODEL_DIR=<root>/giga-world-0-video-pretrain-2b
SAM2_CHECKPOINT=<root>/sam2/sam2.1_hiera_large.pt
SAM2_CHECKPOINT (tiny variant)=<root>/sam2/sam2.1_hiera_tiny.pt
GROUNDING_DINO_WEIGHTS=<root>/groundingdino/groundingdino_swint_ogc.pth
GROUNDING_DINO_CONFIG=<root>/groundingdino/GroundingDINO_SwinT_OGC.cfg.py
QWEN_IF_MODEL=<root>/qwen2.5-vl-7b-instruct
```

Copy the ones you need into `.env`, next to `EVEWORLD_CHECKPOINT_ROOT` and
`HF_HOME`; [`installation.md`](installation.md) lists every variable with its
meaning. `HF_HOME` and the checkpoint root both grow with a fresh download, so
point them at the same volume when disk space is tight.

## Layout on disk

```text
checkpoints/
├── giga-world-0-video-pretrain-2b/
│   └── transformer/
│       ├── config.json
│       └── diffusion_pytorch_model.safetensors
├── giga-world-0-video-gr1-2b/
│   └── transformer/
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

The Hub repositories keep their own structure below each subdirectory, so
`giga-world-0-video-pretrain-2b/transformer/` is the directory the GigaWorld-0
loader expects.

## Run checkpoints

Training produces its own checkpoints in the run's output directory: a
`checkpoint-<step>.pt` every `train.save_every` steps (50 for GigaWorld-0, 100
for FlowWAM) together with a `trainer_state.json` that records the step, the
seed and the resolved config. `--resume` restarts from the newest checkpoint in
the directory and `--output-dir` moves the directory; both flags are described
in [`training.md`](training.md), which also covers the run layout. The released
tree above is read-only and no script writes into it.

## Which checkpoint each paper row uses

| Paper row | Initialisation | Checkpoint scored |
|---|---|---|
| Table 1, GigaWorld-0 (pretrained) | none | the released video-pretrain weights as downloaded |
| Table 1 and Table 6, Standard SFT | `Video-Pretrain-2B` | raw step 250 |
| Table 1, Table 3 and Table 6, EVEWorld | `Video-Pretrain-2B` | raw step 250 |
| Table 6, IGR only and TIA only | `Video-Pretrain-2B` | raw step 250 |
| Table 4, AgiBot transfer (EWMBench) | `Video-Pretrain-2B` | raw step 50 |
| Table 5, RoboTwin transfer | released FlowWAM Stage-1 on `wan2.2-ti2v-5b` | the final epoch of the 1,128-step (four-epoch) rank-32 LoRA, screened on the episode-45 development slice |

No stable-EMA weights are evaluated anywhere in the paper. Every GigaWorld-0
row scores the raw training checkpoint at the step the config sets in
`train.max_steps`, so a run is identified by its step and seed alone. The
checkpoint-step study in `configs/ablations/checkpoint/steps.yaml` re-scores
steps 150 and 250 over the seed range 1-70 instead of averaging weights.
Which checkpoint produced a published number is recoverable from the run's
`trainer_state.json` and from the settings each evaluator records beside its
item-level output under `outputs/evaluation/item_level/<benchmark>/` — see
[`reproduction.md`](reproduction.md).

## Verifying a download

```bash
bash scripts/setup/download_models.sh --dry-run
python scripts/setup/check_environment.py
```

`check_environment.py` prints a table with a `checkpoints` row for the resolved
root and one row per checkpoint-path variable, marks each path `ok`, `missing`
or `unset`, and exits non-zero while anything required is absent. Run it after
setting the variables in `.env`; a path that is `missing` usually means the
variable still points at the example value copied from `.env.example`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `huggingface-cli not found on PATH` | the Hub CLI is not installed and `hf` is absent too | `pip install 'huggingface_hub[cli]'`, or set `HF_CLI` to an installed CLI |
| The script downloads nothing | `--only` filter | drop the flag, or name every entry needed |
| The files land in an unexpected tree | `EVEWORLD_CHECKPOINT_ROOT` is already exported in the shell | pass `--checkpoint-root` explicitly; the resolved root is echoed before the first transfer |
| A loader tries to reach the Hub on an offline node | the cache is cold or `HF_HOME` points elsewhere | warm the cache once, then set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` as described in [`installation.md`](installation.md) |
| `check_environment.py` reports `path not found` for a checkpoint | the variable is unset or stale | copy the value the downloader printed into `.env` |
| The MLR detector cannot load its weights | the GroundingDINO and SAM2 variables were never set | set `GROUNDING_DINO_CONFIG`, `GROUNDING_DINO_WEIGHTS` and `SAM2_CHECKPOINT` |
| A transfer stops part-way | interrupted download | re-run the same command; complete files are skipped and partial ones resume |

## Related pages

- [`third_party.md`](third_party.md) — where the backbone source trees come
  from, and what is not redistributed here.
- [`data_preparation.md`](data_preparation.md) — the datasets the checkpoints
  are trained and evaluated on.
- [`inference.md`](inference.md) — how `--checkpoint` selects a run
  checkpoint at generation time.
