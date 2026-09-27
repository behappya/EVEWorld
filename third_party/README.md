# Third-Party Backbones

EVEWorld is a supervision recipe, not a backbone. Both backbones it is applied
to stay upstream, and the repository only holds the glue: hooks, corruption
builders, correspondence adapters and the training/evaluation entry points that
live under [`src/eveworld/integrations/`](../src/eveworld/integrations/).

Neither directory below is tracked here — the root `.gitignore` drops
`third_party/giga-world-0/` and `third_party/FlowWAM/` — so clone them after
checking out this repository:

```bash
bash scripts/setup/clone_gigaworld.sh   # -> third_party/giga-world-0
bash scripts/setup/clone_flowwam.sh     # -> third_party/FlowWAM
```

Both scripts accept a destination argument and a `--ref` override if you want a
different revision than the pinned one.

## GigaWorld-0

- Upstream: <https://github.com/open-gigaai/giga-world-0>
- Role: the main backbone. DreamGenBench, WorldArena 1.0, EWMBench (AgiBot) and
  PBench all fine-tune and evaluate GigaWorld-0 Video-Pretrain-2B.
- Training also imports the two upstream framework packages, which are not on
  PyPI:

  ```bash
  pip install git+https://github.com/open-gigaai/giga-train.git
  pip install git+https://github.com/open-gigaai/giga-datasets.git
  ```

- The backbone is Apache-2.0; see the `LICENSE` inside the checkout.

## FlowWAM

- Upstream: <https://github.com/YixiangChen515/FlowWAM> (Wan2.2-TI2V-5B based)
- Role: the cross-backbone experiment. IGR and TIA are ported to FlowWAM and
  evaluated on held-out RoboTwin episodes, with the reference protocol in
  [`docs/reproduction.md`](../docs/reproduction.md) and the metric stack in
  [`src/eveworld/evaluation/robotwin/`](../src/eveworld/evaluation/robotwin/).
- Training starts from the released **FlowWAM Stage-1** checkpoint plus the
  Wan2.2-TI2V-5B base weights; generation and metrics import from the checkout,
  so point `FLOWWAM_ROOT` at the clone (or leave it unset and keep the default
  `third_party/FlowWAM`).

## Other external tools

| Tool | Used for |
|---|---|
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | target detection for IGR annotation and for MLR counting |
| [SAM2](https://github.com/facebookresearch/sam2) | instance tracking and the robot-occluder masks behind the MLR occlusion check |
| [WorldArena 1.0](https://github.com/WorldArena-Official/WorldArena) | the official WorldArena metric harness |
| [DreamGen](https://github.com/NVIDIA/DreamGen) | the DreamGen GR1 split and its judging protocol |
| [EWMBench](https://github.com/AgibotTech/EWMBench) | the AgiBot metric suite |

Their weights and config paths are read from the environment; copy
[`.env.example`](../.env.example) to `.env` and fill in what you have.
`scripts/setup/check_environment.py` reports which of them it can find.
