# Third-party components

EVEWorld is a supervision recipe applied to upstream video world models.
Nothing upstream is vendored or redistributed here: this repository carries the
supervision, training and evaluation code, and every external component is
obtained from its own project under its own licence. The two backbones are
cloned by the scripts under `scripts/setup/`, the released weights are fetched
by [`download_models.sh`](../scripts/setup/download_models.sh), and the
datasets stay outside the repository as described in
[`data_preparation.md`](data_preparation.md).

## Backbones

| Component | Role | How it is obtained | Licence |
|---|---|---|---|
| [giga-world-0](https://github.com/open-gigaai/giga-world-0) | Main backbone, including the GigaModels package: DreamGenBench, WorldArena 1.0, EWMBench and PBench fine-tune and evaluate its Video-Pretrain-2B checkpoint. | `bash scripts/setup/clone_gigaworld.sh` clones it into `third_party/giga-world-0`. | Apache-2.0 |
| [giga-train](https://github.com/open-gigaai/giga-train) | Training framework imported by the GigaWorld-0 recipe; not on PyPI. | `pip install git+https://github.com/open-gigaai/giga-train.git` | Apache-2.0 |
| [giga-datasets](https://github.com/open-gigaai/giga-datasets) | Data-processing framework imported by the GigaWorld-0 recipe; not on PyPI. | `pip install git+https://github.com/open-gigaai/giga-datasets.git` | Apache-2.0 |
| [FlowWAM](https://github.com/YixiangChen515/FlowWAM) | Cross-backbone recipe on Wan2.2-TI2V-5B; IGR and TIA are ported to it and evaluated on RoboTwin episodes. | `bash scripts/setup/clone_flowwam.sh` clones it into `third_party/FlowWAM`. | Apache-2.0 |

The root `.gitignore` drops both clone directories, so an existing checkout can
be placed there instead; the clone scripts also accept a destination argument
and a `--ref` revision. [`../third_party/README.md`](../third_party/README.md)
describes what each backbone contributes to the recipe.

## Released weights

`bash scripts/setup/download_models.sh` fetches every row below into
`$EVEWORLD_CHECKPOINT_ROOT` (default `checkpoints/`) and prints the environment
variables that point the code at the files; [`checkpoints.md`](checkpoints.md)
documents the flags, the landing paths and the subset each paper row uses.

| Weights | Used by | Source | Licence |
|---|---|---|---|
| GigaWorld-0 Video-Pretrain-2b | The GigaWorld-0 arms: DreamGenBench, WorldArena 1.0, EWMBench and PBench. | [open-gigaai/GigaWorld-0-Video-Pretrain-2b](https://huggingface.co/open-gigaai/GigaWorld-0-Video-Pretrain-2b) | Apache-2.0 |
| GigaWorld-0 Video-GR1-2b | Start point of the GR1 fine-tuning arm. | [open-gigaai/GigaWorld-0-Video-GR1-2b](https://huggingface.co/open-gigaai/GigaWorld-0-Video-GR1-2b) | Apache-2.0 |
| Wan2.2-TI2V-5B Diffusers weights | Base weights for FlowWAM training and RoboTwin generation. | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | Apache-2.0 |
| FlowWAM Stage-1 and RoboTwin safetensors | FlowWAM training start point, the released RoboTwin weights, and the action-normalisation statistics that ship with them. | [YixiangChen/FlowWAM](https://huggingface.co/YixiangChen/FlowWAM) | Apache-2.0 |
| GroundingDINO Swin-T OGC weights and config | The MLR detector. | [ShilongLiu/GroundingDINO](https://huggingface.co/ShilongLiu/GroundingDINO) | Apache-2.0 |
| SAM2.1 Hiera large and tiny checkpoints | The MLR tracker and the robot-occluder masks. | [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large), [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) | Apache-2.0 |
| Qwen2.5-VL-7B-Instruct | The local Qwen-IF judge. | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Apache-2.0 |

## Metric stacks and benchmark harnesses

| Component | Role | Obtained from | Licence |
|---|---|---|---|
| GroundingDINO | Movable-object detection for IGR annotation and for MLR counting; the harness reads the weights and the config through `GROUNDING_DINO_WEIGHTS` and `GROUNDING_DINO_CONFIG`. | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| SAM2 | Instance tracking and the robot-occluder masks behind the MLR occlusion rule; `SAM2_CHECKPOINT` names the checkpoint and the tracker config is resolved inside the SAM2 checkout. | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Apache-2.0 |
| RAFT | Optical-flow model behind Flow-EPE on the RoboTwin rollouts; imported from the FlowWAM checkout. | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | BSD-3-Clause |
| LPIPS | Perceptual distance in the RoboTwin metric stack; installed from PyPI by the `eval` extra. | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) | BSD-2-Clause |
| WorldArena 1.0 evaluator | The eight core metrics and the EWMScore-local-8 aggregate behind the zero-shot table; checked out separately and pinned to a fixed revision. | [worldarena/WorldArena](https://github.com/worldarena/WorldArena) | No licence file upstream |
| DreamGen / DreamGenBench | The GR1 clips and the official instruction-following and PA judging code that DreamGenBench adapts. | [NVIDIA/GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) | Apache-2.0 |
| EWMBench | The AgiBot metric suite: Motion, Semantics, DYN, HSD, nDTW, Scene and Overall. | [AgibotTech/EWMBench](https://github.com/AgibotTech/EWMBench) | No licence file upstream |
| PBench Robot | Physical-commonsense QA pairs over generated rollouts, scored by a Qwen-VL judge. | [nvidia/PBench](https://huggingface.co/datasets/nvidia/PBench) | CC BY-NC 4.0 (non-commercial) |

## Judges

The instruction-following judges call model endpoints rather than library code.
Qwen-IF sends the template in `data/prompts/qwen_if.txt` to an
OpenAI-compatible server, which is normally the local Qwen2.5-VL-7B-Instruct
weights above; Gemini-IF reaches hosted Gemini models, through the gateway
described in [`installation.md`](installation.md) when one is configured, and
the MLR audit prompt `data/prompts/vlm_audit.txt` re-checks flagged events
against the same kind of endpoint. The SDKs (`openai`, `google-genai`) come
with the `eval` extra, and no credentials are stored in the repository.

## What this repository redistributes

Nothing from the projects listed above. The repository ships code, split files,
per-clip metadata and prompt templates; the two backbone checkouts are dropped
by the root `.gitignore`, the downloaded weights land under the gitignored
`checkpoints/` root, and the datasets stay under the data roots named in
`.env`. The comparison models quoted in the results tables
(CogVideoX1.5-5B-I2V, Wan2.2-I2V-A14B, Cosmos-Predict2-2B) are obtained from
their own projects and scored zero-shot in the same way.

Two rows above have no upstream licence file, and PBench is published for
non-commercial use; check the upstream page of a component before reusing
anything derived from it.

## See also

- [`../third_party/README.md`](../third_party/README.md) — what each backbone
  contributes and how the mirroring is wired into the recipe.
- [`installation.md`](installation.md) — environments, extras and the
  environment variables the code reads.
- [`data_preparation.md`](data_preparation.md) — the datasets, their splits
  and where the prepared files land.
- [`checkpoints.md`](checkpoints.md) — the download script and the checkpoint
  variables.
- [`evaluation.md`](evaluation.md) — the metric stacks above in the context
  of the protocols they implement.
