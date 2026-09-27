# Installation

EVEWorld is a Python package with a `src/` layout plus three conda
environments that isolate the two backbones from the evaluation stack. The
library itself only needs numpy and PyYAML; the heavy dependencies come from
the extras and from the environment files under `envs/`.

## Environments

| Environment | File | Purpose |
|---|---|---|
| `gigaworld` | [`envs/gigaworld.yaml`](../envs/gigaworld.yaml) | GigaWorld-0 training and generation: DreamGenBench, AgiBot / EWMBench, WorldArena and PBench arms. |
| `flowwam` | [`envs/flowwam.yaml`](../envs/flowwam.yaml) | FlowWAM training, inference and the RoboTwin metric stack. |
| `eveworld-eval` | [`envs/evaluation.yaml`](../envs/evaluation.yaml) | Model Laziness Rate, the instruction-following judges and the per-benchmark harnesses. |

Create the one you need and install the package into it:

```bash
conda env create -f envs/gigaworld.yaml
conda activate gigaworld
pip install -e ".[train,eval]"
```

```bash
conda env create -f envs/flowwam.yaml
conda activate flowwam
pip install -e ".[train,eval]"
```

```bash
conda env create -f envs/evaluation.yaml
conda activate eveworld-eval
pip install -e ".[eval]"
```

All three files pin Python 3.11 and torch 2.11.0. Nothing is copied into the
environment: `pip install -e .` reads `pyproject.toml`, finds the package under
`src/` (`[tool.setuptools.packages.find] where = ["src"]`) and puts a link on
`sys.path`, so edits to `src/eveworld/` take effect without reinstalling.

The conda environment name and the repository checkout are independent. The
environment records where *its* packages live; the checkout is wherever the
repository was cloned, and every script resolves paths relative to the
repository root rather than to the environment. Two checkouts can share one
environment, and one checkout can be evaluated from any environment that can
import `eveworld`, provided the backbone it needs is installed there.

## Extras

The extras keep the three environments from installing everything at
once. The RoboTwin stack runs on CPU for its offline metrics; the
GigaWorld-0 recipe needs the full training stack.

| Extras | Install | Contents |
|---|---|---|
| base | `pip install -e .` | numpy, PyYAML, OmegaConf, einops, pillow, tqdm, rich. |
| `train` | `pip install -e ".[train]"` | torch, torchvision, diffusers, transformers, accelerate, peft, safetensors, datasets, tensorboard, fairscale. |
| `eval` | `pip install -e ".[eval]"` | torch, torchvision, OpenCV, SciPy, scikit-image, scikit-learn, LPIPS, Segment Anything, PyAV, decord, imageio, pandas, matplotlib, the judge SDKs. |
| `dev` | `pip install -e ".[dev]"` | pytest and pre-commit; see [`../CONTRIBUTING.md`](../CONTRIBUTING.md). |

## Backbones

Neither backbone is vendored. Clone the one you train on before importing the
integrations:

```bash
bash scripts/setup/clone_gigaworld.sh   # -> third_party/giga-world-0
bash scripts/setup/clone_flowwam.sh     # -> third_party/FlowWAM
```

The GigaWorld-0 recipe additionally installs the two upstream framework
packages, which are not on PyPI:

```bash
pip install git+https://github.com/open-gigaai/giga-train.git
pip install git+https://github.com/open-gigaai/giga-datasets.git
```

FlowWAM training starts from the released FlowWAM Stage-1 checkpoint plus
the Wan2.2-TI2V-5B base weights; point `FLOWWAM_ROOT` at the checkout only
if it does not live under `third_party/FlowWAM`. Both clone directories are
gitignored, and neither tree nor its weights is redistributed here — see
[`third_party.md`](third_party.md).

## Reference hardware

All training and generation runs in the paper use a node with 8x H20Z GPUs,
PyTorch 2.11.0 built against CUDA 12.8, and bf16 autocast. The full GigaWorld-0
recipe (effective batch 64, 480 x 768, 93 frames at 16 FPS) fits on that node
with a per-GPU batch of 8 and gradient accumulation; the rank-32 FlowWAM LoRA
fits comfortably on a single card at batch 8. CAME-8bit is used for the
GigaWorld-0 optimiser and falls back to AdamW with a logged warning when
`came_pytorch` is not installed, which is enough for a smoke run but not for
reproducing the published numbers.

Check the CUDA build before a long run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Evaluation-only dependencies

The MLR detector needs GroundingDINO and SAM2, neither of which is on
PyPI, plus a SAM2.1 checkpoint for the tracker; RAFT (Flow-EPE) and LPIPS
come from the `eval` extra. GroundingDINO and SAM2 are installed from
source — see [`third_party.md`](third_party.md) for the projects and
[`evaluation.md`](evaluation.md) for the protocol they implement. The Qwen-IF
judge talks to an OpenAI-compatible endpoint (a local Qwen3-VL server or a
hosted model), and Gemini-IF goes through the gateway described in the same
page.

## Environment variables

Everything in [`../.env.example`](../.env.example) is optional; copy it to
`.env` and fill in the paths you have.

| Variable | Meaning |
|---|---|
| `HF_HOME` | Cache for Hugging Face downloads (GigaWorld-0 weights, Qwen3-VL, SAM2, RAFT). |
| `EVEWORLD_CHECKPOINT_ROOT` | Where [`scripts/setup/download_models.sh`](../scripts/setup/download_models.sh) writes the released checkpoints. |
| `GW0_MODEL_DIR` | Directory of the GigaWorld-0 video-pretrain weights; without it the default under `EVEWORLD_CHECKPOINT_ROOT` is used. |
| `GIGA_MODELS_DIR` | Location of the GigaModels package when it is not installed into the environment. |
| `GIGA_MODELS_CACHE` | Cache directory for the GigaModels weight loader. |
| `GIGA_MODELS_REPO_CACHE` | Cache directory for GigaModels checkouts. |
| `FLOWWAM_ROOT` | FlowWAM checkout, when it is not `third_party/FlowWAM`. |
| `GROUNDING_DINO_CONFIG` | GroundingDINO config file used by the locator. |
| `GROUNDING_DINO_WEIGHTS` | GroundingDINO checkpoint used by the locator. |
| `SAM2_CHECKPOINT` | SAM2.1 checkpoint used by the tracker and the occlusion masks. |
| `OPENAI_API_KEY` | OpenAI-compatible key for the Qwen-IF judge; `DASHSCOPE_API_KEY` is used first when set. |
| `GOOGLE_API_KEY` | Key for the Gemini judge used by the Gemini-IF arm; `GEMINI_API_KEY` is used first when set. |
| `QWEN_IF_MODEL` | Local Qwen3-VL model directory for the Qwen-IF judge. |
| `DREAMGEN_DATA_ROOT` | DreamGen GR1 clips and DreamGenBench inputs. |
| `AGIBOT_DATA_ROOT` | AgiBot clip tree of the cross-distribution training split. |
| `EWMBENCH_DATA_ROOT` | EWMBench checkout the AgiBot leakage guard reads the `GT/` episodes from; without it the built-in episode list is used. |
| `ROBOTWIN_DATA_ROOT` | RoboTwin episodes and the FlowWAM inputs. |
| `WORLDARENA_DATA_ROOT` | WorldArena 1.0 manifest and videos. |

The judges add a few runtime variables that are set on the serving side rather
than in `.env`. The Qwen judge reads `QWEN_BASE_URL` and takes its key from
`DASHSCOPE_API_KEY` and then `OPENAI_API_KEY`. The Gemini judge reads
`GEMINI_BASE_URL` and then `DIFROST_GENAI_BASE_URL`, with `DIFROST_API_TOKEN`,
`DIFROST_HOST` and `DIFROST_THINKING_LEVEL`, takes its key from
`GEMINI_API_KEY` and then `GOOGLE_API_KEY`, and uses the public Gemini API when
no gateway is configured. `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` switch
the loaders to a warm cache, and `CUDA_VISIBLE_DEVICES` selects the cards for
every script.

## Verifying the setup

```bash
python scripts/setup/check_environment.py
```

The checker prints one table per group — Python version, required packages,
optional packages, environment variables and the two third-party checkouts —
marks each row as present or missing, and exits non-zero when something
required is absent. Rows that are marked optional only disable the parts of the
suite that need them; for example, a missing `SAM2_CHECKPOINT` still allows MLR
runs with `--occlusion-rule none`, and a missing judge key only disables that
judge. Run it inside the environment you intend to use; the three environments
report differently by design.
