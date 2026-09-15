# eveworld/ — Method Core Package

Python package implementing the EVEWorld stack: the IGR + TIA training pipeline that supervises target-instance evolution, the Model Laziness Rate (MLR) diagnostics, generation/evaluation tooling, domain and backbone transfer ports, and the alternative designs analyzed as negative controls in the paper appendix.

The trainer entry points are exposed lazily from [`__init__.py`](__init__.py) (`EveCausalTrainer`, `EveLadLoraTrainer`, `EveFrontierTrainer`); GigaTrain configs select them through the `runners` field, e.g. `runners = ["eveworld.EveCausalTrainer"]`.

## Paper mapping

| Paper component | Code |
|---|---|
| IGR — Instance-Guided Restoration (Sec. 4.1, App. B.1) | [`pipeline/t4g_aug_trainer.py`](pipeline/t4g_aug_trainer.py), [`pipeline/t4g_aug_paste.py`](pipeline/t4g_aug_paste.py), [`flowwam_port/igr_paste.py`](flowwam_port/igr_paste.py) |
| TIA — Temporal Instance Alignment (Sec. 4.2, App. B.2–B.3) | [`pipeline/t4g_corr_trainer.py`](pipeline/t4g_corr_trainer.py), [`tia_transport/`](tia_transport/) |
| Joint IGR + TIA trainer (App. B.4 objective) | [`pipeline/t4g_joint_trainer.py`](pipeline/t4g_joint_trainer.py) + [`pipeline/t4g_joint_config.py`](pipeline/t4g_joint_config.py) |
| MLR and process-faithfulness metrics (Sec. 3, App. A) | [`diagnosis/`](diagnosis/) |
| DreamGenBench / WorldArena 1.0 evaluation (Sec. 5) | [`evaluation/`](evaluation/) and the per-benchmark pipelines in repo-level [`benchmarks/`](../benchmarks/) |
| Transfer: EWMBench/AgiBot, RoboTwin-FlowWAM (Sec. 5.3, App. E) | [`agibot/`](agibot/), [`flowwam_port/`](flowwam_port/) |
| Alternative designs (App. D): PhysicsLatent, EAG, LAD-LoRA, Causal Frontier, ICH-D | [`alternatives/physlatent/`](alternatives/physlatent/), [`method/`](method/), [`pipeline/t4g_ich_*.py`](pipeline/) |
| Frozen data splits for controlled comparisons (App. C) | [`data_curation/splits/`](data_curation/splits/) |

## Contents

| Path | What it does |
|---|---|
| [`pipeline/`](pipeline/) | Main training pipeline: GroundingDINO-based target detection, paste-augmentation asset preparation, weight maps, the corr/aug/joint trainers and their configs, plus the ICH-D inference-time deletion variant |
| [`tia_transport/`](tia_transport/) | TIA cross-frame feature transport: transport trainer/config and its multi-seed controlled evaluation scripts |
| [`method/`](method/) | Training stack for the alternative (negative-control) designs — Causal Frontier, LAD-LoRA, EAG guidance, CPC-style contrastive prototype — with configs, losses, the latent action model, and the held-out experiment harness. See [`method/README.md`](method/README.md) |
| [`alternatives/physlatent/`](alternatives/physlatent/) | PhysicsLatent process-token adapter; also the base trainer (`PhysicsLatentGigaWorld0Trainer`) that several `method/` trainers inherit |
| [`diagnosis/`](diagnosis/) | Process metrics (premature completion, motion-before-contact, teleport, laziness rate) computed from extracted events |
| [`evaluation/`](evaluation/) | Generation/evaluation dispatchers, multi-seed aggregation with bootstrap CIs, instruction-following and MLR scoring helpers |
| [`data_curation/`](data_curation/) | GigaTrain dataset filters (`filters.py`) and frozen split manifests (`splits/frontier_20260715/`) |
| [`agibot/`](agibot/) | AgiBot-distribution data preparation, detection, augmentation, and training configs for the EWMBench transfer |
| [`flowwam_port/`](flowwam_port/) | Port of IGR + TIA to the FlowWAM backbone for the RoboTwin experiment (dataset, trainer, TIA adapter/probe) |
| [`common/`](common/) | Shared shell environment (`env.sh`) sourced by the wrappers |

## Usage

- **Environment.** Most shell wrappers source [`common/env.sh`](common/env.sh), which defines:
  - `GAGI_ROOT` — cluster data/output root (default `/data/datasets/gagi`; **adapt to your site**),
  - `GW0_MODEL_DIR` — GigaWorld-0 pretrained checkpoint directory,
  - `GR1_DATA_ROOT` — packed GR1 fine-tuning data,
  - `EVE_OUT` / `EVE_VIDEO_ROOT` — EVE output root and generated-video input root.
  The `launch_*`/`kjob_*` wrappers additionally expect `EVEWORLD_ROOT` to point at the repository root (they resolve payloads such as `benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh` relative to it).
- **Training.** Training is config-driven through the GigaTrain launcher: a config module under `pipeline/` or `method/configs/` sets `runners = [...]`, and a wrapper submits the job. Example (main-method joint trainer):

  ```bash
  export EVEWORLD_ROOT=/path/to/EVEWorld
  # config: eveworld/pipeline/t4g_joint_config.py (runner: T4GJointTrainer)
  bash benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
      BASE_CONFIG_MODULE=eveworld.pipeline.t4g_joint_config
  ```

- **Cluster-job caveat.** `kjob_*.sh` and `launch_*.sh` are SLURM-style wrappers for our cluster (SBATCH headers, pod polling); their absolute paths and the scheduler integration must be adapted to your environment. They are the reference protocol, not portable tooling.
- **Tests.** CPU-runnable unit tests for the frontier mechanics:

  ```bash
  python -m unittest eveworld.method.tests.test_eve_frontier
  ```

## Notes

- Depends on the vendored snapshots at the repo root: [`giga-models/`](../giga-models/) (training framework, installed as `giga_models`/`giga_train`) and [`giga-world-0/`](../giga-world-0/) (backbone, configs, `scripts/train.py`). Trainers in `method/` additionally subclass the PhysicsLatent trainer from [`alternatives/physlatent/`](alternatives/physlatent/), imported as `eveworld.alternatives.physlatent`.
- Heavy dependencies: PyTorch, diffusers, transformers, peft, DeepSpeed, accelerate; GroundingDINO for the detection stages; Qwen2.5-VL / Gemini judges for instruction-following evaluation. See `docs/ENVIRONMENT.md`.
- Data prerequisites: the packed GR1 fine-tuning dataset, GigaWorld-0 pretrained weights, and (for controlled runs) the frozen manifests under [`data_curation/splits/`](data_curation/splits/).
